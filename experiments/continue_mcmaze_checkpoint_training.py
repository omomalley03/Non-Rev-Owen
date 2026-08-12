"""Continue an MC Maze checkpoint in-place and append checkpoint manifests."""

import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model.py").is_file())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from data import gaussian_smooth, load_mcmaze_cached, make_windows, soft_normalize, train_val_split
from loss import loss_fn, non_reversibility_components, _batch_rms_normalize
from decoders.predict_mcmaze_velocity import build_model_from_checkpoint, resolve_run_dir
from train import (
    _checkpoint_payload,
    _mean_plane_zeta,
    _metric_value,
    _resolve_best_checkpoint_metric,
    _resolve_threshold_checkpoints,
    _threshold_metric_label,
)
from visualize_loss import plot_loss_curve


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(cfg: Config):
    spikes_raw, bin_width_s, trial_info, time_index_s, _ = load_mcmaze_cached(cfg.nwb_path, cfg.bin_ms)
    sigma_samples = round((cfg.sigma_ms * 1e-3) / bin_width_s)
    X_smooth = gaussian_smooth(spikes_raw, sigma_samples)
    softnorm = getattr(cfg, "softnorm_method", "none")
    if softnorm and softnorm != "none":
        X_smooth = soft_normalize(X_smooth, method=softnorm)

    temporal_context_bins = (
        getattr(cfg, "temporal_context_bins", 0)
        if getattr(cfg, "temporal_filters", 0) > 0
        else 0
    )
    windows = make_windows(
        X_smooth,
        trial_info,
        time_index_s,
        bin_width_s,
        strategy=getattr(cfg, "window_strategy", "trial_aligned"),
        window_size=cfg.window_size,
        align_field=getattr(cfg, "align_field", "move_onset_time"),
        pre_ms=getattr(cfg, "pre_ms", 100),
        context_bins=temporal_context_bins,
    )
    if getattr(cfg, "split", "dataset") == "random":
        trial_info = trial_info.drop(columns=["split"], errors="ignore")
    train_ds, val_ds = train_val_split(windows, trial_info, cfg.val_split, cfg.seed)
    return spikes_raw, train_ds, val_ds


def mean_train_epoch(model, loader, optimizer, cfg: Config, device, lambda_scale: float):
    model.train()
    losses, ss, c_pluses, zetas, mean_plane_zetas, regs = [], [], [], [], [], []
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        F = model(batch)
        F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
        loss = loss_fn(F, cfg=cfg, training=True, lambda_scale=lambda_scale)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            F_norm = _batch_rms_normalize(F)
            c_minus, c_plus, zeta = non_reversibility_components(F_norm, "mean")
            mean_plane_zeta = _mean_plane_zeta(F_norm)
        losses.append(loss.item())
        ss.append(c_minus.item())
        c_pluses.append(c_plus.item())
        zetas.append(zeta.item())
        mean_plane_zetas.append(mean_plane_zeta.item())
        regs.append(loss.item() + c_minus.item())
    return tuple(float(np.mean(values)) for values in (losses, ss, c_pluses, zetas, mean_plane_zetas, regs))


def mean_val_epoch(model, loader, cfg: Config, device, lambda_scale: float):
    model.eval()
    losses, ss, c_pluses, zetas, mean_plane_zetas, regs = [], [], [], [], [], []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            if batch.shape[0] < 2:
                continue
            F = model(batch)
            F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
            loss, _info = loss_fn(
                F,
                cfg=cfg,
                training=False,
                lambda_scale=lambda_scale,
                return_components=True,
            )
            F_norm = _batch_rms_normalize(F)
            c_minus, c_plus, zeta = non_reversibility_components(F_norm, "mean")
            mean_plane_zeta = _mean_plane_zeta(F_norm)
            losses.append(loss.item())
            ss.append(c_minus.item())
            c_pluses.append(c_plus.item())
            zetas.append(zeta.item())
            mean_plane_zetas.append(mean_plane_zeta.item())
            regs.append(loss.item() + c_minus.item())
    return tuple(float(np.mean(values)) for values in (losses, ss, c_pluses, zetas, mean_plane_zetas, regs))


def existing_thresholds(path: str) -> set[float]:
    if not os.path.isfile(path):
        return set()
    with open(path, newline="") as f:
        return {float(row["threshold"]) for row in csv.DictReader(f)}


def ensure_threshold_manifest(path: str):
    if os.path.isfile(path):
        return
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow([
            "metric", "threshold", "epoch", "val_s", "val_zeta",
            "val_mean_plane_zeta", "val_c_plus",
            "val_loss", "train_loss", "train_s", "train_zeta",
            "train_mean_plane_zeta", "path",
        ])


def ensure_epoch_manifest(path: str):
    if os.path.isfile(path):
        return
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow([
            "interval", "epoch", "val_s", "val_zeta",
            "val_mean_plane_zeta", "val_c_plus",
            "val_loss", "train_loss", "train_s", "train_zeta",
            "train_mean_plane_zeta", "path",
        ])


def merge_continuation_log(canonical_log_path: str, continuation_log_path: str) -> tuple[int, int]:
    rows_by_epoch: dict[int, dict[str, str]] = {}
    fieldnames: list[str] | None = None

    if os.path.isfile(canonical_log_path):
        with open(canonical_log_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row.get("epoch"):
                    rows_by_epoch[int(float(row["epoch"]))] = row

    with open(continuation_log_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header in continuation log {continuation_log_path!r}")
        if fieldnames is None:
            fieldnames = reader.fieldnames
        elif reader.fieldnames != fieldnames:
            raise ValueError(
                f"Continuation log header does not match canonical log: "
                f"{reader.fieldnames} != {fieldnames}"
            )

        added = 0
        for row in reader:
            if not row.get("epoch"):
                continue
            epoch = int(float(row["epoch"]))
            if epoch not in rows_by_epoch:
                added += 1
            rows_by_epoch[epoch] = row

    if fieldnames is None:
        raise ValueError("No log headers found")
    with open(canonical_log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in sorted(rows_by_epoch):
            writer.writerow(rows_by_epoch[epoch])

    max_epoch = max(rows_by_epoch) if rows_by_epoch else 0
    return added, max_epoch


def save_payload(model, cfg, epoch, path, payload):
    torch.save(_checkpoint_payload(model, cfg, epoch, **payload), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="Existing run dir. Defaults to newest completed MC Maze run.")
    parser.add_argument("--checkpoint", default="epoch_0150.pt",
                        help="Checkpoint filename under checkpoints/ or explicit path.")
    parser.add_argument("--extra-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(run_dir, "checkpoints", ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path!r}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]
    if args.seed is not None:
        cfg.seed = int(args.seed)
    set_seed(int(cfg.seed))

    start_epoch = int(ckpt.get("epoch") or 0)
    final_epoch = start_epoch + int(args.extra_epochs)
    cfg.epochs = max(int(getattr(cfg, "epochs", final_epoch)), final_epoch)
    print(f"Using run: {run_dir}")
    print(f"Continuing from: {ckpt_path} (epoch {start_epoch})")
    print(f"Continuing through epoch {final_epoch}")

    print("Loading data...")
    spikes_raw, train_ds, val_ds = prepare_data(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0)
    print(f"Train={len(train_ds)} val={len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(cfg, ckpt["model_state_dict"], spikes_raw.shape[0]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.T_0,
        T_mult=cfg.T_mult,
    )

    threshold_metric, checkpoint_thresholds = _resolve_threshold_checkpoints(cfg)
    best_checkpoint_metric = _resolve_best_checkpoint_metric(cfg, threshold_metric)
    epoch_interval = int(getattr(cfg, "checkpoint_every_epochs", 0) or 0)
    threshold_manifest = os.path.join(cfg.ckpt_dir, f"val_{threshold_metric}_checkpoints.csv")
    epoch_manifest = os.path.join(cfg.ckpt_dir, "epoch_checkpoints.csv")
    ensure_threshold_manifest(threshold_manifest)
    ensure_epoch_manifest(epoch_manifest)
    saved_thresholds = existing_thresholds(threshold_manifest)

    best_score = _metric_value(
        best_checkpoint_metric,
        float(ckpt.get("val_s", float("-inf"))),
        float(ckpt.get("val_zeta", float("-inf"))),
        float(ckpt.get("val_mean_plane_zeta", float("-inf"))),
    )
    log_path = os.path.join(cfg.out_dir, f"continue_{start_epoch:04d}_to_{final_epoch:04d}.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_loss",
            "train_s", "val_s",
            "train_c_plus", "val_c_plus",
            "train_zeta", "val_zeta",
            "train_mean_plane_zeta", "val_mean_plane_zeta",
            "train_reg", "val_reg",
        ])

    for epoch in range(start_epoch + 1, final_epoch + 1):
        lambda_scale = 1.0
        train_loss, train_s, train_c_plus, train_zeta, train_mean_plane_zeta, train_reg = mean_train_epoch(
            model,
            train_loader,
            optimizer,
            cfg,
            device,
            lambda_scale,
        )
        scheduler.step(epoch)
        val_loss, val_s, val_c_plus, val_zeta, val_mean_plane_zeta, val_reg = mean_val_epoch(
            model,
            val_loader,
            cfg,
            device,
            lambda_scale,
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss,
                train_s, val_s,
                train_c_plus, val_c_plus,
                train_zeta, val_zeta,
                train_mean_plane_zeta, val_mean_plane_zeta,
                train_reg, val_reg,
            ])

        payload = {
            "val_loss": val_loss,
            "val_s": val_s,
            "val_c_plus": val_c_plus,
            "val_zeta": val_zeta,
            "val_mean_plane_zeta": val_mean_plane_zeta,
            "train_loss": train_loss,
            "train_s": train_s,
            "train_c_plus": train_c_plus,
            "train_zeta": train_zeta,
            "train_mean_plane_zeta": train_mean_plane_zeta,
            "continued_from": ckpt_path,
        }

        best_score_candidate = _metric_value(best_checkpoint_metric, val_s, val_zeta, val_mean_plane_zeta)
        if best_score_candidate > best_score:
            best_score = best_score_candidate
            save_payload(
                model,
                cfg,
                epoch,
                os.path.join(cfg.ckpt_dir, "best.pt"),
                {
                    **payload,
                    "checkpoint_selection": f"best_val_{best_checkpoint_metric}",
                    "val_checkpoint_metric": threshold_metric,
                    "val_best_checkpoint_metric": best_checkpoint_metric,
                    "val_checkpoint_score": best_score_candidate,
                },
            )

        score = _metric_value(threshold_metric, val_s, val_zeta, val_mean_plane_zeta)
        for threshold in checkpoint_thresholds:
            if threshold in saved_thresholds:
                continue
            if score >= threshold:
                ckpt_name = f"val_{threshold_metric}_{threshold:g}.pt"
                threshold_path = os.path.join(cfg.ckpt_dir, ckpt_name)
                save_payload(
                    model,
                    cfg,
                    epoch,
                    threshold_path,
                    {
                        **payload,
                        "checkpoint_selection": f"val_{threshold_metric}_threshold",
                        "val_checkpoint_metric": threshold_metric,
                        "val_checkpoint_threshold": threshold,
                        f"val_{threshold_metric}_threshold": threshold,
                    },
                )
                saved_thresholds.add(threshold)
                with open(threshold_manifest, "a", newline="") as f:
                    csv.writer(f).writerow([
                        threshold_metric,
                        threshold,
                        epoch,
                        val_s,
                        val_zeta,
                        val_mean_plane_zeta,
                        val_c_plus,
                        val_loss,
                        train_loss,
                        train_s,
                        train_zeta,
                        train_mean_plane_zeta,
                        threshold_path,
                    ])
                print(
                    f"Saved val-{_threshold_metric_label(threshold_metric)} checkpoint {ckpt_name}: "
                    f"epoch={epoch} val_{threshold_metric}={score:.4f}"
                )

        if epoch_interval > 0 and epoch % epoch_interval == 0:
            epoch_name = f"epoch_{epoch:04d}.pt"
            epoch_path = os.path.join(cfg.ckpt_dir, epoch_name)
            save_payload(
                model,
                cfg,
                epoch,
                epoch_path,
                {
                    **payload,
                    "checkpoint_selection": "epoch_interval",
                    "checkpoint_every_epochs": epoch_interval,
                },
            )
            with open(epoch_manifest, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch_interval,
                    epoch,
                    val_s,
                    val_zeta,
                    val_mean_plane_zeta,
                    val_c_plus,
                    val_loss,
                    train_loss,
                    train_s,
                    train_zeta,
                    train_mean_plane_zeta,
                    epoch_path,
                ])
            print(
                f"Saved epoch checkpoint {epoch_name}: "
                f"val_s={val_s:.4f} val_zeta={val_zeta:.4f} "
                f"mean_plane_zeta={val_mean_plane_zeta:.4f}"
            )

        if epoch == start_epoch + 1 or epoch % 10 == 0 or epoch == final_epoch:
            print(
                f"Epoch {epoch:3d}/{final_epoch}  "
                f"train loss={train_loss:.4f}  val loss={val_loss:.4f}  "
                f"S[mean]={val_s:.4f}  zeta={val_zeta:.4f}  "
                f"mean_plane_zeta={val_mean_plane_zeta:.4f}"
            )

    print(f"Continuation log: {log_path}")
    canonical_log_path = os.path.join(cfg.out_dir, "log.csv")
    try:
        added, max_epoch = merge_continuation_log(canonical_log_path, log_path)
        print(
            f"Updated canonical log: {canonical_log_path} "
            f"through epoch {max_epoch} ({added} new continuation rows)"
        )
        plot_loss_curve(run_dir)
    except Exception as exc:
        print(f"Could not update canonical loss curve from continuation log: {exc}")


if __name__ == "__main__":
    main()
