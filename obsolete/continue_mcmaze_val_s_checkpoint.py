"""Continue an MC Maze embedder until a missing validation-S checkpoint is reached."""

import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import gaussian_smooth, load_mcmaze_cached, make_windows, soft_normalize, train_val_split
from loss import _batch_rms_normalize, loss_fn, non_reversibility_S
from predict_mcmaze_velocity import build_model_from_checkpoint, resolve_run_dir


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_name(threshold: float) -> str:
    return f"val_s_{threshold:g}.pt"


def manifest_has_threshold(path: str, threshold: float) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if abs(float(row["threshold"]) - threshold) < 1e-12:
                return True
    return False


def ensure_manifest_header(path: str):
    if os.path.isfile(path):
        return
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow([
            "threshold", "epoch", "val_s", "val_loss", "train_loss", "train_s", "path",
        ])


def append_manifest(path, threshold, epoch, val_s, val_loss, train_loss, train_s, ckpt_path):
    ensure_manifest_header(path)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            threshold, epoch, val_s, val_loss, train_loss, train_s, ckpt_path,
        ])


def prepare_data(cfg):
    spikes_raw, bin_width_s, trial_info, time_index_s, _ = load_mcmaze_cached(cfg.nwb_path, cfg.bin_ms)
    sigma_samples = round((cfg.sigma_ms * 1e-3) / bin_width_s)
    X_smooth = gaussian_smooth(spikes_raw, sigma_samples)
    softnorm = getattr(cfg, "softnorm_method", "none")
    if softnorm and softnorm != "none":
        X_smooth = soft_normalize(X_smooth, method=softnorm)
    windows = make_windows(
        X_smooth,
        trial_info,
        time_index_s,
        bin_width_s,
        strategy=getattr(cfg, "window_strategy", "trial_aligned"),
        window_size=cfg.window_size,
        align_field=getattr(cfg, "align_field", "move_onset_time"),
        pre_ms=getattr(cfg, "pre_ms", 100),
    )
    if getattr(cfg, "split", "dataset") == "random":
        trial_info = trial_info.drop(columns=["split"], errors="ignore")
    train_ds, val_ds = train_val_split(windows, trial_info, cfg.val_split, cfg.seed)
    return spikes_raw, train_ds, val_ds


def mean_train_epoch(model, loader, optimizer, cfg, device):
    model.train()
    losses = []
    scores = []
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        F = model(batch)
        F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
        loss = loss_fn(F, cfg=cfg, training=True, lambda_scale=1.0)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            s = non_reversibility_S(_batch_rms_normalize(F), "mean").item()
        losses.append(loss.item())
        scores.append(s)
    return float(np.mean(losses)), float(np.mean(scores))


def mean_val_epoch(model, loader, cfg, device):
    model.eval()
    losses = []
    scores = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            if batch.shape[0] < 2:
                continue
            F = model(batch)
            F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
            loss = loss_fn(F, cfg=cfg, training=False, lambda_scale=1.0)
            s = non_reversibility_S(_batch_rms_normalize(F), "mean").item()
            losses.append(loss.item())
            scores.append(s)
    return float(np.mean(losses)), float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory or rank.")
    parser.add_argument("--checkpoint", default="best.pt",
                        help="Checkpoint filename under checkpoints/ or explicit path.")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--max-extra-epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)

    run_dir = resolve_run_dir(args.run)
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(run_dir, "checkpoints", ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path!r}")

    manifest_path = os.path.join(run_dir, "checkpoints", "val_s_checkpoints.csv")
    if manifest_has_threshold(manifest_path, args.threshold):
        print(f"Manifest already contains threshold {args.threshold:g}; nothing to do.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    start_epoch = int(ckpt.get("epoch") or 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using run: {run_dir}")
    print(f"Continuing from: {ckpt_path} (epoch {start_epoch})")
    print(f"Target validation S threshold: {args.threshold:g}")
    print("Loading data...")
    spikes_raw, train_ds, val_ds = prepare_data(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0)
    print(f"Train={len(train_ds)} val={len(val_ds)}")

    model = build_model_from_checkpoint(cfg, ckpt["model_state_dict"], spikes_raw.shape[0]).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(args.lr if args.lr is not None else cfg.lr),
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult
    )

    best_val_loss = float("inf")
    log_path = os.path.join(run_dir, "outputs", f"continue_to_val_s_{args.threshold:g}.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["extra_epoch", "global_epoch", "train_loss", "val_loss", "train_s", "val_s"])

    for extra_epoch in range(1, args.max_extra_epochs + 1):
        global_epoch = start_epoch + extra_epoch
        train_loss, train_s = mean_train_epoch(model, train_loader, optimizer, cfg, device)
        scheduler.step(global_epoch)
        val_loss, val_s = mean_val_epoch(model, val_loader, cfg, device)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([extra_epoch, global_epoch, train_loss, val_loss, train_s, val_s])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "epoch": global_epoch,
                    "val_loss": val_loss,
                    "val_s": val_s,
                    "train_loss": train_loss,
                    "train_s": train_s,
                    "continued_from": ckpt_path,
                },
                os.path.join(run_dir, "checkpoints", f"continue_best_to_s_{args.threshold:g}.pt"),
            )

        if extra_epoch == 1 or extra_epoch % 10 == 0 or val_s >= args.threshold:
            print(
                f"Continue epoch {extra_epoch:3d}/{args.max_extra_epochs} "
                f"global={global_epoch} train loss={train_loss:.4f} "
                f"val loss={val_loss:.4f} S[mean]={val_s:.4f}"
            )

        if val_s >= args.threshold:
            out_ckpt = os.path.join(run_dir, "checkpoints", checkpoint_name(args.threshold))
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "epoch": global_epoch,
                    "val_s_threshold": args.threshold,
                    "val_loss": val_loss,
                    "val_s": val_s,
                    "train_loss": train_loss,
                    "train_s": train_s,
                    "continued_from": ckpt_path,
                },
                out_ckpt,
            )
            append_manifest(
                manifest_path,
                args.threshold,
                global_epoch,
                val_s,
                val_loss,
                train_loss,
                train_s,
                out_ckpt,
            )
            print(f"Saved {out_ckpt}")
            print(f"Appended {manifest_path}")
            return

    raise RuntimeError(
        f"Did not reach validation S>={args.threshold:g} after {args.max_extra_epochs} extra epochs. "
        f"See {log_path}"
    )


if __name__ == "__main__":
    main()
