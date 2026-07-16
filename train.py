import os
import csv
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from loss import loss_fn, non_reversibility_components, _batch_rms_normalize


def _make_loader(dataset, batch_size: int, shuffle: bool, drop_last: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )


def _fmt_threshold(value: float) -> str:
    return f"{value:g}"


def _checkpoint_payload(model, cfg: Config, epoch: int, **metrics):
    payload = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "epoch": epoch,
    }
    payload.update(metrics)
    return payload


def _resolve_threshold_checkpoints(cfg: Config):
    metric = str(getattr(cfg, "val_checkpoint_metric", "zeta") or "zeta").strip().lower()
    if metric in {"ζ", "zetas"}:
        metric = "zeta"
    if metric not in {"zeta", "s"}:
        raise ValueError(f"VAL_CHECKPOINT_METRIC must be 'zeta' or 's', got {metric!r}")

    generic = tuple(float(x) for x in getattr(cfg, "val_checkpoint_thresholds", ()) or ())
    zeta_thresholds = tuple(float(x) for x in getattr(cfg, "val_zeta_checkpoint_thresholds", ()) or ())
    s_thresholds = tuple(float(x) for x in getattr(cfg, "val_s_checkpoint_thresholds", ()) or ())

    thresholds = generic if generic else (s_thresholds if metric == "s" else zeta_thresholds)
    return metric, sorted(set(float(x) for x in thresholds))


def _threshold_metric_value(metric: str, mean_val_s: float, mean_val_zeta: float) -> float:
    return mean_val_zeta if metric == "zeta" else mean_val_s


def _threshold_metric_label(metric: str) -> str:
    return "ζ" if metric == "zeta" else "S"


# Unused learned plane-predictor adversary removed from the active training path
# while the current sweep focuses on BT, plane-aware BT, and block-CCA.


def train(model, train_ds, val_ds, cfg: Config, loss_function=loss_fn) -> dict:
    """Run the full training loop.

    Returns a history dict with keys:
      'train_loss' : list of mean train loss per epoch
      'val_loss'   : list of mean val loss per epoch
      'best_val_zeta' : float, selected checkpoint score
      'best_val_loss' : float, minimum validation loss seen during training
      'elapsed_s'     : total wall-clock seconds
    """
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = _make_loader(train_ds, cfg.batch_size, shuffle=True,  drop_last=True)
    val_loader   = _make_loader(val_ds,   cfg.batch_size, shuffle=False, drop_last=False)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult
    )

    # active regularizers (lambda > 0) tracked individually for the loss curve
    reg_lambdas = {
        "xp": cfg.lambda_xp,
        "bt": cfg.lambda_bt,
        "plane_bt": getattr(cfg, "lambda_plane_bt", 0.0),
        "cca": getattr(cfg, "lambda_block_cca", 0.0),
    }
    active_regs = [k for k, v in reg_lambdas.items() if v > 0]

    def lambda_scale(epoch: int) -> float:
        """Linear warm-up: cfg.lambda_start_frac at epoch 1 → 1.0 at the last epoch."""
        start = getattr(cfg, "lambda_start_frac", 1.0)
        if cfg.epochs <= 1:
            return 1.0
        return start + (1.0 - start) * (epoch - 1) / (cfg.epochs - 1)

    best_val_zeta = float("-inf")
    best_val_epoch = None
    best_checkpoint_val_loss = float("nan")
    min_val_loss = float("inf")
    min_val_loss_epoch = None
    threshold_metric, checkpoint_thresholds = _resolve_threshold_checkpoints(cfg)
    epoch_checkpoint_interval = int(getattr(cfg, "checkpoint_every_epochs", 0) or 0)
    saved_checkpoint_thresholds = set()
    checkpoint_records = []
    epoch_checkpoint_records = []
    history = {"train_loss": [], "val_loss": [],
               "train_s": [], "val_s": [],
               "train_c_plus": [], "val_c_plus": [],
               "train_zeta": [], "val_zeta": [],
               "train_reg": [], "val_reg": [],
               "lambda_scale": [],
               "reg_raw":    {k: [] for k in active_regs},
               "reg_scaled": {k: [] for k in active_regs}}
    log_path = os.path.join(cfg.out_dir, "log.csv")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss",
                                 "train_s", "val_s",
                                 "train_c_plus", "val_c_plus",
                                 "train_zeta", "val_zeta",
                                 "train_reg", "val_reg"])

    threshold_ckpt_log_path = os.path.join(cfg.ckpt_dir, f"val_{threshold_metric}_checkpoints.csv")
    if checkpoint_thresholds:
        with open(threshold_ckpt_log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "metric", "threshold", "epoch", "val_s", "val_zeta", "val_c_plus",
                "val_loss", "train_loss", "train_s", "train_zeta", "path",
            ])

    epoch_ckpt_log_path = os.path.join(cfg.ckpt_dir, "epoch_checkpoints.csv")
    if epoch_checkpoint_interval > 0:
        with open(epoch_ckpt_log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "interval", "epoch", "val_s", "val_zeta", "val_c_plus",
                "val_loss", "train_loss", "train_s", "train_zeta", "path",
            ])

    t0 = time.time()
    quiet_train = os.environ.get("QUIET_TRAIN", "").lower() in {"1", "true", "yes"}

    for epoch in range(1, cfg.epochs + 1):
        scale = lambda_scale(epoch)
        # --- train ---
        model.train()
        epoch_losses, epoch_s, epoch_c_plus, epoch_zeta, epoch_reg = [], [], [], [], []
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.epochs} [train]",
            leave=False,
            disable=quiet_train,
        )
        for (batch,) in pbar:
            batch = batch.to(device)                    # (K, N, T)
            optimizer.zero_grad()
            F = model(batch)                            # (K, d, T)
            F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
            loss = loss_function(F, cfg=cfg, training=True, lambda_scale=scale)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                c_minus, c_plus, zeta = non_reversibility_components(
                    _batch_rms_normalize(F), "mean"
                )
                s = c_minus.item()
            l = loss.item()
            epoch_losses.append(l)
            epoch_s.append(s)
            epoch_c_plus.append(c_plus.item())
            epoch_zeta.append(zeta.item())
            epoch_reg.append(l + s)     # total = -S + reg  →  reg = total + S
            pbar.set_postfix(loss=f"{l:.4f}", S=f"{s:.4f}")

        scheduler.step(epoch)
        mean_train_loss = sum(epoch_losses) / len(epoch_losses)
        mean_train_s    = sum(epoch_s)      / len(epoch_s)
        mean_train_c_plus = sum(epoch_c_plus) / len(epoch_c_plus)
        mean_train_zeta = sum(epoch_zeta) / len(epoch_zeta)
        mean_train_reg  = sum(epoch_reg)    / len(epoch_reg)

        # --- validate ---
        model.eval()
        val_losses, val_s, val_c_plus, val_zeta, val_reg = [], [], [], [], []
        val_raw    = {k: [] for k in active_regs}
        val_scaled = {k: [] for k in active_regs}
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                if batch.shape[0] < 2:
                    continue
                F = model(batch)
                F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
                loss, info = loss_function(F, cfg=cfg, training=False,
                                           lambda_scale=scale, return_components=True)
                c_minus, c_plus, zeta = non_reversibility_components(
                    _batch_rms_normalize(F), "mean"
                )
                s = c_minus.item()
                val_losses.append(loss.item())
                val_s.append(s)
                val_c_plus.append(c_plus.item())
                val_zeta.append(zeta.item())
                val_reg.append(loss.item() + s)
                for k in active_regs:
                    val_raw[k].append(info["reg_raw"][k])
                    val_scaled[k].append(info["reg_scaled"][k])

        mean_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
        mean_val_s    = sum(val_s)      / len(val_s)      if val_s      else float("nan")
        mean_val_c_plus = sum(val_c_plus) / len(val_c_plus) if val_c_plus else float("nan")
        mean_val_zeta = sum(val_zeta) / len(val_zeta) if val_zeta else float("nan")
        mean_val_reg  = sum(val_reg)    / len(val_reg)    if val_reg    else float("nan")

        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(mean_val_loss)
        history["train_s"].append(mean_train_s)
        history["val_s"].append(mean_val_s)
        history["train_c_plus"].append(mean_train_c_plus)
        history["val_c_plus"].append(mean_val_c_plus)
        history["train_zeta"].append(mean_train_zeta)
        history["val_zeta"].append(mean_val_zeta)
        history["train_reg"].append(mean_train_reg)
        history["val_reg"].append(mean_val_reg)
        history["lambda_scale"].append(scale)
        for k in active_regs:
            history["reg_raw"][k].append(sum(val_raw[k]) / len(val_raw[k]) if val_raw[k] else float("nan"))
            history["reg_scaled"][k].append(sum(val_scaled[k]) / len(val_scaled[k]) if val_scaled[k] else float("nan"))

        if not quiet_train or epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:3d}/{cfg.epochs}  "
                f"train loss={mean_train_loss:.4f}  val loss={mean_val_loss:.4f}  "
                f"S[mean]={mean_val_s:.4f}  ζ={mean_val_zeta:.4f}  reg={mean_val_reg:.4f}  "
                f"λscale={scale:.2f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, mean_train_loss, mean_val_loss,
                                     mean_train_s, mean_val_s,
                                     mean_train_c_plus, mean_val_c_plus,
                                     mean_train_zeta, mean_val_zeta,
                                     mean_train_reg, mean_val_reg])

        if mean_val_loss < min_val_loss:
            min_val_loss = mean_val_loss
            min_val_loss_epoch = epoch

        if mean_val_zeta > best_val_zeta:
            best_val_zeta = mean_val_zeta
            best_val_epoch = epoch
            best_checkpoint_val_loss = mean_val_loss
            torch.save(
                _checkpoint_payload(
                    model,
                    cfg,
                    epoch,
                    checkpoint_selection="best_val_zeta",
                    val_loss=mean_val_loss,
                    val_s=mean_val_s,
                    val_c_plus=mean_val_c_plus,
                    val_zeta=mean_val_zeta,
                    train_loss=mean_train_loss,
                    train_s=mean_train_s,
                    train_c_plus=mean_train_c_plus,
                    train_zeta=mean_train_zeta,
                ),
                os.path.join(cfg.ckpt_dir, "best.pt"),
            )

        threshold_value = _threshold_metric_value(threshold_metric, mean_val_s, mean_val_zeta)
        for threshold in checkpoint_thresholds:
            if threshold in saved_checkpoint_thresholds:
                continue
            if threshold_value >= threshold:
                label = _fmt_threshold(threshold)
                ckpt_name = f"val_{threshold_metric}_{label}.pt"
                ckpt_path = os.path.join(cfg.ckpt_dir, ckpt_name)
                threshold_payload = {
                    "checkpoint_selection": f"val_{threshold_metric}_threshold",
                    "val_checkpoint_metric": threshold_metric,
                    "val_checkpoint_threshold": threshold,
                    f"val_{threshold_metric}_threshold": threshold,
                    "val_loss": mean_val_loss,
                    "val_s": mean_val_s,
                    "val_c_plus": mean_val_c_plus,
                    "val_zeta": mean_val_zeta,
                    "train_loss": mean_train_loss,
                    "train_s": mean_train_s,
                    "train_c_plus": mean_train_c_plus,
                    "train_zeta": mean_train_zeta,
                }
                torch.save(
                    _checkpoint_payload(
                        model,
                        cfg,
                        epoch,
                        **threshold_payload,
                    ),
                    ckpt_path,
                )
                saved_checkpoint_thresholds.add(threshold)
                checkpoint_records.append(
                    {
                        "metric": threshold_metric,
                        "threshold": threshold,
                        "epoch": epoch,
                        "value": threshold_value,
                    }
                )
                with open(threshold_ckpt_log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        threshold_metric,
                        threshold,
                        epoch,
                        mean_val_s,
                        mean_val_zeta,
                        mean_val_c_plus,
                        mean_val_loss,
                        mean_train_loss,
                        mean_train_s,
                        mean_train_zeta,
                        ckpt_path,
                    ])
                print(
                    f"Saved val-{_threshold_metric_label(threshold_metric)} checkpoint {ckpt_name}: "
                    f"threshold={threshold:g} epoch={epoch} "
                    f"val_{threshold_metric}={threshold_value:.4f}"
                )

        if epoch_checkpoint_interval > 0 and epoch % epoch_checkpoint_interval == 0:
            ckpt_name = f"epoch_{epoch:04d}.pt"
            ckpt_path = os.path.join(cfg.ckpt_dir, ckpt_name)
            torch.save(
                _checkpoint_payload(
                    model,
                    cfg,
                    epoch,
                    checkpoint_selection="epoch_interval",
                    checkpoint_every_epochs=epoch_checkpoint_interval,
                    val_loss=mean_val_loss,
                    val_s=mean_val_s,
                    val_c_plus=mean_val_c_plus,
                    val_zeta=mean_val_zeta,
                    train_loss=mean_train_loss,
                    train_s=mean_train_s,
                    train_c_plus=mean_train_c_plus,
                    train_zeta=mean_train_zeta,
                ),
                ckpt_path,
            )
            epoch_checkpoint_records.append(
                {
                    "interval": epoch_checkpoint_interval,
                    "epoch": epoch,
                    "val_zeta": mean_val_zeta,
                    "val_loss": mean_val_loss,
                    "train_loss": mean_train_loss,
                }
            )
            with open(epoch_ckpt_log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch_checkpoint_interval,
                    epoch,
                    mean_val_s,
                    mean_val_zeta,
                    mean_val_c_plus,
                    mean_val_loss,
                    mean_train_loss,
                    mean_train_s,
                    mean_train_zeta,
                    ckpt_path,
                ])
            print(
                f"Saved epoch checkpoint {ckpt_name}: "
                f"epoch={epoch} val_loss={mean_val_loss:.4f} val_zeta={mean_val_zeta:.4f}"
            )

    history["best_val_zeta"] = best_val_zeta
    history["best_val_epoch"] = best_val_epoch
    history["best_checkpoint_val_loss"] = best_checkpoint_val_loss
    history["best_val_loss"] = min_val_loss
    history["best_val_loss_epoch"] = min_val_loss_epoch
    history["checkpoint_threshold_metric"] = threshold_metric
    history["checkpoint_records"] = checkpoint_records
    history["epoch_checkpoint_records"] = epoch_checkpoint_records
    history["elapsed_s"] = time.time() - t0

    # --- save per-reg history for post-hoc visualisation ---
    if active_regs:
        reg_hist_path = os.path.join(cfg.out_dir, "reg_history.csv")
        cols = (["epoch", "lambda_scale"]
                + [f"raw_{k}"    for k in active_regs]
                + [f"scaled_{k}" for k in active_regs])
        with open(reg_hist_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for i in range(cfg.epochs):
                row = ([i + 1, history["lambda_scale"][i]]
                       + [history["reg_raw"][k][i]    for k in active_regs]
                       + [history["reg_scaled"][k][i] for k in active_regs])
                w.writerow(row)

    # --- loss curve ---
    # Keep this to the main training-dynamics panel so checkpoint markers stay
    # readable as the number of active regularizers changes.
    ep = list(range(1, cfg.epochs + 1))
    # Compute total λ·reg as the sum of individually tracked scaled components.
    # history["val_reg"] = loss + s is unreliable with the softmin objective because
    # non_reversibility_S("softmin") silently falls through to "sum" mode, so the
    # subtraction loses the softmin term and inflates the result by ~2x.
    total_scaled_reg = [
        sum(history["reg_scaled"][k][i] for k in active_regs)
        for i in range(len(ep))
    ] if active_regs else [0.0] * len(ep)
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.plot(ep, history["val_s"],     label="S mean/plane (↑)",  color="steelblue")
    ax.plot(
        ep,
        history["val_c_plus"],
        label=r"$\|C^{(+)}\|_F^2$",
        color="mediumpurple",
        alpha=0.35,
    )
    ax.plot(ep, total_scaled_reg,     label="reg (↓)",   color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Embedding validation loss components")
    ax.spines[["top", "right"]].set_visible(False)

    ax_zeta = ax.twinx()
    ax_zeta.plot(ep, history["val_zeta"], label="ζ", color="seagreen", alpha=0.6)
    ax_zeta.set_ylabel("Validation ζ")
    ax_zeta.spines["top"].set_visible(False)

    if best_val_epoch is not None:
        idx = best_val_epoch - 1
        if 0 <= idx < len(history["val_zeta"]):
            y = history["val_zeta"][idx]
            ax_zeta.scatter(
                [best_val_epoch],
                [y],
                s=52,
                color="goldenrod",
                edgecolors="black",
                linewidths=0.7,
                zorder=6,
            )
            ax_zeta.annotate(
                "best val ζ",
                xy=(best_val_epoch, y),
                xytext=(5, -10),
                textcoords="offset points",
                va="top",
                ha="left",
                fontsize=8,
                color="black",
            )

    handles, labels = ax.get_legend_handles_labels()
    z_handles, z_labels = ax_zeta.get_legend_handles_labels()
    ax.legend(handles + z_handles, labels + z_labels, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    return history
