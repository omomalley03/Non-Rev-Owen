from __future__ import annotations

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _read_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _loss_paths(run_dir_or_log_path: str) -> tuple[str, str, str]:
    if os.path.basename(run_dir_or_log_path) == "log.csv":
        log_path = run_dir_or_log_path
        out_dir = os.path.dirname(log_path)
        run_dir = os.path.dirname(out_dir) if os.path.basename(out_dir) == "outputs" else out_dir
        return run_dir, out_dir, log_path

    run_dir = run_dir_or_log_path
    out_dir = os.path.join(run_dir, "outputs")
    log_path = os.path.join(out_dir, "log.csv")
    return run_dir, out_dir, log_path


def plot_loss_curve(run_dir: str, cfg=None) -> None:
    """Regenerate the training loss curve from saved CSVs.

    Reads outputs/log.csv and, when present, outputs/reg_history.csv. The
    checkpoint config is optional so this can be used for log-only plotting.
    """
    run_dir, out_dir, log_path = _loss_paths(run_dir)
    if not os.path.isfile(log_path):
        print(f"  [loss curve] no log.csv found at {log_path!r} - skipping.")
        return

    epochs, val_s, val_c_plus, val_zeta, val_losses, val_reg = [], [], [], [], [], []
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            epoch = _read_float(row, "epoch")
            if epoch is None:
                continue
            epochs.append(int(epoch))
            val_s.append(_read_float(row, "val_s"))
            val_c_plus.append(_read_float(row, "val_c_plus"))
            val_zeta.append(_read_float(row, "val_zeta"))
            val_losses.append(_read_float(row, "val_loss"))
            val_reg.append(_read_float(row, "val_reg"))

    if not epochs:
        print(f"  [loss curve] no epoch rows found in {log_path!r} - skipping.")
        return

    def full_series(values: list[float | None]) -> bool:
        return len(values) == len(epochs) and all(v is not None for v in values)

    has_s = full_series(val_s)
    has_c_plus = full_series(val_c_plus)
    has_zeta = full_series(val_zeta)
    has_val_reg = full_series(val_reg)

    reg_scaled: dict[str, list[float]] = {}
    reg_path = os.path.join(out_dir, "reg_history.csv")
    if os.path.isfile(reg_path):
        with open(reg_path, newline="") as f:
            reader = csv.DictReader(f)
            csv_regs = [
                c[len("scaled_"):]
                for c in (reader.fieldnames or [])
                if c.startswith("scaled_")
            ]
            for k in csv_regs:
                reg_scaled[k] = []
            for row in reader:
                for k in csv_regs:
                    value = _read_float(row, f"scaled_{k}")
                    if value is not None:
                        reg_scaled[k].append(value)

    active_regs = list(reg_scaled)
    if cfg is not None and not active_regs:
        reg_lambdas = {
            "xp": getattr(cfg, "lambda_xp", 0.0),
            "bt": getattr(cfg, "lambda_bt", 0.0),
            "plane_bt": getattr(cfg, "lambda_plane_bt", 0.0),
            "cca": getattr(cfg, "lambda_block_cca", 0.0),
        }
        active_regs = [k for k, v in reg_lambdas.items() if v > 0]

    fig, ax = plt.subplots(figsize=(5.6, 4))
    if has_s:
        ax.plot(epochs, val_s, label="S mean/plane (up)", color="steelblue")
    if has_c_plus:
        ax.plot(
            epochs,
            val_c_plus,
            label=r"$\|C^{(+)}\|_F^2$",
            color="mediumpurple",
            alpha=0.35,
        )
    if reg_scaled:
        total_scaled = [
            sum(
                reg_scaled[k][i]
                for k in active_regs
                if k in reg_scaled and i < len(reg_scaled[k])
            )
            for i in range(len(epochs))
        ]
        ax.plot(epochs, total_scaled, label="total lambda*reg (down)", color="tomato")
    elif has_val_reg:
        ax.plot(epochs, val_reg, label="val reg from log.csv (down)", color="tomato")
    if not (has_s or has_c_plus or reg_scaled or has_val_reg):
        plotted = [v if v is not None else float("nan") for v in val_losses]
        ax.plot(epochs, plotted, label="val loss (down)", color="steelblue")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Embedding validation loss components")
    ax.spines[["top", "right"]].set_visible(False)

    ax_zeta = None
    if has_zeta:
        ax_zeta = ax.twinx()
        ax_zeta.plot(epochs, val_zeta, label="zeta", color="seagreen", alpha=0.6)
        ax_zeta.set_ylabel("Validation zeta")
        ax_zeta.spines["top"].set_visible(False)

    best_epoch = None
    best_label = "best val loss"
    best_ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if os.path.isfile(best_ckpt_path):
        try:
            ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            best_epoch = int(ckpt["epoch"])
            if ckpt.get("checkpoint_selection") == "best_val_zeta":
                best_label = "best val zeta"
        except Exception as exc:
            print(f"  [loss curve] could not read best.pt ({exc}); falling back to log.csv.")
    if has_zeta and best_label != "best val zeta":
        best_epoch = max(range(1, len(val_zeta) + 1), key=lambda i: val_zeta[i - 1])
        best_label = "best val zeta"
    if best_epoch is None and any(v is not None for v in val_losses):
        valid_loss_epochs = [
            (i, v) for i, v in enumerate(val_losses, start=1) if v is not None
        ]
        best_epoch = min(valid_loss_epochs, key=lambda item: item[1])[0]

    if best_epoch is not None:
        idx = best_epoch - 1
        best_on_zeta = (
            best_label == "best val zeta"
            and ax_zeta is not None
            and 0 <= idx < len(val_zeta)
            and val_zeta[idx] is not None
        )
        if 0 <= idx < len(val_s):
            target_ax = ax_zeta if best_on_zeta else ax
            series = val_zeta if best_on_zeta else val_s
            y = series[idx]
            if y is not None:
                target_ax.scatter(
                    [best_epoch],
                    [y],
                    s=52,
                    color="goldenrod",
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=6,
                )
                target_ax.annotate(
                    best_label,
                    xy=(best_epoch, y),
                    xytext=(5, -10),
                    textcoords="offset points",
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="black",
                )

    handles, labels = ax.get_legend_handles_labels()
    if ax_zeta is not None:
        z_handles, z_labels = ax_zeta.get_legend_handles_labels()
        handles += z_handles
        labels += z_labels
    ax.legend(handles, labels, loc="lower right")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")
