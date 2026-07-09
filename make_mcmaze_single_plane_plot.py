from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data import gaussian_smooth, load_mcmaze_cached, make_windows, soft_normalize, train_val_split
from model import MLP
from visualize import _get_condition_groups


def load_validation_planes(run_dir: Path):
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]

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
        strategy=cfg.window_strategy,
        window_size=cfg.window_size,
        align_field=getattr(cfg, "align_field", "move_onset_time"),
        pre_ms=getattr(cfg, "pre_ms", 100),
    )
    if getattr(cfg, "split", "dataset") == "random":
        trial_info = trial_info.drop(columns=["split"], errors="ignore")

    _, val_ds = train_val_split(windows, trial_info, cfg.val_split, cfg.seed)
    val_info = trial_info.iloc[list(val_ds.indices)].reset_index(drop=True)
    groups, _ = _get_condition_groups(val_info)

    model = MLP(
        in_channels=spikes_raw.shape[0],
        d=cfg.d,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        dropout=cfg.dropout,
        temporal_filters=getattr(cfg, "temporal_filters", 0),
        temporal_kernel_size=getattr(cfg, "temporal_kernel_size", 31),
        temporal_frontend=getattr(cfg, "temporal_frontend", "symmetric"),
        residual_kernels=getattr(cfg, "residual_kernels", "3,7,15,31"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    (val_tensor,) = next(iter(DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)))
    with torch.no_grad():
        F = model(val_tensor)
        F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)

    planes = F.cpu().numpy().reshape(len(val_ds), cfg.d // 2, 2, cfg.window_size)
    time_ms = np.arange(cfg.window_size) * cfg.bin_ms - getattr(cfg, "pre_ms", 0)
    return planes, groups, time_ms, cfg


def plot_plane(planes, groups, time_ms, plane: int, out_path: Path, show_title: bool, show_text: bool):
    n_planes = planes.shape[1]
    if plane < 0 or plane >= n_planes:
        raise ValueError(f"plane {plane} is out of range for {n_planes} planes")

    dim0, dim1 = 2 * plane, 2 * plane + 1
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(time_ms[0], time_ms[-1])

    fig, ax = plt.subplots(figsize=(6.3, 5.25))
    fig.subplots_adjust(left=0.15, right=0.86, top=0.90 if show_title else 0.97, bottom=0.14)

    all_xy = []
    for idx_list in groups.values():
        mean_traj = planes[idx_list, plane].mean(axis=0)
        x, y = mean_traj[0], mean_traj[1]
        all_xy.append(np.column_stack([x, y]))
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.25, alpha=0.86)
        lc.set_array(time_ms[:-1])
        ax.add_collection(lc)

    all_xy = np.concatenate(all_xy, axis=0)
    mins, maxs = all_xy.min(axis=0), all_xy.max(axis=0)
    center = (mins + maxs) / 2
    span = max(maxs - mins)
    half = span * 0.58 if span > 0 else 0.1
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="k", lw=0.5, alpha=0.25)
    ax.axvline(0, color="k", lw=0.5, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(f"Embedding dim {dim0}", fontsize=11)
    ax.set_ylabel(f"Embedding dim {dim1}", fontsize=11, labelpad=8)

    if show_title:
        ax.set_title(f"Plane {plane} (dims {dim0}, {dim1})", fontsize=13, pad=10)
    if show_text:
        ax.text(
            0.02,
            0.98,
            "Validation condition averages",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=4),
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Time from move onset (ms)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--planes", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("mcmaze"))
    parser.add_argument("--show-title", action="store_true")
    parser.add_argument("--show-text", action="store_true")
    args = parser.parse_args()

    planes, groups, time_ms, _ = load_validation_planes(args.run)
    for plane in args.planes:
        out_path = args.out_dir / f"no_cca_d256_plane{plane}_dims{2 * plane}_{2 * plane + 1}_clean.png"
        plot_plane(planes, groups, time_ms, plane, out_path, args.show_title, args.show_text)
        print(out_path)


if __name__ == "__main__":
    main()
