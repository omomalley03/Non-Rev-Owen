from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import gaussian_smooth, load_mcmaze_cached, make_windows, soft_normalize, train_val_split
from model import MLP
from visualize import _get_condition_groups


def _cfg_get(cfg: dict, name: str, default=None):
    return cfg[name] if name in cfg else default


def load_finetuned_embeddings(model_path: Path):
    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = saved["config"]

    spikes_raw, bin_width_s, trial_info, time_index_s, _ = load_mcmaze_cached(
        cfg["nwb_path"], cfg["bin_ms"]
    )
    sigma_samples = round((cfg["sigma_ms"] * 1e-3) / bin_width_s)
    X_smooth = gaussian_smooth(spikes_raw, sigma_samples)
    softnorm = _cfg_get(cfg, "softnorm_method", "none")
    if softnorm and softnorm != "none":
        X_smooth = soft_normalize(X_smooth, method=softnorm)

    windows = make_windows(
        X_smooth,
        trial_info,
        time_index_s,
        bin_width_s,
        strategy=_cfg_get(cfg, "window_strategy", "trial_aligned"),
        window_size=int(cfg["window_size"]),
        align_field=_cfg_get(cfg, "align_field", "move_onset_time"),
        pre_ms=_cfg_get(cfg, "pre_ms", 100),
    )
    if _cfg_get(cfg, "split", "dataset") == "random":
        trial_info = trial_info.drop(columns=["split"], errors="ignore")

    _, val_ds = train_val_split(windows, trial_info, cfg["val_split"], cfg["seed"])
    val_info = trial_info.iloc[list(val_ds.indices)].reset_index(drop=True)
    groups, colors = _get_condition_groups(val_info)

    model = MLP(
        in_channels=spikes_raw.shape[0],
        d=cfg["d"],
        hidden_dim=cfg["hidden_dim"],
        depth=cfg["depth"],
        dropout=cfg["dropout"],
        temporal_filters=_cfg_get(cfg, "temporal_filters", 0),
        temporal_kernel_size=_cfg_get(cfg, "temporal_kernel_size", 31),
        temporal_frontend=_cfg_get(cfg, "temporal_frontend", "symmetric"),
        residual_kernels=_cfg_get(cfg, "residual_kernels", "3,7,15,31"),
    )
    model.load_state_dict(saved["embedder_state_dict"])
    model.eval()

    (val_tensor,) = next(iter(DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)))
    with torch.no_grad():
        F = model(val_tensor)
        mean_axis = _cfg_get(cfg, "F_mean_axis", (0, 2))
        F = F - F.mean(dim=mean_axis, keepdim=True)

    return F.cpu().numpy(), groups, colors


def plot_first_planes(F_hat, groups, colors, out_path: Path, n_planes: int = 8):
    _, d, _ = F_hat.shape
    max_planes = d // 2
    n_planes = min(n_planes, max_planes)
    planes = F_hat.reshape(F_hat.shape[0], max_planes, 2, F_hat.shape[2])

    ncols = 4
    nrows = int(np.ceil(n_planes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows), squeeze=False)

    for p in range(n_planes):
        ax = axes[p // ncols, p % ncols]
        for cond_key, idx_list in groups.items():
            mean_traj = planes[idx_list, p].mean(axis=0)
            ax.plot(mean_traj[0], mean_traj[1], lw=1.4, color=colors[cond_key], alpha=0.9)
            ax.scatter(mean_traj[0, 0], mean_traj[1, 0], color=colors[cond_key], s=18, zorder=5)

        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"Plane {p}", fontsize=12)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="datalim")

    for p in range(n_planes, nrows * ncols):
        axes[p // ncols, p % ncols].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-planes", type=int, default=8)
    args = parser.parse_args()

    F_hat, groups, colors = load_finetuned_embeddings(args.model)
    plot_first_planes(F_hat, groups, colors, args.out, args.n_planes)
    print(args.out)


if __name__ == "__main__":
    main()
