"""
Visualize pairwise non-reversibility S for ALL dimension pairs.

For d-dimensional embeddings, computes a d×d matrix where entry (i,j)
is the non-reversibility score when dims i and j are treated as a 2D
rotation plane.  The native planes (0,1), (2,3), ... should dominate
the diagonal blocks.  High off-diagonal entries indicate cross-plane
rotation structure — which is what non_rev_regularizer tries to suppress.

Usage
-----
    python visualize_pairwise_s.py runs/some_run/
    python visualize_pairwise_s.py              # most recent run
    python visualize_pairwise_s.py --run 2      # 2nd most recent
    python visualize_pairwise_s.py --synth      # most recent synth run
    python visualize_pairwise_s.py --synth --data rotations_v3.npy
"""

from __future__ import annotations

import argparse
import csv

import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import Config
from paths import RUNS_DIR, SYNTH_RUNS_DIR
from loss import (
    S_ratio as compute_S_ratio,
    _batch_rms_normalize,
)


def _resolve_run_dir(run_arg, synth=False):
    runs_root = SYNTH_RUNS_DIR if synth else RUNS_DIR
    if run_arg is None:
        run_arg = "1"
    if run_arg.isdigit():
        idx = int(run_arg) - 1
        candidates = []
        for d in os.listdir(runs_root):
            ckpt = os.path.join(runs_root, d, "checkpoints", "best.pt")
            if os.path.isfile(ckpt):
                candidates.append((os.path.getmtime(ckpt), os.path.join(runs_root, d)))
        candidates.sort(reverse=True)
        if idx >= len(candidates):
            raise FileNotFoundError(f"Only {len(candidates)} runs found, asked for #{idx+1}")
        return candidates[idx][1]
    return run_arg


def pairwise_S_matrix(F: torch.Tensor) -> np.ndarray:
    """Compute d×d matrix of normalised non-reversibility S_ratio for every dim pair.

    Entry (i,j) = S_ratio(F[:, [i,j], :])  ∈ [0, 1].
    Symmetric, so only the upper triangle is computed.
    """
    K, d, T = F.shape
    S_mat = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            F_pair = F[:, [i, j], :]  # (K, 2, T)
            s = compute_S_ratio(F_pair).item()
            S_mat[i, j] = s
            S_mat[j, i] = s
        # diagonal: S of a single dim with itself is 0 (no rotation in 1D)
    return S_mat


def load_and_infer(run_dir: str):
    """Load checkpoint, run inference, return F_hat and config (MC_Maze data)."""
    from data import load_mcmaze_cached, gaussian_smooth, make_windows, train_val_split
    from model import MLP, infer_multiscale_symmetric_conv_layers

    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    spikes_raw, bin_width_s, trial_info, time_index_s, _ = load_mcmaze_cached(
        cfg.nwb_path, cfg.bin_ms
    )
    sigma_samples = round((cfg.sigma_ms * 1e-3) / bin_width_s)
    X_smooth = gaussian_smooth(spikes_raw, sigma_samples)
    softnorm = getattr(cfg, "softnorm_method", "none")
    if softnorm and softnorm != "none":
        from data import soft_normalize
        X_smooth = soft_normalize(X_smooth, method=softnorm)

    windows = make_windows(
        X_smooth, trial_info, time_index_s, bin_width_s,
        strategy=cfg.window_strategy, window_size=cfg.window_size,
        align_field=getattr(cfg, "align_field", "move_onset_time"),
        pre_ms=getattr(cfg, "pre_ms", 100),
    )
    if cfg.split == "random":
        trial_info = trial_info.drop(columns=["split"], errors="ignore")
    _, val_ds = train_val_split(windows, trial_info, cfg.val_split, cfg.seed)

    state_dict = ckpt["model_state_dict"]
    model = MLP(
        in_channels=windows.shape[1], d=cfg.d, hidden_dim=cfg.hidden_dim,
        depth=cfg.depth, dropout=getattr(cfg, "dropout", 0.0),
        temporal_filters=getattr(cfg, "temporal_filters", 0),
        temporal_kernel_size=getattr(cfg, "temporal_kernel_size", 31),
        temporal_frontend=getattr(cfg, "temporal_frontend", "symmetric"),
        residual_kernels=getattr(cfg, "residual_kernels", "3,7,15,31"),
        multiscale_symmetric_conv_layers=infer_multiscale_symmetric_conv_layers(
            state_dict,
            getattr(cfg, "multiscale_symmetric_conv_layers", 1),
        ),
    )
    model.load_state_dict(state_dict)
    model.eval()

    loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    (val_tensor,) = next(iter(loader))

    with torch.no_grad():
        F = model(val_tensor)
        F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
        F_hat = _batch_rms_normalize(F)

    return F_hat, cfg


def load_and_infer_synth(run_dir: str, data_path: str | None = None):
    """Load checkpoint, run inference, return F_hat and config (synthetic data)."""
    from model import MLP, infer_multiscale_symmetric_conv_layers
    from visualize_synth import load_synthetic_windows, train_val_split_synth

    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    path = data_path or getattr(cfg, "synth_data_path", "rotations.npy")
    windows = load_synthetic_windows(cfg, data_path=path)
    _, val_ds = train_val_split_synth(windows, cfg.val_split, cfg.seed)

    state_dict = ckpt["model_state_dict"]
    model = MLP(
        in_channels=windows.shape[1], d=cfg.d, hidden_dim=cfg.hidden_dim,
        depth=cfg.depth, dropout=getattr(cfg, "dropout", 0.0),
        temporal_filters=getattr(cfg, "temporal_filters", 0),
        temporal_kernel_size=getattr(cfg, "temporal_kernel_size", 31),
        temporal_frontend=getattr(cfg, "temporal_frontend", "symmetric"),
        residual_kernels=getattr(cfg, "residual_kernels", "3,7,15,31"),
        multiscale_symmetric_conv_layers=infer_multiscale_symmetric_conv_layers(
            state_dict,
            getattr(cfg, "multiscale_symmetric_conv_layers", 1),
        ),
    )
    model.load_state_dict(state_dict)
    model.eval()

    loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    (val_tensor,) = next(iter(loader))

    with torch.no_grad():
        F = model(val_tensor)
        F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
        F_hat = _batch_rms_normalize(F)

    return F_hat, cfg


def plot_pairwise_s(run_dir: str, out_path: str | None = None,
                    F_hat: torch.Tensor | None = None, cfg: Config | None = None,
                    synth: bool = False, data_path: str | None = None):
    """Create the pairwise S diagnostic figure.

    If F_hat and cfg are provided, uses them directly (skips loading).
    Otherwise loads from run_dir using synth or MC_Maze pipeline.
    """
    if F_hat is None:
        print(f"Loading {run_dir}...")
        if synth:
            F_hat, cfg = load_and_infer_synth(run_dir, data_path=data_path)
        else:
            F_hat, cfg = load_and_infer(run_dir)
    d = F_hat.shape[1]
    D = d // 2
    exp_name = os.path.basename(run_dir.rstrip("/"))

    if out_path is None:
        out_path = os.path.join(run_dir, "outputs", "pairwise_s.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Computing pairwise S_ratio matrix ({d}x{d} = {d*(d-1)//2} pairs)...")
    S_mat = pairwise_S_matrix(F_hat)

    # dump the raw matrix as CSV alongside the figure
    csv_path = os.path.join(os.path.dirname(out_path), "pairwise_s_matrix.csv")
    labels = [f"d{i}" for i in range(d)]
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow([""] + labels)
        for i in range(d):
            w.writerow([labels[i]] + [f"{S_mat[i, j]:.6f}" for j in range(d)])
    print(f"Saved -> {csv_path}")

    # ── Single d×d heatmap ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(S_mat, vmin=0, vmax=1, cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    ax.set_xticklabels([str(i) for i in range(d)], fontsize=8)
    ax.set_yticklabels([str(i) for i in range(d)], fontsize=8)
    ax.set_xlabel("Embedding dim", fontsize=10)
    ax.set_ylabel("Embedding dim", fontsize=10)

    # draw native plane blocks
    for p in range(D):
        rect = plt.Rectangle((2*p - 0.5, 2*p - 0.5), 2, 2, fill=False,
                              edgecolor="blue", lw=2, linestyle="--")
        ax.add_patch(rect)

    # annotate each cell
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            color = "white" if S_mat[i, j] > 0.5 else "black"
            ax.text(j, i, f"{S_mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="S ratio")
    ax.set_title(f"Pairwise S ratio (dim_i, dim_j)\n{exp_name}", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", default=None,
                        help="Run directory path, integer (1=most recent), or omit for most recent.")
    parser.add_argument("--run", dest="run_flag", default=None,
                        help="Alternative: --run <path|int>")
    parser.add_argument("--synth", action="store_true",
                        help="Use synthetic data pipeline (looks in synth_runs/).")
    parser.add_argument("--data", default=None,
                        help="Override synthetic data .npy path from checkpoint config.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_arg = args.run or args.run_flag
    run_dir = _resolve_run_dir(run_arg, synth=args.synth)
    print(f"Using run: {run_dir}")
    plot_pairwise_s(run_dir, out_path=args.out, synth=args.synth,
                    data_path=args.data)


if __name__ == "__main__":
    main()
