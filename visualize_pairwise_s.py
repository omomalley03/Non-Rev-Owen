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
"""

import argparse
import csv
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

from config import Config
from loss import (
    S_ratio as compute_S_ratio,
    _batch_rms_normalize,
    non_reversibility_S,
    non_reversibility_S_per_plane,
)


def _resolve_run_dir(run_arg):
    runs_root = "runs"
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
    """Compute d×d matrix of non-reversibility S for every dim pair.

    Entry (i,j) = non_reversibility_S(F[:, [i,j], :]).
    Symmetric, so only the upper triangle is computed.
    """
    K, d, T = F.shape
    S_mat = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            F_pair = F[:, [i, j], :]  # (K, 2, T)
            s = non_reversibility_S(F_pair).item()
            S_mat[i, j] = s
            S_mat[j, i] = s
        # diagonal: S of a single dim with itself is 0 (no rotation in 1D)
    return S_mat


def load_and_infer(run_dir: str):
    """Load checkpoint, run inference, return F_hat and config."""
    from data import load_mcmaze_cached, gaussian_smooth, make_windows, train_val_split
    from model import MLP

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

    model = MLP(
        in_channels=windows.shape[1], d=cfg.d, hidden_dim=cfg.hidden_dim,
        depth=cfg.depth, dropout=getattr(cfg, "dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    (val_tensor,) = next(iter(loader))

    with torch.no_grad():
        F = model(val_tensor)
        F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
        F_hat = _batch_rms_normalize(F)

    return F_hat, cfg


def plot_pairwise_s(run_dir: str, out_path: str | None = None):
    """Create the pairwise S diagnostic figure."""
    print(f"Loading {run_dir}...")
    F_hat, cfg = load_and_infer(run_dir)
    d = F_hat.shape[1]
    D = d // 2
    exp_name = os.path.basename(run_dir.rstrip("/"))

    if out_path is None:
        out_path = os.path.join(run_dir, "outputs", "pairwise_s.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Computing pairwise S matrix ({d}x{d} = {d*(d-1)//2} pairs)...")
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

    S_p = non_reversibility_S_per_plane(F_hat).numpy()
    s_ratio = compute_S_ratio(F_hat).item()

    # ── Derived metrics ───────────────────────────────────────────────
    # Within-plane S: the native (2i, 2i+1) pairs
    within = np.array([S_mat[2*p, 2*p+1] for p in range(D)])
    # Cross-plane S: all off-block entries
    cross_entries = []
    for i in range(d):
        for j in range(i + 1, d):
            if i // 2 != j // 2:  # different planes
                cross_entries.append(S_mat[i, j])
    cross = np.array(cross_entries)

    # ── Figure: 1×3 layout ────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 7))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 0.8, 0.8], wspace=0.3)

    # Panel 1: Full d×d pairwise S heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    vmax = max(S_mat.max(), 0.01)
    im = ax1.imshow(S_mat, vmin=0, vmax=vmax, cmap="YlOrRd", interpolation="nearest")
    ax1.set_xticks(range(d))
    ax1.set_yticks(range(d))
    ax1.set_xticklabels([f"d{i}" for i in range(d)], fontsize=6, rotation=90)
    ax1.set_yticklabels([f"d{i}" for i in range(d)], fontsize=6)
    ax1.set_title("Pairwise non-reversibility S(dim_i, dim_j)", fontsize=11)
    # draw plane blocks
    for p in range(D):
        rect = plt.Rectangle((2*p - 0.5, 2*p - 0.5), 2, 2, fill=False,
                              edgecolor="blue", lw=2, linestyle="--")
        ax1.add_patch(rect)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="S")
    # annotate native plane entries
    for p in range(D):
        i, j = 2*p, 2*p+1
        ax1.text(j, i, f"{S_mat[i,j]:.2f}", ha="center", va="center",
                 fontsize=5, color="blue", fontweight="bold")
        ax1.text(i, j, f"{S_mat[j,i]:.2f}", ha="center", va="center",
                 fontsize=5, color="blue", fontweight="bold")

    # Panel 2: D×D plane-level summary (max cross-plane S per plane pair)
    ax2 = fig.add_subplot(gs[0, 1])
    plane_cross_S = np.zeros((D, D))
    for p in range(D):
        for q in range(D):
            if p == q:
                # within-plane: the native pair
                plane_cross_S[p, q] = S_mat[2*p, 2*p+1]
            else:
                # max S over all 4 cross-pairings between planes p and q
                vals = [S_mat[2*p+a, 2*q+b] for a in range(2) for b in range(2)]
                plane_cross_S[p, q] = max(vals)
    im2 = ax2.imshow(plane_cross_S, vmin=0, vmax=vmax, cmap="YlOrRd", interpolation="nearest")
    ax2.set_xticks(range(D))
    ax2.set_yticks(range(D))
    ax2.set_xticklabels([f"P{i}" for i in range(D)], fontsize=8)
    ax2.set_yticklabels([f"P{i}" for i in range(D)], fontsize=8)
    ax2.set_title("Plane-level summary\n(diagonal=within, off-diag=max cross)", fontsize=10)
    for i in range(D):
        for j in range(D):
            color = "white" if plane_cross_S[i, j] > vmax * 0.6 else "black"
            ax2.text(j, i, f"{plane_cross_S[i,j]:.2f}", ha="center", va="center",
                     fontsize=7, color=color)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="S")

    # Panel 3: Distribution comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(within, bins=max(D // 2, 3), alpha=0.7, color="steelblue",
             edgecolor="black", label=f"Within-plane (n={len(within)})")
    ax3.hist(cross, bins=20, alpha=0.7, color="salmon",
             edgecolor="black", label=f"Cross-plane (n={len(cross)})")
    ax3.axvline(within.mean(), color="steelblue", ls="--", lw=2,
                label=f"Within mean={within.mean():.3f}")
    ax3.axvline(cross.mean(), color="red", ls="--", lw=2,
                label=f"Cross mean={cross.mean():.3f}")
    ax3.set_xlabel("S (non-reversibility)", fontsize=10)
    ax3.set_ylabel("Count", fontsize=10)
    ax3.set_title("Within-plane vs cross-plane S", fontsize=10)
    ax3.legend(fontsize=7, loc="upper right")
    ax3.spines[["top", "right"]].set_visible(False)

    # Summary line
    lxp = getattr(cfg, "lambda_xp", 0)
    lbt = getattr(cfg, "lambda_bt", 0)
    lpbt = getattr(cfg, "lambda_plane_bt", 0)
    lcca = getattr(cfg, "lambda_block_cca", 0)
    fig.suptitle(
        f"{exp_name}\n"
        f"S_ratio={s_ratio:.3f}   "
        f"Within S: mean={within.mean():.3f} min={within.min():.3f}   "
        f"Cross S: mean={cross.mean():.3f} max={cross.max():.3f}   "
        f"Ratio(within/cross)={within.mean()/(cross.mean()+1e-8):.1f}x\n"
        f"lambda_xp={lxp:g}  lambda_bt={lbt:g}  lambda_plane_bt={lpbt:g}  lambda_block_cca={lcca:g}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
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
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_arg = args.run or args.run_flag
    run_dir = _resolve_run_dir(run_arg)
    print(f"Using run: {run_dir}")
    plot_pairwise_s(run_dir, out_path=args.out)


if __name__ == "__main__":
    main()
