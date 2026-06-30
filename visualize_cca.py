"""
Visualize block CCA regularization performance for a trained checkpoint.

Produces a single multi-panel figure:
  - 2x4 grid of rotation planes (condition-avg, time-coded)
  - D×D canonical correlation heatmap (plane-level independence)
  - d×d Pearson correlation heatmap (dim-level)
  - Per-plane S bar chart
  - Summary metrics text

Usage
-----
    python visualize_cca.py runs/block_cca_sweep_.../03_cca_3
    python visualize_cca.py                          # most recent run
    python visualize_cca.py --run 2                  # 2nd most recent
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

from config import Config
from paths import RUNS_DIR
from loss import (
    S_ratio as compute_S_ratio,
    _batch_rms_normalize,
    non_reversibility_S,
    non_reversibility_S_per_plane,
    block_cca_reg,
    barlow_twins_reg,
    plane_barlow_twins_reg,
    _plane_samples,
    _whiten_2d,
)


def _resolve_run_dir(run_arg):
    """Resolve run directory from CLI arg (same logic as visualize.py)."""
    runs_root = RUNS_DIR
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


def _get_condition_groups(trial_info_val, n_bins=8):
    """Group val trials by (trial_type, trial_version) or reach angle bins."""
    n = len(trial_info_val)
    cols = trial_info_val.columns

    if "trial_type" in cols and "trial_version" in cols:
        groups = {}
        for i in range(n):
            row = trial_info_val.iloc[i]
            key = (int(row["trial_type"]), int(row["trial_version"]))
            groups.setdefault(key, []).append(i)

        # sort by reach angle
        angles = {}
        for key, idx_list in groups.items():
            if "reach_angle" in cols:
                angs = [trial_info_val.iloc[i]["reach_angle"] for i in idx_list]
                angs = [a for a in angs if not np.isnan(a)]
                if angs:
                    angles[key] = float(np.arctan2(np.mean(np.sin(angs)), np.mean(np.cos(angs))))
                    continue
            angles[key] = float("inf")
        sorted_keys = sorted(groups.keys(), key=lambda k: angles[k])
        return {k: groups[k] for k in sorted_keys}

    # fallback: per-trial
    return {i: [i] for i in range(n)}


def compute_canonical_corr_matrix(F: torch.Tensor, eps: float = 1e-4) -> np.ndarray:
    """Return (D, D) matrix where entry (p,q) = ||C_pq||_F (Frobenius norm of
    the 2x2 cross-correlation block after whitening)."""
    X = _whiten_2d(_plane_samples(F), eps=eps)  # (D, M, 2)
    D, M, _ = X.shape
    C = torch.einsum("pmi,qmj->pqij", X, X) / M  # (D, D, 2, 2)
    # Frobenius norm of each 2x2 block = sqrt(rho1^2 + rho2^2)
    return C.pow(2).sum(dim=(-2, -1)).sqrt().numpy()


def load_and_infer(run_dir: str):
    """Load checkpoint, prepare data, run inference. Returns (F, F_hat, cfg, groups, S_p)."""
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
    train_ds, val_ds = train_val_split(windows, trial_info, cfg.val_split, cfg.seed)

    model = MLP(
        in_channels=windows.shape[1], d=cfg.d, hidden_dim=cfg.hidden_dim,
        depth=cfg.depth, dropout=getattr(cfg, "dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    (val_tensor,) = next(iter(loader))

    val_indices = list(val_ds.indices)
    trial_info_val = trial_info.iloc[val_indices].reset_index(drop=True)
    groups = _get_condition_groups(trial_info_val)

    with torch.no_grad():
        F = model(val_tensor)
        F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
        F_hat = _batch_rms_normalize(F)
        S_p = non_reversibility_S_per_plane(F_hat)
        s_ratio = compute_S_ratio(F_hat).item()
        cca_raw = block_cca_reg(F, eps=max(getattr(cfg, "block_cca_eps", 1e-4), 1e-4)).item()

    return F, F_hat, cfg, groups, S_p, s_ratio, cca_raw


def pearson_corr(F: torch.Tensor) -> np.ndarray:
    """Compute d×d Pearson correlation matrix."""
    K, d, T = F.shape
    Z = F.permute(0, 2, 1).reshape(K * T, d)
    Z = Z - Z.mean(dim=0, keepdim=True)
    Z = Z / (Z.std(dim=0, keepdim=True) + 1e-6)
    return ((Z.T @ Z) / Z.shape[0]).numpy()


def plot_cca_diagnostic(run_dir: str, out_path: str | None = None):
    """Create the full CCA diagnostic figure for one run."""
    print(f"Loading {run_dir}...")
    F, F_hat, cfg, groups, S_p, s_ratio, cca_raw = load_and_infer(run_dir)
    F_np = F_hat.numpy()
    K, d, T = F_np.shape
    D = d // 2

    # Experiment name from directory
    exp_name = os.path.basename(run_dir.rstrip("/"))

    if out_path is None:
        out_path = os.path.join(run_dir, "outputs", "cca_diagnostic.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ── Layout: 4 rows ────────────────────────────────────────────────
    #   Row 1-2: 2×4 rotation plane grid (condition-avg, time-coded)
    #   Row 3:   [canonical corr heatmap] [per-plane S bar] [Pearson heatmap]
    #   Row 4:   summary text
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1.1, 0.25],
                  hspace=0.35, wspace=0.35)

    cmap = plt.get_cmap("coolwarm")
    planes = F_np.reshape(K, D, 2, T)

    # ── Rows 1-2: rotation planes ─────────────────────────────────────
    for p in range(D):
        row, col = p // 4, p % 4
        ax = fig.add_subplot(gs[row, col])
        for cond_key in groups:
            idx_list = groups[cond_key]
            mean_traj = planes[idx_list, p].mean(axis=0)  # (2, T)
            x, y = mean_traj[0], mean_traj[1]
            for t in range(T - 1):
                ax.plot(x[t:t+2], y[t:t+2], color=cmap(t / (T - 1)),
                        lw=1.1, alpha=0.85)
        ax.axhline(0, color="k", lw=0.3, alpha=0.2)
        ax.axvline(0, color="k", lw=0.3, alpha=0.2)
        ax.set_title(f"Plane {p}  (S={float(S_p[p]):.3f})", fontsize=9)
        ax.set_xlabel(f"dim {2*p}", fontsize=7)
        ax.set_ylabel(f"dim {2*p+1}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.spines[["top", "right"]].set_visible(False)

    # ── Row 3, left: Canonical correlation heatmap ────────────────────
    ax_cca = fig.add_subplot(gs[2, 0:2])
    cca_mat = compute_canonical_corr_matrix(F, eps=max(getattr(cfg, "block_cca_eps", 1e-4), 1e-4))
    # mask diagonal for display
    np.fill_diagonal(cca_mat, np.nan)
    im = ax_cca.imshow(cca_mat, vmin=0, vmax=1.5, cmap="YlOrRd",
                       interpolation="nearest")
    ax_cca.set_xticks(range(D))
    ax_cca.set_yticks(range(D))
    ax_cca.set_xticklabels([f"P{i}" for i in range(D)], fontsize=8)
    ax_cca.set_yticklabels([f"P{i}" for i in range(D)], fontsize=8)
    ax_cca.set_title("Canonical correlation  $\\|C_{pq}\\|_F$\n(0 = independent)", fontsize=10)
    # annotate cells
    for i in range(D):
        for j in range(D):
            if i != j:
                val = cca_mat[i, j]
                color = "white" if val > 0.8 else "black"
                ax_cca.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=7, color=color)
            else:
                ax_cca.text(j, i, "-", ha="center", va="center", fontsize=7, color="gray")
    fig.colorbar(im, ax=ax_cca, fraction=0.046, pad=0.04)

    # ── Row 3, middle: Per-plane S bar chart ──────────────────────────
    ax_bar = fig.add_subplot(gs[2, 2])
    S_np = S_p.numpy()
    colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, D))
    bars = ax_bar.bar(range(D), S_np, color=colors_bar, edgecolor="black", lw=0.5)
    ax_bar.axhline(S_np.mean(), color="red", ls="--", lw=1, label=f"mean={S_np.mean():.3f}")
    ax_bar.axhline(S_np.min(), color="orange", ls=":", lw=1, label=f"min={S_np.min():.3f}")
    ax_bar.set_xticks(range(D))
    ax_bar.set_xticklabels([f"P{i}" for i in range(D)], fontsize=8)
    ax_bar.set_ylabel("S (per plane)", fontsize=9)
    ax_bar.set_title(f"Per-plane non-reversibility\nCV = {S_np.std()/S_np.mean():.3f}", fontsize=10)
    ax_bar.legend(fontsize=7, loc="lower right")
    ax_bar.spines[["top", "right"]].set_visible(False)

    # ── Row 3, right: Pearson correlation heatmap ─────────────────────
    ax_corr = fig.add_subplot(gs[2, 3])
    corr = pearson_corr(F)
    im2 = ax_corr.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
    ax_corr.set_xticks(range(d))
    ax_corr.set_yticks(range(d))
    ax_corr.set_xticklabels([f"d{i}" for i in range(d)], fontsize=5, rotation=90)
    ax_corr.set_yticklabels([f"d{i}" for i in range(d)], fontsize=5)
    ax_corr.set_title("Pearson correlation (dim-level)", fontsize=10)
    # draw plane blocks
    for p in range(D):
        rect = plt.Rectangle((2*p - 0.5, 2*p - 0.5), 2, 2, fill=False,
                              edgecolor="black", lw=1.5, linestyle="--")
        ax_corr.add_patch(rect)
    fig.colorbar(im2, ax=ax_corr, fraction=0.046, pad=0.04)

    # ── Row 4: summary text ───────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[3, :])
    ax_txt.axis("off")

    # compute off-diagonal pearson stats
    mask = ~np.eye(d, dtype=bool)
    offdiag_abs = np.abs(corr[mask])

    # compute cross-plane canonical corr stats
    cca_offdiag = cca_mat[~np.eye(D, dtype=bool) & ~np.isnan(cca_mat)]

    summary = (
        f"S_ratio = {s_ratio:.3f}    "
        f"S_total = {float(non_reversibility_S(_batch_rms_normalize(F))):.3f}    "
        f"S_min = {S_np.min():.3f}    "
        f"S_cv = {S_np.std()/S_np.mean():.3f}    "
        f"Block CCA = {cca_raw:.4f}    "
        f"Mean |canonical corr| = {cca_offdiag.mean():.3f}    "
        f"Max |canonical corr| = {cca_offdiag.max():.3f}    "
        f"Mean |Pearson| = {offdiag_abs.mean():.3f}"
    )
    cfg_str = (
        f"lambda_block_cca={getattr(cfg, 'lambda_block_cca', 0):.3g}    "
        f"lambda_plane_bt={getattr(cfg, 'lambda_plane_bt', 0):.3g}    "
        f"lambda_bt={getattr(cfg, 'lambda_bt', 0):.3g}    "
        f"s_objective={getattr(cfg, 's_objective', 'sum')}    "
        f"epochs={cfg.epochs}"
    )
    ax_txt.text(0.5, 0.7, summary, ha="center", va="center", fontsize=10,
                fontfamily="monospace", transform=ax_txt.transAxes)
    ax_txt.text(0.5, 0.2, cfg_str, ha="center", va="center", fontsize=9,
                fontfamily="monospace", color="gray", transform=ax_txt.transAxes)

    fig.suptitle(f"CCA Regularization Diagnostic: {exp_name}\n", fontsize=13, fontweight="bold")
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
    parser.add_argument("--out", default=None,
                        help="Output path for the figure (default: <run>/outputs/cca_diagnostic.png)")
    args = parser.parse_args()

    run_arg = args.run or args.run_flag
    run_dir = _resolve_run_dir(run_arg)
    print(f"Using run: {run_dir}")
    plot_cca_diagnostic(run_dir, out_path=args.out)


if __name__ == "__main__":
    main()
