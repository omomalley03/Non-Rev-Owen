"""
Visualise non-reversibility quality from a trained checkpoint.

Produces two figures:

Figure 1 — SCA-style trajectory plots (outputs/visualize.png)
  [0,0] Raw    — time-coloured (viridis), all K trials, top-2 variance channels
  [0,1] Embed  — time-coloured (viridis), all K trials, top-2 PCA dims
  [1,0] Raw    — trial-coded  (coolwarm), all K trials
  [1,1] Embed  — trial-coded  (coolwarm), all K trials, S_ratio in title
  [2,0] Signed-area histogram — raw vs embedding
  [2,1] Forward vs time-reversed overlay — embedding space

Figure 2 — Collapse diagnostics (outputs/collapse.png)
  [0,0] PCA explained variance of d=128 embedding (dimensional collapse check)
  [0,1] Covariance matrix heatmap (Barlow Twins target = identity)
  [1,0] Between-trial vs within-trial variance ratio per timepoint (trial collapse check)
  [1,1] Embedding norm distribution across trials

Usage
-----
    python visualize.py                        # uses checkpoints/best.pt
    python visualize.py --ckpt path/to.pt      # custom checkpoint
    python visualize.py --out outputs/vis.png  # custom output path
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

from config import Config
from data import load_mcmaze, gaussian_smooth, make_windows, train_val_split
from model import MLP
from loss import S_ratio as compute_S_ratio


# ── geometry helpers ──────────────────────────────────────────────────────────

def signed_area(x, y):
    """Shoelace signed area of a 2D trajectory. Positive = counterclockwise."""
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def pca2(Z):
    """Project (M, d) array onto its top-2 principal directions. Returns (M, 2)."""
    Z_c = Z - Z.mean(axis=0)
    _, _, Vh = np.linalg.svd(Z_c, full_matrices=False)
    return Z_c @ Vh[:2].T


# ── phasor builders ───────────────────────────────────────────────────────────

def raw_phasors(windows, ch_a, ch_b):
    """Z-score two channels → (K, 2, T)."""
    x = windows[:, ch_a, :].astype(np.float32)
    y = windows[:, ch_b, :].astype(np.float32)
    x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)
    y = (y - y.mean(-1, keepdims=True)) / (y.std(-1, keepdims=True) + 1e-6)
    return np.stack([x, y], axis=1)   # (K, 2, T)


def emb_phasors(F_hat):
    """Global PCA on (K, d, T) embeddings → top-2 dims → (K, 2, T)."""
    K, d, T = F_hat.shape
    Z = F_hat.transpose(0, 2, 1).reshape(K * T, d)
    Z2 = pca2(Z)
    return Z2.reshape(K, T, 2).transpose(0, 2, 1)   # (K, 2, T)


def all_signed_areas(phasors):
    """(K, 2, T) → (K,) signed area per trial."""
    return np.array([signed_area(p[0], p[1]) for p in phasors])


# ── SCA-style plot helpers ────────────────────────────────────────────────────

def plot_2D(ax, phasors, title, xlabel="Dim 1", ylabel="Dim 2"):
    """Replicate SCA utils.plot_2D: all K trials, time-coloured viridis.

    Each segment is coloured by its position in time (early=dark, late=bright).
    Matching the SCA implementation exactly.
    """
    K, _, T = phasors.shape
    cmap = plt.get_cmap("viridis")

    for k in range(K):
        x = phasors[k, 0, :]
        y = phasors[k, 1, :]
        for t in range(T - 1):
            ax.plot(x[t:t+2], y[t:t+2], color=cmap(t / (T - 1)), lw=0.6, alpha=0.6)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, T - 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="time (bins)", fraction=0.046, pad=0.04)


def plot_2D_K_coded(ax, phasors, s_ratio_val, title, xlabel="Dim 1", ylabel="Dim 2"):
    """Replicate SCA utils.plot_2D_K_coded: all K trials, trial-coded coolwarm.

    Each trial gets a fixed colour from coolwarm so you can track individual
    trajectories. S_ratio shown in title, matching SCA convention.
    """
    K, _, T = phasors.shape
    cmap = plt.get_cmap("coolwarm", K)

    for k in range(K):
        x = phasors[k, 0, :]
        y = phasors[k, 1, :]
        color = cmap(k / (K - 1))
        ax.plot(x, y, linestyle="-", marker=".", markersize=1.5,
                linewidth=0.7, color=color, alpha=0.7)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{title}\nS_ratio = {s_ratio_val:.4f}", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def plot_area_histogram(ax, areas_raw, areas_emb):
    """Signed-area distributions for raw vs embedding."""
    kw = dict(bins=30, alpha=0.55, density=True, edgecolor="none")
    ax.hist(areas_raw, color="steelblue",
            label=f"raw  μ={areas_raw.mean():.3f}  σ={areas_raw.std():.3f}", **kw)
    ax.hist(areas_emb, color="darkorange",
            label=f"emb  μ={areas_emb.mean():.3f}  σ={areas_emb.std():.3f}", **kw)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Signed area (shoelace formula)", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title("Signed-area distribution\n(non-zero mean → net circulation)", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_forward_reversed(ax, phasors, n_show, rng):
    """Forward (solid blue) vs time-reversed (dashed red) overlay, n_show trials."""
    idx = rng.choice(len(phasors), size=min(n_show, len(phasors)), replace=False)
    for i in idx:
        x_f, y_f = phasors[i, 0], phasors[i, 1]
        ax.plot(x_f, y_f, color="steelblue", alpha=0.3, lw=0.7)
        ax.plot(x_f[::-1], y_f[::-1], color="tomato", alpha=0.3, lw=0.7, ls="--")

    legend = [
        Line2D([0], [0], color="steelblue", lw=1.4, label="forward"),
        Line2D([0], [0], color="tomato",    lw=1.4, ls="--", label="time-reversed"),
    ]
    ax.legend(handles=legend, fontsize=7, loc="upper right")
    ax.set_title("Embedding — forward vs time-reversed", fontsize=9)
    ax.set_xlabel("PC 1", fontsize=8)
    ax.set_ylabel("PC 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.spines[["top", "right"]].set_visible(False)


# ── collapse diagnostics ──────────────────────────────────────────────────────

def plot_collapse_diagnostics(F_hat, out_path):
    """Four-panel figure checking for dimensional and trial collapse.

    F_hat : (K, d, T) numpy array — raw MLP embeddings

    Panel [0,0] — PCA explained variance
        Fit PCA on all (K*T, d) embedding snapshots. Plot cumulative explained
        variance vs number of PCs. Healthy: gradual ramp. Collapsed: step
        function where PC1 alone explains ~100%.

    Panel [0,1] — Embedding covariance matrix
        Normalise embeddings to zero mean / unit variance per dimension (same
        as the Barlow Twins regulariser), then plot the (d, d) cross-correlation
        matrix as a heatmap. Healthy: near-identity (diagonal, white off-diags).
        Collapsed: large off-diagonal blocks (redundant dimensions).

    Panel [1,0] — Between-trial vs within-trial variance ratio
        At each timepoint t, compute:
          between[t] = var of trial-mean embedding across trials     (signal)
          within[t]  = mean of per-trial variance across time        (noise proxy)
        Plot ratio between/(between+within) across time. Near 1 = trials are
        distinct; near 0 = all trials look the same (trial collapse).

    Panel [1,1] — Embedding norm distribution
        Histogram of ‖F_k‖_F per trial. If all norms are identical the model
        has found a trivial constant-magnitude solution.
    """
    from sklearn.decomposition import PCA

    K, d, T = F_hat.shape
    # (K*T, d) — all snapshots flattened
    Z = F_hat.transpose(0, 2, 1).reshape(K * T, d)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Embedding collapse diagnostics", fontsize=11, fontweight="bold")

    # ── [0,0] PCA explained variance ─────────────────────────────────────────
    ax = axes[0, 0]
    pca = PCA(n_components=min(d, K * T))
    pca.fit(Z)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

    ax.plot(np.arange(1, len(cumvar) + 1), cumvar, lw=1.5, color="steelblue")
    ax.axhline(95, color="tomato", lw=0.8, ls="--", label="95%")
    ax.axhline(99, color="darkorange", lw=0.8, ls="--", label="99%")
    n95 = int(np.searchsorted(cumvar, 95)) + 1
    n99 = int(np.searchsorted(cumvar, 99)) + 1
    ax.set_xlabel("Number of PCs", fontsize=8)
    ax.set_ylabel("Cumulative explained variance (%)", fontsize=8)
    ax.set_title(
        f"PCA explained variance (d={d})\n"
        f"95% in {n95} PCs,  99% in {n99} PCs",
        fontsize=9,
    )
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    # ── [0,1] Covariance matrix heatmap ──────────────────────────────────────
    ax = axes[0, 1]
    Z_norm = Z - Z.mean(axis=0)
    Z_norm = Z_norm / (Z_norm.std(axis=0) + 1e-6)
    Cov = (Z_norm.T @ Z_norm) / Z_norm.shape[0]   # (d, d)

    # Show top-32 dims for readability if d is large
    n_show = min(d, 32)
    im = ax.imshow(Cov[:n_show, :n_show], cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    off_diag_mean = float(np.abs(Cov - np.eye(d)).mean())
    ax.set_title(
        f"Embedding covariance (top {n_show} dims)\n"
        f"mean |Cov − I| = {off_diag_mean:.4f}  (0 = identity)",
        fontsize=9,
    )
    ax.set_xlabel("Embedding dim", fontsize=8)
    ax.set_ylabel("Embedding dim", fontsize=8)
    ax.tick_params(labelsize=7)

    # ── [1,0] Between vs within trial variance ────────────────────────────────
    ax = axes[1, 0]
    # F_hat: (K, d, T) — compute per-timepoint stats across trials
    trial_mean = F_hat.mean(axis=0, keepdims=True)          # (1, d, T)
    between = ((F_hat - trial_mean) ** 2).mean(axis=(0, 1)) # (T,) — var across trials & dims
    within  = F_hat.var(axis=2).mean(axis=(0, 1))           # scalar — mean temporal var

    ratio = between / (between + within + 1e-8)             # (T,) ∈ [0, 1]
    t_axis = np.arange(T)
    ax.plot(t_axis, ratio, lw=1.2, color="steelblue")
    ax.axhline(ratio.mean(), color="tomato", lw=0.8, ls="--",
               label=f"mean = {ratio.mean():.3f}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time bin", fontsize=8)
    ax.set_ylabel("Between / (Between + Within)", fontsize=8)
    ax.set_title(
        "Trial discriminability over time\n"
        "(1 = fully distinct trials,  0 = all trials identical)",
        fontsize=9,
    )
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    # ── [1,1] Embedding norm distribution ────────────────────────────────────
    ax = axes[1, 1]
    norms = np.linalg.norm(F_hat.reshape(K, -1), axis=1)   # (K,) Frobenius norm per trial
    ax.hist(norms, bins=30, color="steelblue", alpha=0.8, edgecolor="none")
    ax.axvline(norms.mean(), color="tomato", lw=1, ls="--",
               label=f"mean = {norms.mean():.2f}")
    ax.set_xlabel("‖F_k‖_F  (Frobenius norm per trial)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    cv = norms.std() / norms.mean()
    ax.set_title(
        f"Embedding norm distribution\n"
        f"CV = {cv:.3f}  (0 = all trials identical magnitude)",
        fontsize=9,
    )
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None,
                        help="Run to visualise: an integer (1 = most recent) or "
                             "a full path (e.g. runs/20260429_143022_d10_...). "
                             "Omit to use the most recent run.")
    parser.add_argument("--n_fwdrev", type=int, default=30,
                        help="Trials to show in forward/reversed panel")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── discover runs ─────────────────────────────────────────────────────────
    runs_root = "runs"
    if not os.path.isdir(runs_root):
        raise FileNotFoundError("No 'runs/' directory found. Run `python main.py` first.")

    completed = sorted(
        [os.path.join(runs_root, d) for d in os.listdir(runs_root)
         if os.path.isfile(os.path.join(runs_root, d, "checkpoints", "best.pt"))],
        key=os.path.getmtime, reverse=True,   # newest first
    )
    if not completed:
        raise FileNotFoundError("No completed runs found in 'runs/'. Run `python main.py` first.")

    print("Available runs (newest first):")
    for i, r in enumerate(completed, 1):
        print(f"  [{i}] {os.path.basename(r)}")
    print()

    # ── resolve run directory ─────────────────────────────────────────────────
    if args.run is None:
        run_dir = completed[0]
        print(f"Using most recent run: {os.path.basename(run_dir)}")
    elif args.run.isdigit():
        idx = int(args.run) - 1
        if idx < 0 or idx >= len(completed):
            raise ValueError(f"--run {args.run} out of range (1–{len(completed)})")
        run_dir = completed[idx]
        print(f"Using run [{args.run}]: {os.path.basename(run_dir)}")
    else:
        run_dir = args.run
        print(f"Using run: {os.path.basename(run_dir)}")

    ckpt_path    = os.path.join(run_dir, "checkpoints", "best.pt")
    out_path     = os.path.join(run_dir, "outputs", "visualize.png")
    collapse_out = os.path.join(run_dir, "outputs", "collapse.png")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at '{ckpt_path}'.")

    # ── load checkpoint + config ──────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # ── reproduce data pipeline ───────────────────────────────────────────────
    print("Loading data…")
    spikes_raw, bin_width_s, trial_info, time_index_s = load_mcmaze(cfg.nwb_path, cfg.bin_ms)
    N = spikes_raw.shape[0]
    sigma_samples = round((cfg.sigma_ms * 1e-3) / bin_width_s)
    X_smooth = gaussian_smooth(spikes_raw, sigma_samples)
    windows = make_windows(
        X_smooth, trial_info, time_index_s, bin_width_s,
        strategy=cfg.window_strategy, window_size=cfg.window_size,
    )
    _, val_ds = train_val_split(windows, cfg.val_split, cfg.seed)

    loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    (val_tensor,) = next(iter(loader))
    val_np = val_tensor.numpy()   # (K, N, T)
    K = len(val_np)
    print(f"Val set: {K} trials  |  N={N}  |  T={cfg.window_size}")

    # ── embed with trained model ──────────────────────────────────────────────
    print("Computing embeddings…")
    model = MLP(in_channels=N, d=cfg.d, hidden_dim=cfg.hidden_dim, depth=cfg.depth)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        F_hat = model(val_tensor)                        # (K, d, T) tensor
        s_ratio_val = compute_S_ratio(F_hat).item()
        F_hat = F_hat.numpy()

    # ── collapse diagnostics ──────────────────────────────────────────────────
    plot_collapse_diagnostics(F_hat, collapse_out)

    # ── phasor arrays ─────────────────────────────────────────────────────────
    ch_var = val_np.var(axis=(0, 2))                     # (N,) variance per channel
    top2_ch = np.argsort(ch_var)[-2:][::-1]
    ch_a, ch_b = int(top2_ch[0]), int(top2_ch[1])
    print(f"Raw phasor channels: {ch_a} (var={ch_var[ch_a]:.4f}),  "
          f"{ch_b} (var={ch_var[ch_b]:.4f})")

    phasors_raw = raw_phasors(val_np, ch_a, ch_b)       # (K, 2, T)
    phasors_emb = emb_phasors(F_hat)                     # (K, 2, T)

    areas_raw = all_signed_areas(phasors_raw)            # (K,)
    areas_emb = all_signed_areas(phasors_emb)            # (K,)

    s_ratio_raw_val = float(np.sum(areas_raw > 0)) / K  # fraction with positive area (proxy)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(11, 13))
    fig.suptitle(
        f"SCA-style non-reversibility diagnostics\n"
        f"(checkpoint epoch {ckpt['epoch']}, {K} val trials,  "
        f"S_ratio={s_ratio_val:.4f})",
        fontsize=11, fontweight="bold",
    )

    # Row 0 — time-coloured (viridis): replicates SCA plot_2D
    plot_2D(axes[0, 0], phasors_raw,
            title=f"Raw — time-coloured (ch {ch_a} vs ch {ch_b})",
            xlabel=f"Channel {ch_a} (z-scored)",
            ylabel=f"Channel {ch_b} (z-scored)")

    plot_2D(axes[0, 1], phasors_emb,
            title="Embedding — time-coloured (top-2 PCA)",
            xlabel="PC 1", ylabel="PC 2")

    # Row 1 — trial-coded (coolwarm): replicates SCA plot_2D_K_coded
    plot_2D_K_coded(axes[1, 0], phasors_raw,
                    s_ratio_val=s_ratio_raw_val,
                    title=f"Raw — trial-coded (ch {ch_a} vs ch {ch_b})",
                    xlabel=f"Channel {ch_a} (z-scored)",
                    ylabel=f"Channel {ch_b} (z-scored)")

    plot_2D_K_coded(axes[1, 1], phasors_emb,
                    s_ratio_val=s_ratio_val,
                    title="Embedding — trial-coded (top-2 PCA)",
                    xlabel="PC 1", ylabel="PC 2")

    # Row 2 — our irreversibility diagnostics
    plot_area_histogram(axes[2, 0], areas_raw, areas_emb)

    plot_forward_reversed(axes[2, 1], phasors_emb, args.n_fwdrev, rng)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    # ── console summary ───────────────────────────────────────────────────────
    print(f"\nS_ratio (embedding, all val pairs): {s_ratio_val:.4f}")
    print(f"\nSigned-area summary ({K} val trials):")
    print(f"  Raw       μ={areas_raw.mean():+.4f}  σ={areas_raw.std():.4f}"
          f"  |μ|/σ = {abs(areas_raw.mean()) / areas_raw.std():.3f}")
    print(f"  Embedding μ={areas_emb.mean():+.4f}  σ={areas_emb.std():.4f}"
          f"  |μ|/σ = {abs(areas_emb.mean()) / areas_emb.std():.3f}")
    print("  (|μ|/σ > 1 suggests consistent circulation across trials)")


if __name__ == "__main__":
    main()
