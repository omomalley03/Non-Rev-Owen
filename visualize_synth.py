"""
Visualise non-reversibility quality for synthetic rotation data.

Each diagnostic plot is saved as its own PNG in the run's `outputs/` dir:

  01_raw_time_coded.png           — raw input, all trials, time-coded (top-2 input PCA)
  02_embed_planes.png             — one subplot per 2D rotation plane, time-coded
  07_covariance_heatmap.png       — embedding correlation matrix

Usage
-----
    python visualize_synth.py               # most recent run
    python visualize_synth.py --run 2       # 2nd most recent
    python visualize_synth.py --run runs/x  # explicit path
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import Config
from data import train_val_split
from model import MLP
from loss import S_ratio as compute_S_ratio


# ── geometry helpers ──────────────────────────────────────────────────────────

def signed_area(x, y):
    """Shoelace signed area of a 2D trajectory. Positive = counterclockwise."""
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def all_signed_areas(phasors):
    """(K, 2, T) → (K,) signed area per trial."""
    return np.array([signed_area(p[0], p[1]) for p in phasors])


def _fit_pca2(X):
    """Fit top-2 PCA on (M, D) matrix. Returns (mean, Vh2) for later projection."""
    mean = X.mean(axis=0)
    _, _, Vh = np.linalg.svd(X - mean, full_matrices=False)
    return mean, Vh[:2]


def _apply_pca2(X, mean, Vh2):
    """Project (M, D) onto precomputed top-2 PCA axes → (M, 2)."""
    return (X - mean) @ Vh2.T


def _windows_to_pca2(windows, mean, Vh2):
    """Project (K, N, T) windows through top-2 PCA → (K, 2, T)."""
    K, N, T = windows.shape
    Z = windows.transpose(0, 2, 1).reshape(K * T, N)   # (K*T, N)
    proj = _apply_pca2(Z, mean, Vh2)                    # (K*T, 2)
    return proj.reshape(K, T, 2).transpose(0, 2, 1)     # (K, 2, T)


# ── plot helpers ──────────────────────────────────────────────────────────────

def _plot_all_trials_time_coded(phasors, title, xlabel, ylabel, out_path,
                                 n_show=60, cmap_name="viridis", seed=0):
    """Plot individual trial trajectories colour-coded by time.

    With hundreds of trials, all at once is unreadable; subsample to n_show.

    Math
    ----
    Each segment (phasor[k, :, t], phasor[k, :, t+1]) is coloured by t/(T-1).
    """
    rng = np.random.default_rng(seed)
    K, _, T = phasors.shape
    idx = rng.choice(K, size=min(n_show, K), replace=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap(cmap_name)

    for k in idx:
        x, y = phasors[k, 0], phasors[k, 1]
        for t in range(T - 1):
            ax.plot(x[t:t+2], y[t:t+2], color=cmap(t / (T - 1)),
                    lw=0.8, alpha=0.5)

    ax.axhline(0, color="k", lw=0.4, alpha=0.25)
    ax.axvline(0, color="k", lw=0.4, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{title}\n({min(n_show, K)} of {K} trials shown)", fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, T - 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="time (bins)", fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _plot_planes_time_coded(F_hat, s_ratio, out_path,
                            n_show=60, cmap_name="viridis", seed=0):
    """Subplot grid: one panel per 2D rotation plane, trials time-coded."""
    K, d, T = F_hat.shape
    D = d // 2
    rng = np.random.default_rng(seed)
    idx = rng.choice(K, size=min(n_show, K), replace=False)
    cmap = plt.get_cmap(cmap_name)

    ncols = min(D, 4)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)

    planes = F_hat.reshape(K, D, 2, T)
    per_plane_areas = np.array([
        [signed_area(planes[k, p, 0], planes[k, p, 1]) for k in range(K)]
        for p in range(D)
    ])

    for p in range(D):
        ax = axes[p // ncols, p % ncols]
        for k in idx:
            x, y = planes[k, p, 0], planes[k, p, 1]
            for t in range(T - 1):
                ax.plot(x[t:t+2], y[t:t+2], color=cmap(t / (T - 1)),
                        lw=0.8, alpha=0.5)

        ax.axhline(0, color="k", lw=0.4, alpha=0.25)
        ax.axvline(0, color="k", lw=0.4, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        mu_a = per_plane_areas[p].mean()
        ax.set_title(f"Plane {p}  (dims {2*p}, {2*p+1})\n"
                     f"mean area = {mu_a:+.3f}", fontsize=9)
        ax.set_xlabel(f"dim {2*p}", fontsize=8)
        ax.set_ylabel(f"dim {2*p+1}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="datalim")

    for p in range(D, nrows * ncols):
        axes[p // ncols, p % ncols].set_visible(False)

    fig.suptitle(f"Embedding rotation planes  (ζ = {s_ratio:.4f},  "
                 f"{min(n_show, K)} of {K} trials)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_signed_area_histogram(areas_raw, areas_emb, out_path):
    """Distribution of shoelace signed area per trial, raw PCA vs embedding PCA.

    Math
    ----
    A_k = ½ Σ_{t} (x_k[t]·y_k[t+1] − x_k[t+1]·y_k[t])
    |μ_A|/σ_A > 1 ⇒ consistent net circulation across trials.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    kw = dict(bins=30, alpha=0.55, density=True, edgecolor="none")
    ax.hist(areas_raw, color="steelblue",
            label=f"raw  μ={areas_raw.mean():+.3f}  σ={areas_raw.std():.3f}", **kw)
    ax.hist(areas_emb, color="darkorange",
            label=f"emb  μ={areas_emb.mean():+.3f}  σ={areas_emb.std():.3f}", **kw)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Signed area  A_k  (shoelace formula)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Signed-area distribution\n(non-zero mean → net circulation)", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_pca_explained_variance(F_hat, out_path):
    """Cumulative PCA explained variance of the (K*T, d) embedding snapshots.

    Math
    ----
    Z ∈ R^(M×d), M = K*T.  Centred Z̃ = Z − mean.
    cumvar(j) = (Σ_{i≤j} σ_i²) / (Σ_i σ_i²)
    A large step at PC1 ⇒ dimensional collapse (embedding is near 1D).
    """
    from sklearn.decomposition import PCA

    K, d, T = F_hat.shape
    Z = F_hat.transpose(0, 2, 1).reshape(K * T, d)

    fig, ax = plt.subplots(figsize=(6, 4))
    pca = PCA(n_components=min(d, K * T))
    pca.fit(Z)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

    ax.plot(np.arange(1, len(cumvar) + 1), cumvar, lw=1.5, color="steelblue", marker="o", ms=4)
    ax.axhline(95, color="tomato",     lw=0.8, ls="--", label="95%")
    ax.axhline(99, color="darkorange", lw=0.8, ls="--", label="99%")
    n95 = int(np.searchsorted(cumvar, 95)) + 1
    n99 = int(np.searchsorted(cumvar, 99)) + 1
    ax.set_xlabel("Number of PCs", fontsize=9)
    ax.set_ylabel("Cumulative explained variance (%)", fontsize=9)
    ax.set_title(
        f"Embedding PCA  (d={d})\n95% in {n95} PCs,  99% in {n99} PCs",
        fontsize=10,
    )
    ax.set_xticks(np.arange(1, d + 1))
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

def plot_covariance_heatmap(F_hat, out_path):
    """Empirical correlation matrix of the d embedding dims.

    Math
    ----
    Z̃_{m,i} = (Z_{m,i} − μ_i) / σ_i
    Corr_{ij} = (1/M) Σ_m Z̃_{m,i} Z̃_{m,j}
    Barlow-Twins target: Corr = I.  Off-diagonal mass ⇒ redundant dims.
    """
    K, d, T = F_hat.shape
    Z = F_hat.transpose(0, 2, 1).reshape(K * T, d)
    # Z = Z - Z.mean(axis=0)
    # Z = Z / (Z.std(axis=0) + 1e-6)
    Cov = (Z.T @ Z) / Z.shape[0]


    n_show = min(d, 32)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Cov[:n_show, :n_show], cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    off_diag_mean = float(np.abs(Cov - np.eye(d)).mean())
    ax.set_title(
        f"Embedding correlation  (top {n_show} of {d} dims)\n"
        f"mean |Corr − I| = {off_diag_mean:.4f}  (0 = identity)",
        fontsize=10,
    )
    ax.set_xlabel("Embedding dim", fontsize=9)
    ax.set_ylabel("Embedding dim", fontsize=9)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")



def plot_norm_distribution(F_hat, out_path):
    """Distribution of per-trial Frobenius norms ‖F_k‖_F.

    Math
    ----
    ‖F_k‖_F = √(Σ_{i,t} F_k[i,t]²),  CV = std/mean across k.
    CV → 0 ⇒ all trials have the same scale (trivial constant-magnitude solution).
    """
    K = F_hat.shape[0]
    norms = np.linalg.norm(F_hat.reshape(K, -1), axis=1)
    cv = norms.std() / (norms.mean() + 1e-12)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(norms, bins=30, color="steelblue", alpha=0.8, edgecolor="none")
    ax.axvline(norms.mean(), color="tomato", lw=1, ls="--",
               label=f"mean = {norms.mean():.2f}")
    ax.set_xlabel("‖F_k‖_F", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(
        f"Embedding norm distribution\nCV = {cv:.3f}  (0 = identical magnitudes)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── main diagnostic entry point ───────────────────────────────────────────────

def make_diagnostic_plots_synth(
    model,
    val_ds,
    cfg: Config,
    run_dir: str,
):
    """Compute embeddings on val_ds and write all diagnostic PNGs to run_dir/outputs/.

    Parameters
    ----------
    model     : trained MLP (any device; moved to CPU/eval here)
    val_ds    : torch Subset of the windows TensorDataset
    cfg       : Config used to train the model
    run_dir   : run directory; outputs written to {run_dir}/outputs/
    """
    out_dir = os.path.join(run_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    model = model.cpu().eval()

    loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    (val_tensor,) = next(iter(loader))
    val_np = val_tensor.numpy()          # (K, N_in, T)  N_in = N or 2N with velocity
    K = val_np.shape[0]

    print(f"Val set: {K} trials  |  input shape: {val_np.shape}")

    # ── embeddings ───────────────────────────────────────────────────────────
    print("Computing embeddings…")
    with torch.no_grad():
        F_hat_t = model(val_tensor)
        s_ratio_val = compute_S_ratio(F_hat_t).item()
        F_hat = F_hat_t.numpy()          # (K, d, T)

    # ── raw input PCA (spike channels only — first N cols if velocity was used) ──
    N_in = val_np.shape[1]
    N_spikes = N_in // 2 if getattr(cfg, "use_velocity", False) else N_in
    spikes_only = val_np[:, :N_spikes, :]   # (K, N_spikes, T)

    raw_flat = spikes_only.transpose(0, 2, 1).reshape(K * spikes_only.shape[2], N_spikes)
    raw_pca_mean, raw_pca_Vh2 = _fit_pca2(raw_flat)
    phasors_raw = _windows_to_pca2(spikes_only, raw_pca_mean, raw_pca_Vh2)  # (K, 2, T)

    # ── plots ─────────────────────────────────────────────────────────────────
    _plot_all_trials_time_coded(
        phasors_raw,
        title="Raw rotations input — top-2 PCA",
        xlabel="PC 1", ylabel="PC 2",
        out_path=os.path.join(out_dir, "01_raw_time_coded.png"),
    )
    _plot_planes_time_coded(
        F_hat, s_ratio_val,
        out_path=os.path.join(out_dir, "02_embed_planes.png"),
    )
    plot_covariance_heatmap(
        F_hat, out_path=os.path.join(out_dir, "07_covariance_heatmap.png"),
    )

    print(f"\nS_ratio (all val pairs): {s_ratio_val:.4f}  (max ≈ 1.0 for perfect rotation)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _resolve_run_dir(arg_run):
    runs_root = "synth_runs"
    if not os.path.isdir(runs_root):
        raise FileNotFoundError("No 'runs/' directory found.")
    completed = sorted(
        [os.path.join(runs_root, d) for d in os.listdir(runs_root)
         if os.path.isfile(os.path.join(runs_root, d, "checkpoints", "best.pt"))],
        key=os.path.getmtime, reverse=True,
    )
    if not completed:
        raise FileNotFoundError("No completed runs found in 'runs/'.")

    print("Available runs (newest first):")
    for i, r in enumerate(completed, 1):
        print(f"  [{i}] {os.path.basename(r)}")
    print()

    if arg_run is None:
        return completed[0]
    if arg_run.isdigit():
        idx = int(arg_run) - 1
        if idx < 0 or idx >= len(completed):
            raise ValueError(f"--run {arg_run} out of range (1–{len(completed)})")
        return completed[idx]
    return arg_run


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None,
                        help="Integer (1=most recent) or explicit path.")
    parser.add_argument("--data", default="rotations.npy",
                        help="Path to synthetic data .npy file (default: rotations.npy).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run)
    print(f"Using run: {os.path.basename(run_dir)}")

    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Load synthetic data with the same preprocessing used during training.
    print(f"Loading synthetic data from {args.data} …")
    raw = np.load(args.data)
    windows = np.transpose(raw, (0, 2, 1)).astype(np.float32)   # (K, N, T)
    N = windows.shape[1]

    # Apply velocity augmentation if the checkpoint was trained with it.
    if getattr(cfg, "use_velocity", False):
        velocity = np.diff(windows, axis=2, prepend=windows[:, :, :1])  # (K, N, T)
        windows = np.concatenate([windows, velocity], axis=1)            # (K, 2N, T)
        N_in = 2 * N
        print(f"  Velocity augmentation ON — input: (K, {N_in}, T)")
    else:
        N_in = N
        print(f"  Velocity augmentation OFF — input: (K, {N_in}, T)")

    _, val_ds = train_val_split(windows, None, cfg.val_split, cfg.seed)

    model = MLP(in_channels=N_in, d=cfg.d, hidden_dim=cfg.hidden_dim, depth=cfg.depth)
    model.load_state_dict(ckpt["model_state_dict"])

    make_diagnostic_plots_synth(
        model=model,
        val_ds=val_ds,
        cfg=cfg,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
