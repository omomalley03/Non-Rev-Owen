"""
Visualise non-reversibility quality for synthetic rotation data.

Each diagnostic plot is saved as its own PNG in the run's `outputs/` dir:

  01_raw_time_coded.png           — raw input, time-coded (top-2 input PCA)
  02_embed_planes_time_coded.png  — one subplot per 2D rotation plane, time-coded
  07_covariance_heatmap.png       — embedding correlation matrix
  08_between_within_variance.png  — trial-discriminability over time
  09_embedding_norm_distribution.png — embedding norm distribution

Usage
-----
    python visualize_synth.py               # most recent run
    python visualize_synth.py --run 2       # 2nd most recent
    python visualize_synth.py --run synth_runs/x  # explicit path
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from torch.utils.data import DataLoader, TensorDataset, random_split

from config import Config
from paths import SYNTH_RUNS_DIR
from model import MLP
from loss import S_ratio as compute_S_ratio, _batch_rms_normalize, _whiten_2d, _plane_samples


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


def load_synthetic_windows(cfg: Config, data_path: str | None = None) -> np.ndarray:
    """Load synthetic rotations as (K, N, T), matching main_synth.py."""
    path = data_path or getattr(cfg, "synth_data_path", "rotations.npy")
    windows = np.load(path).astype(np.float32)
    windows = np.transpose(windows, (0, 2, 1))  # source is (K, T, N)

    noise_std = getattr(cfg, "synth_noise_std", 0.0)
    if noise_std > 0:
        rng = np.random.default_rng(cfg.seed)
        noise = rng.normal(0.0, noise_std, size=windows.shape).astype(np.float32)
        windows = windows + noise

    return windows


def train_val_split_synth(windows: np.ndarray, val_frac: float, seed: int):
    """Random train/val split for synthetic windows."""
    tensor = torch.from_numpy(windows)
    full_ds = TensorDataset(tensor)
    n_val = max(1, int(len(tensor) * val_frac))
    n_train = len(tensor) - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(full_ds, [n_train, n_val], generator=generator)


# ── plot helpers ──────────────────────────────────────────────────────────────

def _pairwise_zeta(F_hat: np.ndarray) -> np.ndarray:
    """Compute ζ (S_ratio) for every (dim 2i, dim 2j+1) pair.

    Returns a (D, D) float array where entry [i, j] is the S_ratio of the
    2D plane formed by dims (2*i, 2*j+1).  Diagonal entries are the native
    rotation planes; off-diagonal are cross-plane surrogate pairs.
    """
    K, d, T = F_hat.shape
    D = d // 2
    zeta = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            plane = np.stack([F_hat[:, 2*i, :], F_hat[:, 2*j+1, :]], axis=1)  # (K,2,T)
            t = torch.from_numpy(plane)
            zeta[i, j] = compute_S_ratio(_batch_rms_normalize(t)).item()
    return zeta

def _plot_all_trials_time_coded(phasors, title, xlabel, ylabel, out_path,
                                 cmap_name="coolwarm", seed=0):
    """Plot individual trial trajectories colour-coded by time."""
    rng = np.random.default_rng(seed)
    K, _, T = phasors.shape
    n_show = int(K * 1.0)
    idx = rng.choice(K, size=min(n_show, K), replace=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap(cmap_name)

    norm = plt.Normalize(0, T - 1)
    seg_t = np.arange(T - 1)
    for k in idx:
        x, y = phasors[k, 0], phasors[k, 1]
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)        # (T, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)     # (T-1, 2, 2)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=0.8, alpha=0.5)
        lc.set_array(seg_t)
        ax.add_collection(lc)
    ax.autoscale()

    ax.axhline(0, color="k", lw=0.4, alpha=0.25)
    ax.axvline(0, color="k", lw=0.4, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=10)
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
                             cmap_name="coolwarm", seed=0):
    """Subplot grid: one panel per 2D rotation plane, trials time-coded."""
    K, d, T = F_hat.shape
    n_show = int(K)
    D = d // 2
    rng = np.random.default_rng(seed)
    idx = rng.choice(K, size=min(n_show, K), replace=False)
    cmap = plt.get_cmap(cmap_name)
    zeta = _pairwise_zeta(F_hat)   # (D, D); diagonal = per-plane ζ

    ncols = min(D, 4)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)

    planes = F_hat.reshape(K, D, 2, T)
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
        ax.set_title("Linear embedding")
        # ax.set_title(f"Plane {p}  (dims {2*p}, {2*p+1})  ζ={zeta[p,p]:.2f}", fontsize=9)
        ax.set_xlabel(f"dim {2*p}", fontsize=8)
        ax.set_ylabel(f"dim {2*p+1}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="datalim")

    for p in range(D, nrows * ncols):
        axes[p // ncols, p % ncols].set_visible(False)

    fig.suptitle(f"Embedding — time-coded  (ζ = {s_ratio:.2f})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _plot_dim_grid(F_hat, s_ratio, out_path,
                    cmap_name="coolwarm", seed=0):
    """D×D grid where cell (i,j) plots dim 2*i vs dim 2*j+1, time-coded.

    Diagonal (i==j) = native rotation planes.
    Off-diagonal = cross-plane pairings; should look like Lissajous/scatter if
    the planes are independent, or structured circles if they share a source.
    """
    K, d, T = F_hat.shape
    n_show = int(K * 1.0)
    D = d // 2
    rng = np.random.default_rng(seed)
    idx = rng.choice(K, size=min(n_show, K), replace=False)
    cmap = plt.get_cmap(cmap_name)
    zeta = _pairwise_zeta(F_hat)   # (D, D)

    fig, axes = plt.subplots(D, D, figsize=(3.2 * D, 3.2 * D), squeeze=False)

    for i in range(D):
        for j in range(D):
            ax = axes[i][j]
            x_dim, y_dim = 2 * i, 2 * j + 1
            for k in idx:
                xv, yv = F_hat[k, x_dim], F_hat[k, y_dim]
                for t in range(T - 1):
                    ax.plot(xv[t:t+2], yv[t:t+2], color=cmap(t / (T - 1)),
                            lw=0.6, alpha=0.45)

            ax.axhline(0, color="k", lw=0.3, alpha=0.2)
            ax.axvline(0, color="k", lw=0.3, alpha=0.2)
            ax.set_aspect("equal", adjustable="datalim")
            ax.tick_params(labelsize=6)

            # highlight diagonal (native planes) with a box
            if i == j:
                for spine in ax.spines.values():
                    spine.set_edgecolor("steelblue")
                    spine.set_linewidth(1.5)
            else:
                ax.spines[["top", "right"]].set_visible(False)

            ax.set_xlabel(f"dim {x_dim}", fontsize=12)
            ax.set_ylabel(f"dim {y_dim}", fontsize=12)
            # ax.set_title(f"({x_dim},{y_dim})  ζ={zeta[i,j]:.2f}", fontsize=7, pad=2)

    # fig.suptitle(f"Dim grid — time-coded  (ζ = {s_ratio:.2f})\n",
    #             #  f"cell (i,j): dim 2i vs dim 2j+1   [diagonal = native planes]",
    #              fontsize=10)
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
    Z = Z - Z.mean(axis=0)

    Z = Z / (Z.std(axis=0) + 1e-6)

    Corr = (Z.T @ Z) / Z.shape[0]

    n_show = min(d, 32)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Corr[:n_show, :n_show], cmap="RdBu_r", # vmin=-1, vmax=1,
                   interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    off_diag_mean = float(np.abs(Corr - np.eye(d)).mean())
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


def plot_between_within_variance(F_hat, out_path):
    """Trial-discriminability ratio over time, matching visualize.py."""
    T = F_hat.shape[2]
    trial_mean = F_hat.mean(axis=0, keepdims=True)
    between = ((F_hat - trial_mean) ** 2).mean(axis=(0, 1))
    within = F_hat.var(axis=2).mean(axis=(0, 1))
    ratio = between / (between + within + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 4))
    t_axis = np.arange(T)
    ax.plot(t_axis, ratio, lw=1.2, color="steelblue")
    ax.axhline(ratio.mean(), color="tomato", lw=0.8, ls="--",
               label=f"mean = {ratio.mean():.3f}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time bin", fontsize=9)
    ax.set_ylabel("B(t) / (B(t) + W)", fontsize=9)
    ax.set_title(
        "Trial discriminability over time\n"
        "(1 = fully distinct trials, 0 = all trials identical)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

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

def plot_cca_grid(F_hat, out_path):
    """Penalize linear dependence between whole 2D planes."""
    X = _whiten_2d(_plane_samples(F_hat))             # (D, M, 2)
    # print(X.shape)
    D, M, _ = X.shape
    # if D < 2:
    #     print(F.new_tensor(0.0))
    C = torch.einsum("pmi,qmj->pqij", X, X) / M            # (D, D, 2, 2)
    eye = torch.eye(D, dtype=torch.bool)
    # print(C[~eye].pow(2).sum() / (D * (D - 1)))
    fig, axes = plt.subplots(D,D,figsize=(6, 4))
    for i in range(D):
        for j in range(D):
            ax = axes[i,j]
            ax.imshow(C[i,j],vmin=-1,vmax=1,cmap="RdBu_r")
            score = C[i,j].pow(2).sum()
            ax.set_title(f"reg.={score:.2f}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

def plot_conv_kernels(model, out_path):
    # depthwise zero-phase weights: (out_ch, 1, kernel_size); plot the effective
    # palindromic time kernel (w + flip(w)) for a sample of output channels.
    out_ch, _, kernel_size = model.temporal_conv.weight.shape
    weights = model.temporal_conv.weight.detach().numpy()[:, 0, :]   # (out_ch, k)
    weights_p = weights + np.flip(weights, -1)
    n_show = min(out_ch, 64)
    rows = int(n_show ** 0.5)
    fig, axes = plt.subplots(rows, rows, squeeze=False, figsize=(2 * rows, 1.5 * rows))
    for i in range(rows):
        for j in range(rows):
            k = i * rows + j
            ax = axes[i, j]
            ax.plot(weights_p[k])
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    (val_tensor,) = next(iter(loader))
    val_np = val_tensor.numpy()          # (K, N_in, T)  N_in = N or 2N with velocity
    K = val_np.shape[0]

    print(f"Val set: {K} trials  |  input shape: {val_np.shape}")

    # ── embeddings ───────────────────────────────────────────────────────────
    # Run the forward + S_ratio on GPU when available: at T=2000 the per-timepoint
    # forward and the K^2 pairwise S_ratio einsum are far too slow single-threaded
    # on CPU. Bring F_hat_t back to CPU for the NumPy-based plotting below.
    print(f"Computing embeddings… (device={device})")
    with torch.no_grad():
        F_hat_t = model(val_tensor.to(device))
        F_hat_t = F_hat_t - F_hat_t.mean(dim=cfg.F_mean_axis, keepdim=True)
        s_ratio_val = compute_S_ratio(F_hat_t).item()
        F_hat_t = F_hat_t.cpu()
        F_hat = F_hat_t.numpy()          # (K, d, T)

    # ── raw input PCA ──────────────────────────────────────────────────────────
    N_in = val_np.shape[1]
    raw_input = val_np[:, :N_in, :]

    raw_flat = raw_input.transpose(0, 2, 1).reshape(K * raw_input.shape[2], N_in)
    raw_pca_mean, raw_pca_Vh2 = _fit_pca2(raw_flat)
    phasors_raw = _windows_to_pca2(raw_input, raw_pca_mean, raw_pca_Vh2)  # (K, 2, T)

    # ── plots ─────────────────────────────────────────────────────────────────
    _plot_all_trials_time_coded(
        phasors_raw,
        title="Raw — time-coded  (top-2 PCA)",
        xlabel="PC 1", ylabel="PC 2",
        out_path=os.path.join(out_dir, "01_raw_time_coded.png"),
    )
    _plot_planes_time_coded(
        F_hat, s_ratio_val,
        out_path=os.path.join(out_dir, "02_embed_planes_time_coded.png"),
    )
    _plot_dim_grid(
        F_hat, s_ratio_val,
        out_path=os.path.join(out_dir, "03_dim_grid_time_coded.png"),
    )
    plot_covariance_heatmap(
        F_hat, out_path=os.path.join(out_dir, "07_covariance_heatmap.png"),
    )
    plot_between_within_variance(
        F_hat, out_path=os.path.join(out_dir, "08_between_within_variance.png"),
    )
    plot_norm_distribution(
        F_hat, out_path=os.path.join(out_dir, "09_embedding_norm_distribution.png"),
    )

    plot_conv_kernels(model=model, out_path=os.path.join(out_dir, "11_conv_kernels.png"))
    

    from visualize_pairwise_s import plot_pairwise_s
    F_hat_rms = _batch_rms_normalize(F_hat_t)
    plot_pairwise_s(
        run_dir,
        out_path=os.path.join(out_dir, "pairwise_s.png"),
        F_hat=F_hat_rms,
        cfg=cfg,
    )

    plot_cca_grid(
        F_hat_t, out_path = os.path.join(out_dir,"10_cca_grid.png")
    )

    print(f"\nS_ratio (all val pairs): {s_ratio_val:.4f}  (max ≈ 1.0 for perfect rotation)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _resolve_run_dir(arg_run):
    runs_root = SYNTH_RUNS_DIR
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"No synth_runs directory at {runs_root!r}.")
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
    parser.add_argument("--data", default=None,
                        help="Override synthetic data .npy path from checkpoint config.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run)
    print(f"Using run: {os.path.basename(run_dir)}")

    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # data_path = args.data or getattr(cfg, "synth_data_path", "rotations.npy")
    data_path = "rotations_4planes.npy"
    print(f"Loading synthetic data from {data_path} …")
    windows = load_synthetic_windows(cfg, data_path=data_path)
    if getattr(cfg, "synth_noise_std", 0.0) > 0:
        print(f"  Added deterministic Gaussian noise: std={cfg.synth_noise_std}")
    N_in = windows.shape[1]
    print(f"  Windows shape: {windows.shape}  (K, N, T)")

    train_ds, val_ds = train_val_split_synth(windows, cfg.val_split, cfg.seed)

    model = MLP(in_channels=N_in, d=cfg.d, hidden_dim=cfg.hidden_dim, depth=cfg.depth, dropout=cfg.dropout,
                temporal_filters=cfg.temporal_filters, temporal_kernel_size=cfg.temporal_kernel_size)
    model.load_state_dict(ckpt["model_state_dict"])

    make_diagnostic_plots_synth(
        model=model,
        val_ds=val_ds,
        cfg=cfg,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
