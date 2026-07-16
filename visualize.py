"""
Visualise non-reversibility quality from a trained checkpoint.

Each diagnostic plot is saved as its own PNG in the run's `outputs/` dir:

  01_raw_time_coded.png                 — raw, condition-avg, time-coded
  02_embed_planes_time_coded.png        — per rotation plane, condition-avg, time-coded
  03_raw_condition_hsv.png              — raw, condition-avg, HSV by reach angle
  04_embed_planes_condition_hsv.png     — hand traj + embedding planes, same HSV colours
  05_embed_planes_condition_time.png    — per plane, both dims vs time, condition HSV colours
  07_covariance_heatmap.png             — embedding correlation matrix
  08_between_within_variance.png        — trial-discriminability over time
  09_embedding_norm_distribution.png    — ‖F_k‖_F histogram

Usage
-----
    python visualize.py                   # most recent run
    python visualize.py --run 2           # 2nd most recent run
    python visualize.py --run runs/foo    # explicit path
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

from config import Config
from paths import RUNS_DIR
from data import load_mcmaze_cached, gaussian_smooth, make_windows, train_val_split
from model import MLP, infer_multiscale_symmetric_conv_layers
from loss import S_ratio as compute_S_ratio, _batch_rms_normalize


# ── geometry helpers ──────────────────────────────────────────────────────────

def signed_area(x, y):
    """Shoelace signed area of a 2D trajectory. Positive = counterclockwise."""
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _fit_emb_pca(F_hat):
    """Fit top-2 PCA on all (K*T, d) embedding snapshots.

    Returns (mean, Vh2) so reversed embeddings can use the same axes.
    """
    K, d, T = F_hat.shape
    Z = F_hat.transpose(0, 2, 1).reshape(K * T, d)
    mean = Z.mean(axis=0)
    _, _, Vh = np.linalg.svd(Z - mean, full_matrices=False)
    return mean, Vh[:2]


def _apply_emb_pca(F_sub, mean, Vh2):
    """Project (K', d, T) embeddings onto precomputed top-2 PCA axes → (K', 2, T)."""
    K, d, T = F_sub.shape
    Z = F_sub.transpose(0, 2, 1).reshape(K * T, d)
    proj = (Z - mean) @ Vh2.T
    return proj.reshape(K, T, 2).transpose(0, 2, 1)


def raw_phasors(windows, ch_a, ch_b):
    """Z-score two channels per trial → (K, 2, T)."""
    x = windows[:, ch_a, :].astype(np.float32)
    y = windows[:, ch_b, :].astype(np.float32)
    x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)
    y = (y - y.mean(-1, keepdims=True)) / (y.std(-1, keepdims=True) + 1e-6)
    return np.stack([x, y], axis=1)


def all_signed_areas(phasors):
    """(K, 2, T) → (K,) signed area per trial."""
    return np.array([signed_area(p[0], p[1]) for p in phasors])


# ── condition grouping ───────────────────────────────────────────────────────

def _reach_angle_for_row(row):
    """Return arctan2(y, x) of the active target, or NaN if unavailable."""
    try:
        tgt_pos = np.asarray(row["target_pos"])
        act_idx = int(row["active_target"])
        x, y = tgt_pos[act_idx]
        return float(np.arctan2(y, x))
    except Exception:
        return float("nan")


def _get_condition_groups(trial_info_val, n_bins: int = 8):
    """Group val trials into conditions and assign each a colour.

    Priority:
      1. `(trial_type, trial_version)` — NLB MC_Maze canonical condition key.
         Each (type, version) pair defines one maze layout. Coloured by mean
         reach angle within the group via HSV.
      2. `reach_angle` column — bin into `n_bins` directional bins (Indy/Sabes
         continuous-reach data, or any dataset without trial_type).
      3. `trial_type` alone — fallback if version is absent.
      4. Per-trial fallback.

    Returns
    -------
    groups : dict[hashable, list[int]]
        condition key → list of row indices into trial_info_val.
    colors : dict[hashable, RGBA tuple]
    """
    n = len(trial_info_val)
    cols = trial_info_val.columns

    # Priority 1: NLB MC_Maze (trial_type, trial_version)
    if "trial_type" in cols and "trial_version" in cols:
        groups: dict = {}
        for i in range(n):
            row = trial_info_val.iloc[i]
            key = (int(row["trial_type"]), int(row["trial_version"]))
            groups.setdefault(key, []).append(i)

        cond_angles = {}
        for key, idx_list in groups.items():
            angles = [_reach_angle_for_row(trial_info_val.iloc[i]) for i in idx_list]
            angles = [a for a in angles if not np.isnan(a)]
            if angles:
                cond_angles[key] = float(np.arctan2(
                    np.mean(np.sin(angles)), np.mean(np.cos(angles))))
            else:
                cond_angles[key] = float("inf")

        sorted_keys = sorted(groups.keys(), key=lambda k: cond_angles[k])
        groups = {k: groups[k] for k in sorted_keys}
        colors = {}
        for k in sorted_keys:
            ang = cond_angles[k]
            if ang != float("inf"):
                colors[k] = plt.cm.hsv(ang / (2 * np.pi) + 0.5)
            else:
                colors[k] = plt.cm.hsv(hash(k) % 1000 / 1000.0)
        return groups, colors

    # Priority 2: continuous reach_angle
    if "reach_angle" in cols:
        angles = trial_info_val["reach_angle"].to_numpy(dtype=float)
        edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        bin_idx = np.clip(np.digitize(angles, edges) - 1, 0, n_bins - 1)

        groups = {}
        for i, b in enumerate(bin_idx):
            groups.setdefault(int(b), []).append(i)

        sorted_bins = sorted(groups.keys())
        groups = {b: groups[b] for b in sorted_bins}
        centers = 0.5 * (edges[:-1] + edges[1:])
        colors = {b: plt.cm.hsv(centers[b] / (2 * np.pi) + 0.5)
                  for b in sorted_bins}
        return groups, colors

    # Priority 3: trial_type alone
    if "trial_type" in cols:
        groups = {}
        for i in range(n):
            key = int(trial_info_val.iloc[i]["trial_type"])
            groups.setdefault(key, []).append(i)
        unique = sorted(groups.keys())
        colors = {k: plt.cm.hsv(unique.index(k) / max(len(unique), 1))
                  for k in unique}
        return groups, colors

    # Priority 4: per-trial
    groups = {i: [i] for i in range(n)}
    colors = {i: plt.cm.hsv(i / max(n, 1)) for i in range(n)}
    return groups, colors


# ── plot 1 / 2: condition-averaged, time-coded ───────────────────────────────

def _plot_time_coded(phasors, groups, title, xlabel, ylabel, out_path,
                     cmap_name="coolwarm"):
    """One mean trajectory per condition, segments coloured by time bin.

    Math
    ----
    For condition c:  μ^c[t] = (1/|c|) Σ_{k ∈ c} phasors[k, :, t]
    Each line segment (μ^c[t], μ^c[t+1]) is coloured by t / (T-1).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap(cmap_name)
    T = phasors.shape[2]

    for cond_key in groups:
        idx_list = groups[cond_key]
        mean_traj = phasors[idx_list].mean(axis=0)
        x, y = mean_traj[0], mean_traj[1]
        for t in range(T - 1):
            ax.plot(x[t:t+2], y[t:t+2], color=cmap(t / (T - 1)),
                    lw=1.1, alpha=0.85)

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


# ── plot 2 / 4: per-plane condition-averaged ────────────────────────────────

def _plot_planes_time_coded(F_hat, groups, s_ratio, out_path, cmap_name="coolwarm"):
    """Subplot grid: one panel per 2D rotation plane, condition-avg, time-coded."""
    K, d, T = F_hat.shape
    D = d // 2
    planes = F_hat.reshape(K, D, 2, T)
    cmap = plt.get_cmap(cmap_name)

    ncols = min(D, 4)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)

    for p in range(D):
        ax = axes[p // ncols, p % ncols]
        for cond_key in groups:
            idx_list = groups[cond_key]
            mean_traj = planes[idx_list, p].mean(axis=0)  # (2, T)
            x, y = mean_traj[0], mean_traj[1]
            for t in range(T - 1):
                ax.plot(x[t:t+2], y[t:t+2], color=cmap(t / (T - 1)),
                        lw=1.1, alpha=0.85)

        ax.axhline(0, color="k", lw=0.4, alpha=0.25)
        ax.axvline(0, color="k", lw=0.4, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"Plane {p}  (dims {2*p}, {2*p+1})", fontsize=9)
        ax.set_xlabel(f"dim {2*p}", fontsize=8)
        ax.set_ylabel(f"dim {2*p+1}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="datalim")

    for p in range(D, nrows * ncols):
        axes[p // ncols, p % ncols].set_visible(False)

    fig.suptitle(f"Embedding — condition-avg, time-coded  (ζ = {s_ratio:.2f})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _plot_planes_condition_hsv(F_hat, groups, colors, s_ratio, out_path,
                               hand_windows_val=None):
    """Subplot grid: hand trajectories (if available) + one panel per 2D rotation plane.

    When hand_windows_val is provided the first panel shows val-set hand
    trajectories using the *same* condition groups and HSV colours as the
    embedding planes, giving a direct side-by-side comparison.
    """
    K, d, T = F_hat.shape
    D = d // 2
    planes = F_hat.reshape(K, D, 2, T)

    has_hand = False # hand_windows_val is not None
    n_panels = (1 if has_hand else 0) + D
    ncols = min(n_panels, 4)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)

    panel = 0

    if has_hand:
        ax = axes[0, 0]
        for cond_key in groups:
            idx_list = groups[cond_key]
            mean_hand = hand_windows_val[idx_list].mean(axis=0)
            color = colors[cond_key]
            ax.plot(mean_hand[0], mean_hand[1], lw=1.4, color=color, alpha=0.9)
            # ax.xlim(-6,6)
            # ax.ylim(-6,6)
            ax.scatter(mean_hand[0, 0], mean_hand[1, 0], color=color, s=25, zorder=5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("Hand trajectory (val)", fontsize=9)
        ax.set_xlabel("hand_x", fontsize=8)
        ax.set_ylabel("hand_y", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="datalim")
        panel = 1

    for p in range(D):
        idx = panel + p
        ax = axes[idx // ncols, idx % ncols]
        for cond_key in groups:
            idx_list = groups[cond_key]
            mean_traj = planes[idx_list, p].mean(axis=0)  # (2, T)
            color = colors[cond_key]
            ax.plot(mean_traj[0], mean_traj[1], lw=1.4, color=color, alpha=0.9)
            ax.scatter(mean_traj[0, 0], mean_traj[1, 0], color=color, s=25, zorder=5)
        # ax.set_xlim(-6,6)
        # ax.set_ylim(-6,6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"Plane {p}  (dims {2*p}, {2*p+1})", fontsize=12)
        ax.set_xlabel(f"dim {2*p}", fontsize=12)
        ax.set_ylabel(f"dim {2*p+1}", fontsize=12)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="datalim")

    for i in range(n_panels, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    n_conds = len(groups)
    n_per = float(np.mean([len(v) for v in groups.values()]))
    # fig.suptitle(f"Embeddings coded by trial (ζ = {s_ratio:.2f},  "
                #  f"{n_conds} conditions, {n_per:.1f} trials/cond)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── plot 5: per-plane condition-averaged, dims vs time ───────────────────────

def _plot_planes_condition_time(F_hat, groups, colors, s_ratio, out_path):
    """Subplot grid: one panel per 2D rotation plane, condition-avg, dims vs time.

    Like plot 4 but instead of plotting the two plane dims against each other,
    both are plotted against time. Each condition keeps its HSV colour; the even
    coordinate (dim 2p, the plot-4 x-axis) is a solid line and the odd
    coordinate (dim 2p+1, the plot-4 y-axis) a dashed line.
    """
    K, d, T = F_hat.shape
    D = d // 2
    planes = F_hat.reshape(K, D, 2, T)
    t_axis = np.arange(T)

    ncols = min(D, 4)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)

    for p in range(D):
        ax = axes[p // ncols, p % ncols]
        for cond_key in groups:
            idx_list = groups[cond_key]
            mean_traj = planes[idx_list, p].mean(axis=0)  # (2, T)
            color = colors[cond_key]
            ax.plot(t_axis, mean_traj[0], lw=1.4, color=color, alpha=0.9,
                    ls="-")
            ax.plot(t_axis, mean_traj[1], lw=1.4, color=color, alpha=0.9,
                    ls="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"Plane {p}  (dims {2*p}, {2*p+1})", fontsize=12)
        ax.set_xlabel("time (bins)", fontsize=12)
        ax.set_ylabel("embedding value", fontsize=12)
        ax.tick_params(labelsize=7)

    # Style legend (shared): solid = even/x dim, dashed = odd/y dim.
    style_handles = [
        Line2D([0], [0], color="0.3", lw=1.4, ls="-", label="even dim (x)"),
        Line2D([0], [0], color="0.3", lw=1.4, ls="--", label="odd dim (y)"),
    ]
    axes[0, 0].legend(handles=style_handles, fontsize=8, loc="best",
                      frameon=False)

    for i in range(D, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── plot 3 / 4: condition-averaged, HSV by reach angle ───────────────────────

def _plot_condition_hsv(phasors, groups, colors, title, xlabel, ylabel, out_path):
    """One mean trajectory per condition, single HSV colour per condition."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for cond_key in groups:
        idx_list = groups[cond_key]
        mean_traj = phasors[idx_list].mean(axis=0)
        color = colors[cond_key]
        ax.plot(mean_traj[0], mean_traj[1], lw=1.4, color=color, alpha=0.9)
        ax.scatter(mean_traj[0, 0], mean_traj[1, 0], color=color, s=25, zorder=5)

    n_conds = len(groups)
    n_per = float(np.mean([len(v) for v in groups.values()]))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{title}\n{n_conds} conditions  ({n_per:.1f} trials/cond avg)",
                 fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── plot 5: signed-area histogram ────────────────────────────────────────────

def plot_signed_area_histogram(areas_raw, areas_emb, out_path):
    """Distribution of shoelace signed area, raw vs embedding.

    Math
    ----
    A_k = ½ Σ_{t=0}^{T-2} (x_k[t]·y_k[t+1] − x_k[t+1]·y_k[t])
    Histogram across trials k. |μ_A|/σ_A > 1 ⇒ consistent net circulation.
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
    ax.set_title("Signed-area distribution\n(non-zero mean → net circulation)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── plot 6: PCA explained variance ───────────────────────────────────────────

def plot_pca_explained_variance(F_hat, out_path):
    """Cumulative PCA explained variance of the (K*T, d) embedding snapshots.

    Math
    ----
    Z ∈ R^(M×d) with M = K*T, rows = embedding snapshots.
    Centred Z̄ = Z − mean.   Eigenvalues λ₁ ≥ … ≥ λ_d of (1/M) Z̄ᵀZ̄.
    cumvar(j) = (Σ_{i≤j} λ_i) / (Σ_{i≤d} λ_i).
    Step at PC1 ⇒ dimensional collapse.
    """
    from sklearn.decomposition import PCA

    K, d, T = F_hat.shape
    Z = F_hat.transpose(0, 2, 1).reshape(K * T, d)

    fig, ax = plt.subplots(figsize=(6, 4))
    pca = PCA(n_components=min(d, K * T))
    pca.fit(Z)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

    ax.plot(np.arange(1, len(cumvar) + 1), cumvar, lw=1.5, color="steelblue")
    ax.axhline(95, color="tomato", lw=0.8, ls="--", label="95%")
    ax.axhline(99, color="darkorange", lw=0.8, ls="--", label="99%")
    n95 = int(np.searchsorted(cumvar, 95)) + 1
    n99 = int(np.searchsorted(cumvar, 99)) + 1
    ax.set_xlabel("Number of PCs", fontsize=9)
    ax.set_ylabel("Cumulative explained variance (%)", fontsize=9)
    ax.set_title(
        f"PCA explained variance  (d={d})\n"
        f"95 % in {n95} PCs,  99 % in {n99} PCs",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

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

def _plot_dim_grid(F_hat, s_ratio, out_path,
                    cmap_name="coolwarm", seed=0):
    """D×D grid where cell (i,j) plots dim 2*i vs dim 2*j+1, time-coded.

    Diagonal (i==j) = native rotation planes.
    Off-diagonal = cross-plane pairings; should look like Lissajous/scatter if
    the planes are independent, or structured circles if they share a source.
    """
    K, d, T = F_hat.shape
    n_show = int(K * 0.1)
    D = d // 2
    rng = np.random.default_rng(seed)
    idx = rng.choice(K, size=min(n_show, K), replace=False)
    cmap = plt.get_cmap(cmap_name)
    zeta = _pairwise_zeta(F_hat)   # (D, D)

    fig, axes = plt.subplots(D, D, figsize=(3.2 * D, 3.2 * D), squeeze=False)

    for i in range(D):
        # print(f"i={i}")
        for j in range(D):
            # print(f"j={j}")
            ax = axes[i][j]
            x_dim, y_dim = 2 * i, 2 * j + 1
            for k in idx:
                # print(f"k={k}")
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

            ax.set_xlabel(f"dim {x_dim}", fontsize=7)
            ax.set_ylabel(f"dim {y_dim}", fontsize=7)
            ax.set_title(f"({x_dim},{y_dim})  ζ={zeta[i,j]:.2f}", fontsize=7, pad=2)

    fig.suptitle(f"Dim grid — time-coded  (ζ = {s_ratio:.2f})\n",
                #  f"cell (i,j): dim 2i vs dim 2j+1   [diagonal = native planes]",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── plot 8: covariance heatmap ───────────────────────────────────────────────

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
    im = ax.imshow(Corr[:n_show, :n_show], cmap="RdBu_r", vmin=-1, vmax=1,
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


# ── plot 9: between vs within trial variance ─────────────────────────────────

def plot_between_within_variance(F_hat, out_path):
    """Trial-discriminability ratio over time.

    Math
    ----
    F̄[i,t] = (1/K) Σ_k F_k[i,t]
    B(t)   = (1/Kd) Σ_{k,i} (F_k[i,t] − F̄[i,t])²        across-trial spread
    W      = (1/Kd) Σ_{k,i} Var_t(F_k[i,:])              avg per-trial temporal var
    r(t)   = B(t) / (B(t) + W)   ∈ [0, 1]
    r → 1: trials distinct (signal).  r → 0: all trials identical (collapse).
    """
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


# ── plot 10: embedding norm distribution ─────────────────────────────────────

def plot_embedding_norm_distribution(F_hat, out_path):
    """Distribution of per-trial Frobenius norms.

    Math
    ----
    ‖F_k‖_F = √(Σ_{i,t} F_k[i,t]²),  CV = std/mean across k.
    CV → 0 ⇒ trivial constant-magnitude solution (all trials same scale).
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
        f"Embedding norm distribution\n"
        f"CV = {cv:.3f}  (0 = identical magnitudes)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── plot 0: condition-split diagnostic ───────────────────────────────────────

def _hand_windows_from_raw(hand_pos_raw, cfg, trial_info, time_index_s, bin_width_s):
    """Window hand_pos using the same alignment as the spike windows.

    Returns (K_all, 2, T) or None if hand_pos_raw is None.
    """
    if hand_pos_raw is None:
        return None
    return make_windows(
        hand_pos_raw, trial_info, time_index_s, bin_width_s,
        strategy=cfg.window_strategy, window_size=cfg.window_size,
        align_field=getattr(cfg, "align_field", "move_onset_time"),
        pre_ms=getattr(cfg, "pre_ms", 100),
    )


def plot_conditions_diagnostic(hand_windows, trial_info, val_indices, out_path):
    """Sanity-check condition splitting using physical hand trajectories.

    Left panel: all trials. Right panel: val-only trials.
    """
    groups_all, colors_all = _get_condition_groups(trial_info)

    trial_info_val = trial_info.iloc[val_indices].reset_index(drop=True)
    hand_windows_val = hand_windows[val_indices]
    groups_val, colors_val = _get_condition_groups(trial_info_val)

    fig, (ax_all, ax_val) = plt.subplots(1, 2, figsize=(14, 6))

    for cond_key in groups_all:
        idx_list = groups_all[cond_key]
        mean_hand = hand_windows[idx_list].mean(axis=0)
        color = colors_all[cond_key]
        ax_all.plot(mean_hand[0], mean_hand[1], lw=0.9, color=color, alpha=0.85)
        ax_all.scatter(mean_hand[0, 0], mean_hand[1, 0], color=color, s=18, zorder=5)

    cond_sizes_all = [len(v) for v in groups_all.values()]
    ax_all.set_title(
        f"All trials ({hand_windows.shape[0]})\n"
        f"{len(groups_all)} conditions  |  trials/cond: "
        f"min={min(cond_sizes_all)}, median={int(np.median(cond_sizes_all))}, "
        f"max={max(cond_sizes_all)}",
        fontsize=10,
    )
    ax_all.set_xlabel("hand_x", fontsize=9)
    ax_all.set_ylabel("hand_y", fontsize=9)
    ax_all.set_aspect("equal", adjustable="datalim")
    ax_all.tick_params(labelsize=8)
    ax_all.spines[["top", "right"]].set_visible(False)

    for cond_key in groups_val:
        idx_list = groups_val[cond_key]
        mean_hand = hand_windows_val[idx_list].mean(axis=0)
        color = colors_val[cond_key]
        ax_val.plot(mean_hand[0], mean_hand[1], lw=0.9, color=color, alpha=0.85)
        ax_val.scatter(mean_hand[0, 0], mean_hand[1, 0], color=color, s=18, zorder=5)

    cond_sizes_val = [len(v) for v in groups_val.values()]
    ax_val.set_title(
        f"Val trials ({hand_windows_val.shape[0]})\n"
        f"{len(groups_val)} conditions  |  trials/cond: "
        f"min={min(cond_sizes_val)}, median={int(np.median(cond_sizes_val))}, "
        f"max={max(cond_sizes_val)}",
        fontsize=10,
    )
    ax_val.set_xlabel("hand_x", fontsize=9)
    ax_val.set_ylabel("hand_y", fontsize=9)
    ax_val.set_aspect("equal", adjustable="datalim")
    ax_val.tick_params(labelsize=8)
    ax_val.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Condition-split diagnostic: trial-avg hand trajectories", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── loss curve ───────────────────────────────────────────────────────────────

def plot_loss_curve(run_dir: str, cfg: Config) -> None:
    """Regenerate the training loss curve from saved CSVs.

    Reads  outputs/log.csv         — epoch-level S and loss
           outputs/reg_history.csv — per-reg raw and λ·reg magnitudes (written
                                     by train.py when at least one regularizer
                                     is active; absent for unregularised runs)

    Produces the same single-panel training-dynamics figure as train.py.
    """
    import csv as _csv

    out_dir = os.path.join(run_dir, "outputs")
    log_path = os.path.join(out_dir, "log.csv")
    if not os.path.isfile(log_path):
        print(f"  [loss curve] no log.csv found at {log_path!r} — skipping.")
        return

    # read log.csv
    epochs, val_s, val_c_plus, val_zeta, val_losses = [], [], [], [], []
    with open(log_path, newline="") as f:
        for row in _csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            val_s.append(float(row["val_s"]))
            if row.get("val_c_plus") not in (None, ""):
                val_c_plus.append(float(row["val_c_plus"]))
            if row.get("val_zeta") not in (None, ""):
                val_zeta.append(float(row["val_zeta"]))
            val_losses.append(float(row["val_loss"]))
    has_c_plus = len(val_c_plus) == len(epochs)
    has_zeta = len(val_zeta) == len(epochs)

    reg_lambdas = {
        "xp": cfg.lambda_xp,
        "bt": cfg.lambda_bt,
        "plane_bt": getattr(cfg, "lambda_plane_bt", 0.0),
        "cca": getattr(cfg, "lambda_block_cca", 0.0),
    }
    active_regs = [k for k, v in reg_lambdas.items() if v > 0]

    # try to load per-epoch reg data (written by train.py; absent for old runs)
    reg_scaled: dict[str, list[float]] = {}
    reg_path = os.path.join(out_dir, "reg_history.csv")
    if os.path.isfile(reg_path):
        with open(reg_path, newline="") as f:
            reader = _csv.DictReader(f)
            csv_regs = [c[4:] for c in (reader.fieldnames or [])
                        if c.startswith("raw_")]
            for k in csv_regs:
                reg_scaled[k] = []
            for row in reader:
                for k in csv_regs:
                    reg_scaled[k].append(float(row[f"scaled_{k}"]))

    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.plot(epochs, val_s, label="S mean/plane (↑)", color="steelblue")
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
        ax.plot(epochs, total_scaled, label="total λ·reg (↓)", color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Embedding validation loss components")
    ax.spines[["top", "right"]].set_visible(False)

    ax_zeta = None
    if has_zeta:
        ax_zeta = ax.twinx()
        ax_zeta.plot(epochs, val_zeta, label="ζ", color="seagreen", alpha=0.6)
        ax_zeta.set_ylabel("Validation ζ")
        ax_zeta.spines["top"].set_visible(False)

    best_epoch = None
    best_label = "best val loss"
    best_ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if os.path.isfile(best_ckpt_path):
        try:
            ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            best_epoch = int(ckpt["epoch"])
            if ckpt.get("checkpoint_selection") == "best_val_zeta":
                best_label = "best val ζ"
        except Exception as exc:
            print(f"  [loss curve] could not read best.pt ({exc}); falling back to log.csv.")
    if has_zeta and best_label != "best val ζ":
        best_epoch = max(range(1, len(val_zeta) + 1), key=lambda i: val_zeta[i - 1])
        best_label = "best val ζ"
    if best_epoch is None and val_losses:
        best_epoch = min(range(1, len(val_losses) + 1), key=lambda i: val_losses[i - 1])

    if best_epoch is not None:
        idx = best_epoch - 1
        best_on_zeta = best_label == "best val ζ" and ax_zeta is not None and 0 <= idx < len(val_zeta)
        if 0 <= idx < len(val_s):
            target_ax = ax_zeta if best_on_zeta else ax
            series = val_zeta if best_on_zeta else val_s
            y = series[idx]
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
    print(f"Saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def _resolve_run_dir(arg_run):
    runs_root = RUNS_DIR
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"No runs directory at {runs_root!r}. Run `python main.py` first.")
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


def make_diagnostic_plots(
    model,
    val_ds,
    trial_info,
    cfg: Config,
    run_dir: str,
    hand_windows=None,
    cond_start: int | None = None,
    cond_stop: int | None = None,
    cond_skip: int | None = None,
):
    """Compute embeddings on val_ds and write all diagnostic PNGs to run_dir/outputs/.

    Reusable from both visualize.py (post-hoc) and main.py (right after training).
    Does not load any data; all heavy inputs come from caller.

    Parameters
    ----------
    model         : trained MLP (any device; will be moved to CPU and put in eval())
    val_ds        : torch Subset of the windows TensorDataset
    trial_info    : full trial_info DataFrame (val_ds.indices selects from it)
    cfg           : Config used to train the model
    run_dir       : run directory; plots are written to {run_dir}/outputs/
    hand_windows  : optional (K_all, 2, T) array; if given, plot 04 includes hand panel
    cond_start    : first condition index to plot (angle-sorted order)
    cond_stop     : one-past-last condition index to plot
    cond_skip     : step size for condition selection
    """

    out_dir = os.path.join(run_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    val_indices = list(val_ds.indices)
    trial_info_val = trial_info.iloc[val_indices].reset_index(drop=True)
    cond_groups, cond_colors = _get_condition_groups(trial_info_val)

    if cond_start is not None or cond_stop is not None or cond_skip is not None:
        all_keys = list(cond_groups.keys())
        keep = all_keys[cond_start:cond_stop:cond_skip]
        print(f"Condition slice [{cond_start}:{cond_stop}:{cond_skip}]: "
              f"plotting {len(keep)} of {len(all_keys)} conditions")
        cond_groups = {k: cond_groups[k] for k in keep}
        cond_colors = {k: cond_colors[k] for k in keep}

    hand_windows_val = hand_windows[val_indices] if hand_windows is not None else None

    # CPU is plenty for inference on a val set of a few hundred trials.
    model = model.cpu().eval()

    loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    (val_tensor,) = next(iter(loader))
    val_np = val_tensor.numpy()
    K, N = val_np.shape[0], val_np.shape[1]
    n_per = float(np.mean([len(v) for v in cond_groups.values()]))
    print(f"Val set: {K} trials  |  N={N}  |  T={cfg.window_size}  |  "
          f"{len(cond_groups)} conditions ({n_per:.1f} trials/cond)")

    print("Computing embeddings…")
    with torch.no_grad():
        F_hat_t = model(val_tensor)
        F_hat_t = F_hat_t - F_hat_t.mean(dim=cfg.F_mean_axis, keepdim=True)  # zero-mean per dim across batch and time
        s_ratio_val = compute_S_ratio(_batch_rms_normalize(F_hat_t)).item()
        F_hat = F_hat_t.numpy()

    ch_var = val_np.var(axis=(0, 2))
    top2_ch = np.argsort(ch_var)[-2:][::-1]
    ch_a, ch_b = int(top2_ch[0]), int(top2_ch[1])
    print(f"Raw phasor channels: {ch_a} (var={ch_var[ch_a]:.4f}),  "
          f"{ch_b} (var={ch_var[ch_b]:.4f})")
    phasors_raw = raw_phasors(val_np, ch_a, ch_b)

    areas_raw = all_signed_areas(phasors_raw)

    _plot_time_coded(
        phasors_raw, cond_groups,
        title=f"Raw — condition-avg, time-coded  (ch {ch_a} vs ch {ch_b})",
        xlabel=f"Ch {ch_a} (z-scored)", ylabel=f"Ch {ch_b} (z-scored)",
        out_path=os.path.join(out_dir, "01_raw_time_coded.png"),
    )
    _plot_planes_time_coded(
        F_hat, cond_groups, s_ratio_val,
        out_path=os.path.join(out_dir, "02_embed_planes_time_coded.png"),
    )
    _plot_condition_hsv(
        phasors_raw, cond_groups, cond_colors,
        title=f"Raw — condition-averaged (ch {ch_a} vs ch {ch_b})",
        xlabel=f"Ch {ch_a} (z-scored)", ylabel=f"Ch {ch_b} (z-scored)",
        out_path=os.path.join(out_dir, "03_raw_condition_hsv.png"),
    )
    _plot_planes_condition_hsv(
        F_hat, cond_groups, cond_colors, s_ratio_val,
        out_path=os.path.join(out_dir, "04_embed_planes_condition_hsv.png"),
        hand_windows_val=hand_windows_val,
    )
    _plot_planes_condition_time(
        F_hat, cond_groups, cond_colors, s_ratio_val,
        out_path=os.path.join(out_dir, "05_embed_planes_condition_time.png"),
    )
    plot_covariance_heatmap(
        F_hat, out_path=os.path.join(out_dir, "07_covariance_heatmap.png"),
    )
    plot_between_within_variance(
        F_hat, out_path=os.path.join(out_dir, "08_between_within_variance.png"),
    )
    plot_embedding_norm_distribution(
        F_hat, out_path=os.path.join(out_dir, "09_embedding_norm_distribution.png"),
    )
    # _plot_dim_grid(
    #     F_hat, s_ratio_val,
    #     out_path=os.path.join(out_dir, "10_dim_grid_time_coded.png"),
    # )
    # plot_conditions_diagnostic()

    print(f"\nS_ratio (embedding, all val pairs): {s_ratio_val:.4f}")
    print(f"Signed area (raw):  μ={areas_raw.mean():+.4f}  σ={areas_raw.std():.4f}"
          f"  |μ|/σ = {abs(areas_raw.mean()) / (areas_raw.std() + 1e-12):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None,
                        help="Integer (1=most recent) or explicit path. Omit for most recent.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=None,
                        help="First condition index to plot (sorted order). Default: 0.")
    parser.add_argument("--stop", type=int, default=None,
                        help="One-past-last condition index to plot (sorted order). Default: all.")
    parser.add_argument("--skip", type=int, default=None,
                        help="Step size for condition selection (e.g. --start 0 --stop 100 --skip 10).")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run)
    print(f"Using run: {os.path.basename(run_dir)}")

    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at '{ckpt_path}'.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    print("Loading data…")
    spikes_raw, bin_width_s, trial_info, time_index_s, hand_pos_raw = load_mcmaze_cached(
        cfg.nwb_path, cfg.bin_ms
    )
    N = spikes_raw.shape[0]
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
    # grand_mean = windows.mean(axis=(0,2), keepdims=True)  # (1, N, T)
    # windows = windows - grand_mean
    if cfg.split == "random":
        trial_info = trial_info.drop(columns=["split"], errors="ignore")
    train_ds, val_ds = train_val_split(windows, trial_info, cfg.val_split, cfg.seed)
    hand_windows = _hand_windows_from_raw(hand_pos_raw, cfg, trial_info, time_index_s, bin_width_s)

    state_dict = ckpt["model_state_dict"]
    temporal_frontend = getattr(cfg, "temporal_frontend", "symmetric")
    if "temporal_conv.weight" in state_dict:
        temporal_frontend = "symmetric"
    elif any(k.startswith("temporal_conv.temporal_branches.0.norm.") for k in state_dict):
        temporal_frontend = "residual"
    elif any(k.startswith("temporal_conv.temporal_branches.0.conv.") for k in state_dict):
        temporal_frontend = "multiscale_symmetric"

    model = MLP(
        in_channels=N,
        d=cfg.d,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        dropout=cfg.dropout,
        temporal_filters=getattr(cfg, "temporal_filters", 0),
        temporal_kernel_size=getattr(cfg, "temporal_kernel_size", 31),
        temporal_frontend=temporal_frontend,
        residual_kernels=getattr(cfg, "residual_kernels", "3,7,15,31"),
        multiscale_symmetric_conv_layers=infer_multiscale_symmetric_conv_layers(
            state_dict,
            getattr(cfg, "multiscale_symmetric_conv_layers", 1),
        ),
    )
    model.load_state_dict(state_dict)

    make_diagnostic_plots(
        model=model,
        val_ds=train_ds,
        trial_info=trial_info,
        cfg=cfg,
        run_dir=run_dir,
        hand_windows=hand_windows,
        cond_start=args.start,
        cond_stop=args.stop,
        cond_skip=args.skip,
    )
    plot_loss_curve(run_dir, cfg)


if __name__ == "__main__":
    main()
