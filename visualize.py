"""
Visualise non-reversibility quality from a trained checkpoint.

Each diagnostic plot is saved as its own PNG in the run's `outputs/` dir:

  00_conditions_diagnostic.png          — per-condition mean hand trajectory (sanity check)
  01_raw_time_coded.png                 — raw, condition-avg, time-coded
  02_embed_planes_time_coded.png        — per rotation plane, condition-avg, time-coded
  03_raw_condition_hsv.png              — raw, condition-avg, HSV by reach angle
  04_embed_planes_condition_hsv.png     — per rotation plane, condition-avg, HSV
  07_covariance_heatmap.png             — embedding correlation matrix
  08_between_within_variance.png        — trial-discriminability over time
  09_embedding_norm_distribution.png    — ‖F_k‖_F histogram

Usage
-----
    python visualize.py                   # most recent run
    python visualize.py --run 2           # 2nd most recent run
    python visualize.py --run runs/foo    # explicit path
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
from data import load_mcmaze, gaussian_smooth, make_windows, train_val_split
from model import MLP
from loss import S_ratio as compute_S_ratio


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

        colors = {}
        for key, idx_list in groups.items():
            angles = [_reach_angle_for_row(trial_info_val.iloc[i]) for i in idx_list]
            angles = [a for a in angles if not np.isnan(a)]
            if angles:
                # circular mean
                ang = float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
                colors[key] = plt.cm.hsv(ang / (2 * np.pi) + 0.5)
            else:
                colors[key] = plt.cm.hsv(hash(key) % 1000 / 1000.0)
        return groups, colors

    # Priority 2: continuous reach_angle
    if "reach_angle" in cols:
        angles = trial_info_val["reach_angle"].to_numpy(dtype=float)
        edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        bin_idx = np.clip(np.digitize(angles, edges) - 1, 0, n_bins - 1)

        groups = {}
        for i, b in enumerate(bin_idx):
            groups.setdefault(int(b), []).append(i)

        centers = 0.5 * (edges[:-1] + edges[1:])
        colors = {b: plt.cm.hsv(centers[b] / (2 * np.pi) + 0.5)
                  for b in groups.keys()}
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

    for cond_key in sorted(groups.keys(), key=lambda k: str(k)):
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
        for cond_key in sorted(groups.keys(), key=lambda k: str(k)):
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


def _plot_planes_condition_hsv(F_hat, groups, colors, s_ratio, out_path):
    """Subplot grid: one panel per 2D rotation plane, condition-avg, HSV-colored."""
    K, d, T = F_hat.shape
    D = d // 2
    planes = F_hat.reshape(K, D, 2, T)

    ncols = min(D, 4)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)

    for p in range(D):
        ax = axes[p // ncols, p % ncols]
        for cond_key in sorted(groups.keys(), key=lambda k: str(k)):
            idx_list = groups[cond_key]
            mean_traj = planes[idx_list, p].mean(axis=0)  # (2, T)
            color = colors[cond_key]
            ax.plot(mean_traj[0], mean_traj[1], lw=1.4, color=color, alpha=0.9)
            ax.scatter(mean_traj[0, 0], mean_traj[1, 0], color=color, s=25, zorder=5)

        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"Plane {p}  (dims {2*p}, {2*p+1})", fontsize=9)
        ax.set_xlabel(f"dim {2*p}", fontsize=8)
        ax.set_ylabel(f"dim {2*p+1}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="datalim")

    for p in range(D, nrows * ncols):
        axes[p // ncols, p % ncols].set_visible(False)

    n_conds = len(groups)
    n_per = float(np.mean([len(v) for v in groups.values()]))
    fig.suptitle(f"Embedding — condition-avg, HSV  (ζ = {s_ratio:.2f},  "
                 f"{n_conds} conditions, {n_per:.1f} trials/cond)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── plot 3 / 4: condition-averaged, HSV by reach angle ───────────────────────

def _plot_condition_hsv(phasors, groups, colors, title, xlabel, ylabel, out_path):
    """One mean trajectory per condition, single HSV colour per condition."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for cond_key in sorted(groups.keys(), key=lambda k: str(k)):
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
    # Z = Z - Z.mean(axis=0)
    # Z = Z / (Z.std(axis=0) + 1e-6)
    Z = Z-Z.mean(axis=0)
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


def plot_conditions_diagnostic(hand_windows, trial_info, out_path):
    """Sanity-check condition splitting using physical hand trajectories.

    For each condition (using the same priority as `_get_condition_groups`,
    over *all* trials — not just val), plot the trial-averaged hand path
    coloured by the condition's HSV reach angle. Distinct curved reach paths
    in different directions ⇒ conditions are splitting on a meaningful axis.
    """
    groups, colors = _get_condition_groups(trial_info)

    fig, ax = plt.subplots(figsize=(7, 7))
    for cond_key in sorted(groups.keys(), key=lambda k: str(k)):
        idx_list = groups[cond_key]
        mean_hand = hand_windows[idx_list].mean(axis=0)   # (2, T)
        color = colors[cond_key]
        ax.plot(mean_hand[0], mean_hand[1], lw=0.9, color=color, alpha=0.85)
        ax.scatter(mean_hand[0, 0], mean_hand[1, 0], color=color, s=18, zorder=5)

    cond_sizes = [len(v) for v in groups.values()]
    ax.set_title(
        f"Condition-split diagnostic: trial-avg hand trajectories\n"
        f"{len(groups)} conditions  |  trials/cond: "
        f"min={min(cond_sizes)}, median={int(np.median(cond_sizes))}, "
        f"max={max(cond_sizes)}",
        fontsize=10,
    )
    ax.set_xlabel("hand_x", fontsize=9)
    ax.set_ylabel("hand_y", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def _resolve_run_dir(arg_run):
    runs_root = "runs"
    if not os.path.isdir(runs_root):
        raise FileNotFoundError("No 'runs/' directory. Run `python main.py` first.")
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
    hand_windows  : optional (K_all, 2, T) array; if given, plot 00 is generated
    """

    out_dir = os.path.join(run_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    val_indices = list(val_ds.indices)
    trial_info_val = trial_info.iloc[val_indices].reset_index(drop=True)
    cond_groups, cond_colors = _get_condition_groups(trial_info_val)

    if hand_windows is not None:
        plot_conditions_diagnostic(
            hand_windows, trial_info,
            out_path=os.path.join(out_dir, "00_conditions_diagnostic.png"),
        )
    else:
        print("Skipping condition diagnostic: no hand_pos available.")

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
        F_hat_t = F_hat_t - F_hat_t.mean(dim=0, keepdim=True)  # zero-mean per dim across batch and time
        s_ratio_val = compute_S_ratio(F_hat_t).item()
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

    print(f"\nS_ratio (embedding, all val pairs): {s_ratio_val:.4f}")
    print(f"Signed area (raw):  μ={areas_raw.mean():+.4f}  σ={areas_raw.std():.4f}"
          f"  |μ|/σ = {abs(areas_raw.mean()) / (areas_raw.std() + 1e-12):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None,
                        help="Integer (1=most recent) or explicit path. Omit for most recent.")
    parser.add_argument("--seed", type=int, default=42)
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
    spikes_raw, bin_width_s, trial_info, time_index_s, hand_pos_raw = load_mcmaze(
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
    grand_mean = windows.mean(axis=0, keepdims=True)  # (1, N, T)
    windows = windows - grand_mean
    
    train_ds, val_ds = train_val_split(windows, trial_info, cfg.val_split, cfg.seed)
    hand_windows = _hand_windows_from_raw(hand_pos_raw, cfg, trial_info, time_index_s, bin_width_s)

    model = MLP(in_channels=N, d=cfg.d, hidden_dim=cfg.hidden_dim, depth=cfg.depth, dropout=cfg.dropout)
    model.load_state_dict(ckpt["model_state_dict"])

    make_diagnostic_plots(
        model=model,
        val_ds=val_ds, 
        trial_info=trial_info,
        cfg=cfg,
        run_dir=run_dir,
        hand_windows=hand_windows,
    )


if __name__ == "__main__":
    main()
