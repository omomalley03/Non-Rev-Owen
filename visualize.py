"""
Visualise non-reversibility quality from a trained checkpoint.

Each diagnostic plot is saved as its own PNG in the run's `outputs/` dir:

  01_raw_time_coded.png                 — raw, condition-avg, time-coded
  02_embed_planes_time_coded.png        — per rotation plane, condition-avg, time-coded
  02b_embed_planes_time_coded_selected_trials.png — per plane, selected individual trials, time-coded
  03_raw_condition_hsv.png              — raw, condition-avg, HSV by reach angle
  04_embed_planes_condition_hsv.png     — hand traj + embedding planes, same HSV colours
  04b_embed_planes_condition_hsv_selected_trials.png — selected individual trials, HSV by condition
  05_embed_planes_condition_time.png    — per plane, both dims vs time, condition HSV colours
  06_plane_validation_zeta_ranking.csv  — all planes ranked by validation ζ
  06_plane_validation_zeta_bars.png     — bar chart of ranked plane validation ζ
  07_covariance_heatmap.png             — embedding correlation matrix
  07_zeta_sorted_correlation_heatmap.png — correlation matrix sorted by branch and plane ζ
  07_block_cca_plane_heatmap.png        — ‖C_pq‖_F² cross-plane CCA matrix
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
import csv
import os

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_nonrev")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

from config import Config
from paths import RUNS_DIR
from data import load_mcmaze_cached, gaussian_smooth, make_windows, train_val_split
from model import MLP, infer_multiscale_symmetric_conv_layers
from loss import (
    S_ratio as compute_S_ratio,
    _batch_rms_normalize,
    _pair_terms_per_plane,
    _plane_samples,
    _whiten_2d,
)
from visualize_loss import plot_loss_curve


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


def _parse_index_list(spec: str | None) -> list[int]:
    spec = str(spec or "").strip()
    if not spec:
        return []
    out = []
    for item in spec.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def _select_groups_by_indices(groups: dict, indices: list[int]) -> dict:
    if not indices:
        return groups
    keys = list(groups.keys())
    bad = [idx for idx in indices if idx < 0 or idx >= len(keys)]
    if bad:
        raise ValueError(
            f"Condition indices out of range: {bad}. Valid range is 0-{len(keys) - 1}."
        )
    return {keys[idx]: groups[keys[idx]] for idx in indices}


def _select_available_groups_by_indices(groups: dict, indices: list[int]) -> tuple[dict, list[int]]:
    """Select valid group indices and return any requested indices that were absent."""
    keys = list(groups.keys())
    keep = [idx for idx in indices if 0 <= idx < len(keys)]
    missing = [idx for idx in indices if idx < 0 or idx >= len(keys)]
    return {keys[idx]: groups[keys[idx]] for idx in keep}, missing


def _sample_trials_per_condition(groups: dict, n_trials: int, seed: int) -> dict:
    if n_trials <= 0:
        return groups
    rng = np.random.default_rng(seed)
    sampled = {}
    for cond_key, idx_list in groups.items():
        idx = np.asarray(idx_list, dtype=int)
        if len(idx) > n_trials:
            idx = np.sort(rng.choice(idx, size=n_trials, replace=False))
        sampled[cond_key] = idx.tolist()
    return sampled


def _plane_indices_for_plot(d: int) -> list[int]:
    """Return plane indices to plot, limiting large embeddings for readability."""
    D = d // 2
    if d > 32:
        return list(range(8)) + list(range(D - 8, D))
    return list(range(D))


def _plane_zeta_values(planes: np.ndarray, plane_indices: list[int]) -> dict[int, float]:
    """Compute ζ for each selected native 2D plane."""
    if not plane_indices:
        return {}
    selected = torch.from_numpy(planes[:, plane_indices])
    K, P, _, T = selected.shape
    selected = selected.reshape(K, 2 * P, T)
    minus_per_plane, plus_per_plane = _pair_terms_per_plane(_batch_rms_normalize(selected))
    zeta = minus_per_plane / (plus_per_plane + 1e-8)
    return {p: zeta[i].item() for i, p in enumerate(plane_indices)}


def _plane_redundancy_matrix(planes: np.ndarray, cfg: Config | None = None) -> np.ndarray:
    """Return ‖C_pq‖_F² between whitened native 2D planes."""
    F = torch.from_numpy(planes.reshape(planes.shape[0], planes.shape[1] * 2, planes.shape[3]))
    eps = float(getattr(cfg, "block_cca_eps", 1e-4)) if cfg is not None else 1e-4
    with torch.no_grad():
        X = _whiten_2d(_plane_samples(F), eps=eps)
        C = torch.einsum("pmi,qmj->pqij", X, X) / X.shape[1]
        return C.pow(2).sum(dim=(-1, -2)).cpu().numpy()


def _select_nonredundant_ranked_planes(
    candidate_indices: list[int],
    plane_zeta: dict[int, float],
    redundancy: np.ndarray,
    max_planes: int,
    reg_thresh: float,
) -> list[int]:
    """Rank by ζ and skip planes too redundant with earlier kept planes."""
    ranked = sorted(candidate_indices, key=lambda p: plane_zeta[p], reverse=True)
    selected: list[int] = []
    for p in ranked:
        if any(float(redundancy[p, q]) >= reg_thresh for q in selected):
            continue
        selected.append(p)
        if len(selected) >= max_planes:
            break
    return selected


def _mixed_parity_plane_split(d: int, cfg: Config | None) -> tuple[int, int] | None:
    """Return (even/symmetric planes, odd/anti-symmetric planes) for mixed parity."""
    if cfg is None:
        return None
    frontend = str(getattr(cfg, "temporal_frontend", "") or "").lower()
    mixed_frontends = {
        "mixed_parity",
        "mixed_symmetric_antisymmetric",
        "mixed_sym_anti",
        "sym_anti",
    }
    if frontend not in mixed_frontends:
        return None

    D = d // 2
    odd_planes = int(getattr(cfg, "antisymmetric_planes", 0))
    if odd_planes < 0:
        odd_planes = max(1, D // 2)
    odd_planes = min(max(odd_planes, 0), D)
    even_planes = D - odd_planes
    return even_planes, odd_planes


def _plane_indices_for_ranked_branch_plot(
    planes: np.ndarray,
    d: int,
    cfg: Config | None,
    per_branch: int = 8,
    reg_thresh: float = 1.0,
) -> tuple[list[int], dict[int, float], dict[int, str]]:
    """Select high-ζ nonredundant planes for plots 02, 04, and 05."""
    D = d // 2
    all_plane_indices = list(range(D))
    plane_zeta = _plane_zeta_values(planes, all_plane_indices)
    redundancy = _plane_redundancy_matrix(planes, cfg)
    max_planes = min(D, 2 * per_branch if d > 32 else D)
    plane_indices = _select_nonredundant_ranked_planes(
        all_plane_indices,
        plane_zeta,
        redundancy,
        max_planes=max_planes,
        reg_thresh=reg_thresh,
    )

    split = _mixed_parity_plane_split(d, cfg)
    if split is None:
        return plane_indices, plane_zeta, {p: "plane" for p in plane_indices}

    even_planes, odd_planes = split
    even_indices = list(range(even_planes))
    odd_indices = list(range(even_planes, even_planes + odd_planes))
    top_even = _select_nonredundant_ranked_planes(
        even_indices,
        plane_zeta,
        redundancy,
        max_planes=min(per_branch, len(even_indices)),
        reg_thresh=reg_thresh,
    )
    top_odd = _select_nonredundant_ranked_planes(
        odd_indices,
        plane_zeta,
        redundancy,
        max_planes=min(per_branch, len(odd_indices)),
        reg_thresh=reg_thresh,
    )
    plane_branch = {p: "even" for p in top_even}
    plane_branch.update({p: "odd" for p in top_odd})
    return top_even + top_odd, plane_zeta, plane_branch


def _plane_title(p: int, zeta: float, branch: str | None = None) -> str:
    branch_label = f" [{branch}]" if branch and branch != "plane" else ""
    return f"Plane {p}{branch_label}  (dims {2*p}, {2*p+1})  ζ={zeta:.2f}"


def _simple_plane_zeta_title(p: int, zeta: float) -> str:
    return rf"Plane {p}: $\zeta$={zeta:.2f}"


def _plane_zeta_ranking_rows(
    F_hat: np.ndarray,
    cfg: Config | None = None,
    reg_thresh: float = 1.0,
) -> list[dict]:
    """Return all native planes ranked by validation ζ."""
    K, d, T = F_hat.shape
    D = d // 2
    planes = F_hat.reshape(K, D, 2, T)
    all_plane_indices = list(range(D))
    ranked_branch_plotted_planes, _, _ = _plane_indices_for_ranked_branch_plot(
        planes,
        d,
        cfg,
        reg_thresh=reg_thresh,
    )
    ranked_branch_plotted_planes = set(ranked_branch_plotted_planes)
    split = _mixed_parity_plane_split(d, cfg)
    if split is None:
        branch_labels = {p: "" for p in all_plane_indices}
    else:
        even_planes, odd_planes = split
        branch_labels = {p: "even" for p in range(even_planes)}
        branch_labels.update({
            p: "odd"
            for p in range(even_planes, even_planes + odd_planes)
        })
    plane_zeta = _plane_zeta_values(planes, all_plane_indices)
    rows = sorted(
        (
            {
                "rank": 0,
                "plane": p,
                "branch": branch_labels[p],
                "dim_even": 2 * p,
                "dim_odd": 2 * p + 1,
                "validation_zeta": plane_zeta[p],
                "is_ranked_branch_plotted_02_04_05": p in ranked_branch_plotted_planes,
            }
            for p in all_plane_indices
        ),
        key=lambda row: row["validation_zeta"],
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _branch_zeta_sorted_plane_rows(F_hat: np.ndarray, cfg: Config | None = None) -> list[dict]:
    """Return planes ordered as even branch by ζ, then odd branch by ζ."""
    rows = _plane_zeta_ranking_rows(F_hat, cfg)
    if not any(row["branch"] for row in rows):
        return rows
    even_rows = [row for row in rows if row["branch"] == "even"]
    odd_rows = [row for row in rows if row["branch"] == "odd"]
    other_rows = [row for row in rows if row["branch"] not in {"even", "odd"}]
    return even_rows + odd_rows + other_rows


def write_plane_zeta_ranking(
    F_hat: np.ndarray,
    out_path: str,
    cfg: Config | None = None,
    reg_thresh: float = 1.0,
):
    """Write all native planes ranked by validation ζ."""
    rows = _plane_zeta_ranking_rows(F_hat, cfg, reg_thresh=reg_thresh)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "plane",
                "branch",
                "dim_even",
                "dim_odd",
                "validation_zeta",
                "is_ranked_branch_plotted_02_04_05",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {out_path}")


def plot_plane_zeta_bars(
    F_hat: np.ndarray,
    out_path: str,
    cfg: Config | None = None,
    reg_thresh: float = 1.0,
):
    """Bar chart of validation ζ for all planes, coloured by mixed-parity branch."""
    rows = _plane_zeta_ranking_rows(F_hat, cfg, reg_thresh=reg_thresh)
    zeta = [row["validation_zeta"] for row in rows]
    colors = [
        "tab:blue" if row["branch"] == "even"
        else "tab:red" if row["branch"] == "odd"
        else "0.45"
        for row in rows
    ]

    fig_width = min(max(len(rows) * 0.18, 8), 18)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    x = np.arange(len(rows))
    ax.bar(x, zeta, color=colors, width=0.85)
    ax.set_title("Validation ζ by plane, ranked", fontsize=11)
    ax.set_xlabel("Plane rank", fontsize=9)
    ax.set_ylabel("Validation ζ", fontsize=9)
    ax.set_ylim(0, max(1.0, max(zeta) * 1.05 if zeta else 1.0))
    tick_step = max(1, len(rows) // 16)
    ticks = x[::tick_step]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(i + 1) for i in ticks], rotation=45, ha="right")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D([0], [0], color="tab:blue", lw=6, label="even planes"),
        Line2D([0], [0], color="tab:red", lw=6, label="odd planes"),
    ]
    if any(row["branch"] == "" for row in rows):
        handles.append(Line2D([0], [0], color="0.45", lw=6, label="planes"))
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


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

def _plot_planes_time_coded(
    F_hat,
    groups,
    s_ratio,
    out_path,
    cmap_name="coolwarm",
    cfg: Config | None = None,
    individual_trials_per_condition: int = 0,
    trial_sample_seed: int = 0,
    reg_thresh: float = 1.0,
):
    """Subplot grid: one panel per 2D rotation plane, time-coded."""
    K, d, T = F_hat.shape
    D = d // 2
    planes = F_hat.reshape(K, D, 2, T)
    plane_indices, plane_zeta, plane_branch = _plane_indices_for_ranked_branch_plot(
        planes,
        d,
        cfg,
        reg_thresh=reg_thresh,
    )
    plot_groups = _sample_trials_per_condition(
        groups,
        individual_trials_per_condition,
        trial_sample_seed,
    )
    plot_individual_trials = individual_trials_per_condition > 0
    cmap = plt.get_cmap(cmap_name)

    ncols = min(len(plane_indices), 4)
    nrows = (len(plane_indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)

    for panel_idx, p in enumerate(plane_indices):
        ax = axes[panel_idx // ncols, panel_idx % ncols]
        for cond_key in plot_groups:
            idx_list = plot_groups[cond_key]
            if plot_individual_trials:
                trajectories = (planes[trial_idx, p] for trial_idx in idx_list)
                linewidth = 0.75
                alpha = 0.45
            else:
                trajectories = (planes[idx_list, p].mean(axis=0),)
                linewidth = 1.1
                alpha = 0.85
            for traj in trajectories:
                x, y = traj[0], traj[1]
                for t in range(T - 1):
                    ax.plot(x[t:t+2], y[t:t+2], color=cmap(t / (T - 1)),
                            lw=linewidth, alpha=alpha)

        ax.axhline(0, color="k", lw=0.4, alpha=0.25)
        ax.axvline(0, color="k", lw=0.4, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(_simple_plane_zeta_title(p, plane_zeta[p]), fontsize=18)
        ax.set_xlabel(f"dim {2*p}", fontsize=12)
        ax.set_ylabel(f"dim {2*p+1}", fontsize=12)
        ax.tick_params(labelsize=10.5)
        ax.set_aspect("equal", adjustable="datalim")

    for panel_idx in range(len(plane_indices), nrows * ncols):
        axes[panel_idx // ncols, panel_idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _plot_planes_condition_hsv(
    F_hat,
    groups,
    colors,
    s_ratio,
    out_path,
    hand_windows_val=None,
    individual_trials_per_condition: int = 0,
    trial_sample_seed: int = 0,
    cfg: Config | None = None,
    reg_thresh: float = 1.0,
):
    """Subplot grid: hand trajectories (if available) + one panel per 2D rotation plane.

    When hand_windows_val is provided the first panel shows val-set hand
    trajectories using the *same* condition groups and HSV colours as the
    embedding planes, giving a direct side-by-side comparison.
    """
    K, d, T = F_hat.shape
    D = d // 2
    planes = F_hat.reshape(K, D, 2, T)
    plane_indices, plane_zeta, plane_branch = _plane_indices_for_ranked_branch_plot(
        planes,
        d,
        cfg,
        reg_thresh=reg_thresh,
    )
    plot_groups = _sample_trials_per_condition(
        groups,
        individual_trials_per_condition,
        trial_sample_seed,
    )
    plot_individual_trials = individual_trials_per_condition > 0

    has_hand = False # hand_windows_val is not None
    n_panels = (1 if has_hand else 0) + len(plane_indices)
    ncols = min(n_panels, 4)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)

    panel = 0

    if has_hand:
        ax = axes[0, 0]
        for cond_key in plot_groups:
            idx_list = plot_groups[cond_key]
            color = colors[cond_key]
            if plot_individual_trials:
                for trial_idx in idx_list:
                    hand = hand_windows_val[trial_idx]
                    ax.plot(hand[0], hand[1], lw=0.75, color=color, alpha=0.35)
                    ax.scatter(hand[0, 0], hand[1, 0], color=color, s=8, zorder=5, alpha=0.55)
            else:
                mean_hand = hand_windows_val[idx_list].mean(axis=0)
                ax.plot(mean_hand[0], mean_hand[1], lw=1.4, color=color, alpha=0.9)
                ax.scatter(mean_hand[0, 0], mean_hand[1, 0], color=color, s=25, zorder=5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("Hand trajectory (val)", fontsize=9)
        ax.set_xlabel("hand_x", fontsize=8)
        ax.set_ylabel("hand_y", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="datalim")
        panel = 1

    for plane_panel_idx, p in enumerate(plane_indices):
        idx = panel + plane_panel_idx
        ax = axes[idx // ncols, idx % ncols]
        for cond_key in plot_groups:
            idx_list = plot_groups[cond_key]
            color = colors[cond_key]
            if plot_individual_trials:
                for trial_idx in idx_list:
                    traj = planes[trial_idx, p]
                    ax.plot(traj[0], traj[1], lw=0.75, color=color, alpha=0.35)
                    ax.scatter(traj[0, 0], traj[1, 0], color=color, s=8, zorder=5, alpha=0.55)
            else:
                mean_traj = planes[idx_list, p].mean(axis=0)  # (2, T)
                ax.plot(mean_traj[0], mean_traj[1], lw=1.4, color=color, alpha=0.9)
                ax.scatter(mean_traj[0, 0], mean_traj[1, 0], color=color, s=25, zorder=5)
        # ax.set_xlim(-6,6)
        # ax.set_ylim(-6,6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(_simple_plane_zeta_title(p, plane_zeta[p]), fontsize=24)
        ax.set_xlabel(f"dim {2*p}", fontsize=18)
        ax.set_ylabel(f"dim {2*p+1}", fontsize=18)
        ax.tick_params(labelsize=10.5)
        ax.set_aspect("equal", adjustable="datalim")

    for i in range(n_panels, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    # else:
    #     fig.suptitle(f"Embeddings coded by condition (ζ = {s_ratio:.2f},  "
    #                  f"{n_conds} conditions, {n_per:.1f} trials/cond avg)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── plot 5: per-plane condition-averaged, dims vs time ───────────────────────

def _plot_planes_condition_time(
    F_hat,
    groups,
    colors,
    s_ratio,
    out_path,
    cfg: Config | None = None,
    condition_labels: dict | None = None,
    reg_thresh: float = 1.0,
):
    """Subplot grid: one panel per 2D rotation plane, condition-avg, dims vs time.

    Like plot 4 but instead of plotting the two plane dims against each other,
    both are plotted against time. Each condition keeps its HSV colour; the even
    coordinate (dim 2p, the plot-4 x-axis) is a solid line and the odd
    coordinate (dim 2p+1, the plot-4 y-axis) a dashed line.
    """
    K, d, T = F_hat.shape
    D = d // 2
    planes = F_hat.reshape(K, D, 2, T)
    plane_indices, plane_zeta, plane_branch = _plane_indices_for_ranked_branch_plot(
        planes,
        d,
        cfg,
        reg_thresh=reg_thresh,
    )
    t_axis = np.arange(T)

    ncols = min(len(plane_indices), 4)
    nrows = (len(plane_indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)

    for panel_idx, p in enumerate(plane_indices):
        ax = axes[panel_idx // ncols, panel_idx % ncols]
        for cond_key in groups:
            idx_list = groups[cond_key]
            mean_traj = planes[idx_list, p].mean(axis=0)  # (2, T)
            color = colors[cond_key]
            ax.plot(t_axis, mean_traj[0], lw=1.4, color=color, alpha=0.9,
                    ls="-")
            ax.plot(t_axis, mean_traj[1], lw=1.4, color=color, alpha=0.9,
                    ls="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(_plane_title(p, plane_zeta[p], plane_branch[p]), fontsize=12)
        ax.set_xlabel("time (bins)", fontsize=12)
        ax.set_ylabel("embedding value", fontsize=12)
        ax.tick_params(labelsize=7)

    # Style legend (shared): solid = even/x dim, dashed = odd/y dim.
    style_handles = [
        Line2D([0], [0], color="0.3", lw=1.4, ls="-", label="dim 2p"),
        Line2D([0], [0], color="0.3", lw=1.4, ls="--", label="dim 2p+1"),
    ]
    condition_handles = [
        Line2D(
            [0],
            [0],
            color=colors[cond_key],
            lw=2.0,
            label=(condition_labels or {}).get(cond_key, str(cond_key)),
        )
        for cond_key in groups
    ]
    legend_ax = axes[0, 0]
    style_legend = legend_ax.legend(
        handles=style_handles,
        fontsize=8,
        loc="upper left",
        frameon=False,
    )
    legend_ax.add_artist(style_legend)
    legend_ax.legend(
        handles=condition_handles,
        title="condition",
        fontsize=8,
        title_fontsize=8,
        loc="upper right",
        frameon=False,
    )

    for i in range(len(plane_indices), nrows * ncols):
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


# ── plot 7: covariance heatmaps ───────────────────────────────────────────────

def _embedding_correlation(F_hat: np.ndarray) -> np.ndarray:
    """Empirical correlation matrix of the d embedding dims."""
    K, d, T = F_hat.shape
    Z = F_hat.transpose(0, 2, 1).reshape(K * T, d)
    Z = Z - Z.mean(axis=0)
    Z = Z / (Z.std(axis=0) + 1e-6)
    return (Z.T @ Z) / Z.shape[0]


def plot_covariance_heatmap(F_hat, out_path):
    """Empirical correlation matrix of the d embedding dims.

    Math
    ----
    Z̃_{m,i} = (Z_{m,i} − μ_i) / σ_i
    Corr_{ij} = (1/M) Σ_m Z̃_{m,i} Z̃_{m,j}
    Barlow-Twins target: Corr = I.  Off-diagonal mass ⇒ redundant dims.
    """
    d = F_hat.shape[1]
    Corr = _embedding_correlation(F_hat)

    fig_size = min(max(d / 16, 6), 10)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(Corr, cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    off_diag_mean = float(np.abs(Corr - np.eye(d)).mean())
    ax.set_title(
        f"Embedding correlation  ({d} dims)\n"
        f"mean |Corr − I| = {off_diag_mean:.4f}  (0 = identity)",
        fontsize=10,
    )
    ax.set_xlabel("Embedding dim", fontsize=9)
    ax.set_ylabel("Embedding dim", fontsize=9)
    if d > 32:
        tick_step = max(1, d // 8)
        ticks = np.arange(0, d, tick_step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_zeta_sorted_correlation_heatmap(
    F_hat: np.ndarray,
    out_path: str,
    cfg: Config | None = None,
):
    """Correlation heatmap with plane dims kept adjacent and planes sorted by ζ."""
    d = F_hat.shape[1]
    rows = _branch_zeta_sorted_plane_rows(F_hat, cfg)
    dim_order = [
        dim
        for row in rows
        for dim in (row["dim_even"], row["dim_odd"])
    ]
    Corr = _embedding_correlation(F_hat)
    Corr_sorted = Corr[np.ix_(dim_order, dim_order)]

    fig_size = min(max(d / 16, 7), 11)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(Corr_sorted, cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    off_diag_mean = float(np.abs(Corr - np.eye(d)).mean())
    ax.set_title(
        "Embedding correlation sorted by validation ζ\n"
        f"planes kept paired; mean |Corr − I| = {off_diag_mean:.4f}",
        fontsize=10,
    )
    ax.set_xlabel("Plane order: even branch then odd branch, each ranked by ζ", fontsize=9)
    ax.set_ylabel("Plane order: even branch then odd branch, each ranked by ζ", fontsize=9)

    plane_centers = np.arange(len(rows)) * 2 + 0.5
    tick_indices = _correlation_plane_tick_indices(rows)
    ticks = plane_centers[tick_indices]
    labels = [str(rows[i]["plane"]) for i in tick_indices]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for tick_label, i in zip(ax.get_xticklabels(), tick_indices):
        tick_label.set_color(_branch_tick_color(rows[i]["branch"]))
    for tick_label, i in zip(ax.get_yticklabels(), tick_indices):
        tick_label.set_color(_branch_tick_color(rows[i]["branch"]))

    boundaries = np.arange(2, d, 2) - 0.5
    ax.vlines(boundaries, -0.5, d - 0.5, color="white", lw=0.25, alpha=0.45)
    ax.hlines(boundaries, -0.5, d - 0.5, color="white", lw=0.25, alpha=0.45)

    branch_counts = {
        "even": sum(1 for row in rows if row["branch"] == "even"),
        "odd": sum(1 for row in rows if row["branch"] == "odd"),
    }
    even_dim_count = 2 * branch_counts["even"]
    if 0 < even_dim_count < d:
        boundary = even_dim_count - 0.5
        ax.axvline(boundary, color="k", lw=1.2, alpha=0.8)
        ax.axhline(boundary, color="k", lw=1.2, alpha=0.8)
        ax.text(
            even_dim_count / 2 - 0.5,
            -2.2,
            "even planes",
            color="tab:blue",
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )
        ax.text(
            even_dim_count + (d - even_dim_count) / 2 - 0.5,
            -2.2,
            "odd planes",
            color="tab:red",
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )

    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _correlation_plane_tick_indices(rows: list[dict]) -> np.ndarray:
    """Choose readable plane-center tick positions for a sorted correlation matrix."""
    if len(rows) <= 32:
        return np.arange(len(rows), dtype=int)
    indices = []
    for branch in ("even", "odd"):
        branch_indices = [i for i, row in enumerate(rows) if row["branch"] == branch]
        if not branch_indices:
            continue
        step = max(1, len(branch_indices) // 8)
        indices.extend(branch_indices[::step])
    if not indices:
        step = max(1, len(rows) // 16)
        indices = list(range(0, len(rows), step))
    return np.array(sorted(set(indices)), dtype=int)


def _branch_tick_color(branch: str) -> str:
    if branch == "even":
        return "tab:blue"
    if branch == "odd":
        return "tab:red"
    return "0.2"


def plot_block_cca_plane_heatmap(
    F_hat: np.ndarray,
    out_path: str,
    cfg: Config | None = None,
):
    """Heatmap of ‖C_pq‖_F² between whitened 2D planes."""
    F = torch.from_numpy(F_hat)
    eps = float(getattr(cfg, "block_cca_eps", 1e-4)) if cfg is not None else 1e-4
    with torch.no_grad():
        X = _whiten_2d(_plane_samples(F), eps=eps)
        D, M, _ = X.shape
        C = torch.einsum("pmi,qmj->pqij", X, X) / M
        C_frob_sq = C.pow(2).sum(dim=(-1, -2)).cpu().numpy()

    C_plot = C_frob_sq.copy()
    np.fill_diagonal(C_plot, np.nan)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.88")

    fig_size = min(max(D / 8, 7), 11)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    finite_vals = C_plot[np.isfinite(C_plot)]
    vmax = float(np.percentile(finite_vals, 99)) if finite_vals.size else 1.0
    vmax = max(vmax, 1e-8)
    im = ax.imshow(C_plot, cmap=cmap, vmin=0, vmax=vmax,
                   interpolation="nearest", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r"$\|C_{pq}\|_F^2$")

    off_diag_mean = float(finite_vals.mean()) if finite_vals.size else float("nan")
    ax.set_title(
        "Cross-plane Regularisation Metric\n"
        rf"off-diagonal mean $\|C_{{pq}}\|_F^2$ = {off_diag_mean:.4f}",
        fontsize=10,
    )
    ax.set_xlabel("Plane", fontsize=9)
    ax.set_ylabel("Plane", fontsize=9)

    tick_indices = _native_plane_tick_indices(D, cfg)
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels([str(i) for i in tick_indices], rotation=45, ha="right")
    ax.set_yticklabels([str(i) for i in tick_indices])

    split = _mixed_parity_plane_split(2 * D, cfg)
    if split is not None:
        even_planes, odd_planes = split
        for tick_label, i in zip(ax.get_xticklabels(), tick_indices):
            tick_label.set_color("tab:blue" if i < even_planes else "tab:red")
        for tick_label, i in zip(ax.get_yticklabels(), tick_indices):
            tick_label.set_color("tab:blue" if i < even_planes else "tab:red")
        if 0 < even_planes < D:
            boundary = even_planes - 0.5
            ax.axvline(boundary, color="white", lw=1.3, alpha=0.9)
            ax.axhline(boundary, color="white", lw=1.3, alpha=0.9)
            ax.text(
                even_planes / 2 - 0.5,
                -1.2,
                "even",
                color="tab:blue",
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=False,
            )
            ax.text(
                even_planes + odd_planes / 2 - 0.5,
                -1.2,
                "odd",
                color="tab:red",
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=False,
            )

    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _native_plane_tick_indices(D: int, cfg: Config | None = None) -> np.ndarray:
    if D <= 32:
        return np.arange(D, dtype=int)
    split = _mixed_parity_plane_split(2 * D, cfg)
    if split is None:
        step = max(1, D // 16)
        return np.arange(0, D, step, dtype=int)
    even_planes, odd_planes = split
    indices = []
    for start, count in ((0, even_planes), (even_planes, odd_planes)):
        if count <= 0:
            continue
        step = max(1, count // 8)
        indices.extend(range(start, start + count, step))
    return np.array(sorted(set(indices)), dtype=int)


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
        context_bins=(
            getattr(cfg, "temporal_context_bins", 0)
            if getattr(cfg, "temporal_filters", 0) > 0
            else 0
        ),
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


# ── main ─────────────────────────────────────────────────────────────────────

def _resolve_run_dir(arg_run, require_checkpoint: bool = True):
    if arg_run is not None and not arg_run.isdigit():
        return arg_run

    runs_root = RUNS_DIR
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"No runs directory at {runs_root!r}. Run `python main.py` first.")
    marker = os.path.join("checkpoints", "best.pt") if require_checkpoint else os.path.join("outputs", "log.csv")
    completed = sorted(
        [os.path.join(runs_root, d) for d in os.listdir(runs_root)
         if os.path.isfile(os.path.join(runs_root, d, marker))],
        key=os.path.getmtime, reverse=True,
    )
    if not completed:
        requirement = "completed runs" if require_checkpoint else "runs with outputs/log.csv"
        raise FileNotFoundError(f"No {requirement} found in {runs_root!r}.")

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
    hsv04_trials_per_condition: int = 0,
    hsv04_condition_indices: str = "",
    hsv04_condition_count: int = 0,
    hsv04_trial_seed: int = 0,
    reg_thresh: float = 1.0,
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
    reg_thresh    : skip a ranked plane if ‖C_pq‖_F² >= this for any kept plane
    """

    out_dir = os.path.join(run_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    val_indices = list(val_ds.indices)
    trial_info_val = trial_info.iloc[val_indices].reset_index(drop=True)
    all_cond_groups, all_cond_colors = _get_condition_groups(trial_info_val)
    cond_groups = all_cond_groups
    cond_colors = all_cond_colors

    if cond_start is not None or cond_stop is not None or cond_skip is not None:
        all_keys = list(cond_groups.keys())
        keep = all_keys[cond_start:cond_stop:cond_skip]
        print(f"Condition slice [{cond_start}:{cond_stop}:{cond_skip}]: "
              f"plotting {len(keep)} of {len(all_keys)} conditions")
        cond_groups = {k: cond_groups[k] for k in keep}
        cond_colors = {k: cond_colors[k] for k in keep}

    hsv04_groups = cond_groups
    hsv04_colors = cond_colors
    hsv04_indices = _parse_index_list(hsv04_condition_indices)
    if hsv04_indices:
        hsv04_groups = _select_groups_by_indices(all_cond_groups, hsv04_indices)
        hsv04_colors = {k: all_cond_colors[k] for k in hsv04_groups}
        print(
            "Plot 04 condition indices: "
            f"{hsv04_indices} from full sorted condition order"
        )
    if hsv04_condition_count and hsv04_condition_count > 0:
        keys = list(hsv04_groups.keys())[:hsv04_condition_count]
        hsv04_groups = {k: hsv04_groups[k] for k in keys}
        hsv04_colors = {k: hsv04_colors[k] for k in keys}
        print(f"Plot 04 condition count cap: plotting {len(keys)} conditions")
    if hsv04_trials_per_condition > 0:
        print(
            "Plot 04 individual trials: "
            f"up to {hsv04_trials_per_condition} per condition, "
            f"{len(hsv04_groups)} conditions"
        )

    selected_trial_indices = [0, 25, 50, 75, 100]
    selected_trial_groups, selected_trial_missing = _select_available_groups_by_indices(
        all_cond_groups,
        selected_trial_indices,
    )
    if selected_trial_groups:
        selected_trial_colors = {
            k: all_cond_colors[k]
            for k in selected_trial_groups
        }
        selected_trial_present = [
            idx
            for idx in selected_trial_indices
            if idx not in selected_trial_missing
        ]
        print(
            "Selected-trial plots 02b/04b condition indices: "
            f"{selected_trial_present} from full sorted condition order; "
            "up to 3 trials per condition"
        )
        if selected_trial_missing:
            print(f"Selected-trial plots skipped missing condition indices: {selected_trial_missing}")
    else:
        selected_trial_colors = {}
        print("Selected-trial plots skipped: no requested condition indices are available")

    plot05_indices = [0, 25, 50, 75, 100]
    plot05_groups, plot05_missing = _select_available_groups_by_indices(
        all_cond_groups,
        plot05_indices,
    )
    if not plot05_groups:
        plot05_groups = cond_groups
        plot05_colors = cond_colors
        plot05_labels = {k: str(k) for k in plot05_groups}
        print(
            "Plot 05 requested condition indices were all out of range; "
            "falling back to the active condition set"
        )
    else:
        plot05_colors = {k: all_cond_colors[k] for k in plot05_groups}
        plot05_labels = {
            k: f"{idx}"
            for idx, k in zip(
                [idx for idx in plot05_indices if idx not in plot05_missing],
                plot05_groups,
            )
        }
        print(
            "Plot 05 condition indices: "
            f"{[idx for idx in plot05_indices if idx not in plot05_missing]} "
            "from full sorted condition order"
        )
        if plot05_missing:
            print(f"Plot 05 skipped missing condition indices: {plot05_missing}")

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

    write_plane_zeta_ranking(
        F_hat,
        out_path=os.path.join(out_dir, "06_plane_validation_zeta_ranking.csv"),
        cfg=cfg,
        reg_thresh=reg_thresh,
    )
    plot_plane_zeta_bars(
        F_hat,
        out_path=os.path.join(out_dir, "06_plane_validation_zeta_bars.png"),
        cfg=cfg,
        reg_thresh=reg_thresh,
    )

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
        cfg=cfg,
        reg_thresh=reg_thresh,
    )
    if selected_trial_groups:
        _plot_planes_time_coded(
            F_hat,
            selected_trial_groups,
            s_ratio_val,
            out_path=os.path.join(out_dir, "02b_embed_planes_time_coded_selected_trials.png"),
            cfg=cfg,
            individual_trials_per_condition=3,
            trial_sample_seed=hsv04_trial_seed,
            reg_thresh=reg_thresh,
        )
    _plot_condition_hsv(
        phasors_raw, cond_groups, cond_colors,
        title=f"Raw — condition-averaged (ch {ch_a} vs ch {ch_b})",
        xlabel=f"Ch {ch_a} (z-scored)", ylabel=f"Ch {ch_b} (z-scored)",
        out_path=os.path.join(out_dir, "03_raw_condition_hsv.png"),
    )
    _plot_planes_condition_hsv(
        F_hat, hsv04_groups, hsv04_colors, s_ratio_val,
        out_path=os.path.join(out_dir, "04_embed_planes_condition_hsv.png"),
        hand_windows_val=hand_windows_val,
        individual_trials_per_condition=hsv04_trials_per_condition,
        trial_sample_seed=hsv04_trial_seed,
        cfg=cfg,
        reg_thresh=reg_thresh,
    )
    if selected_trial_groups:
        _plot_planes_condition_hsv(
            F_hat,
            selected_trial_groups,
            selected_trial_colors,
            s_ratio_val,
            out_path=os.path.join(out_dir, "04b_embed_planes_condition_hsv_selected_trials.png"),
            hand_windows_val=hand_windows_val,
            individual_trials_per_condition=3,
            trial_sample_seed=hsv04_trial_seed,
            cfg=cfg,
            reg_thresh=reg_thresh,
        )
    _plot_planes_condition_time(
        F_hat, plot05_groups, plot05_colors, s_ratio_val,
        out_path=os.path.join(out_dir, "05_embed_planes_condition_time.png"),
        cfg=cfg,
        condition_labels=plot05_labels,
        reg_thresh=reg_thresh,
    )
    plot_covariance_heatmap(
        F_hat, out_path=os.path.join(out_dir, "07_covariance_heatmap.png"),
    )
    plot_zeta_sorted_correlation_heatmap(
        F_hat,
        out_path=os.path.join(out_dir, "07_zeta_sorted_correlation_heatmap.png"),
        cfg=cfg,
    )
    plot_block_cca_plane_heatmap(
        F_hat,
        out_path=os.path.join(out_dir, "07_block_cca_plane_heatmap.png"),
        cfg=cfg,
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
    parser.add_argument("--hsv04-trials-per-condition", type=int, default=0,
                        help="Plot this many individual trials per selected condition in plot 04. Default: condition means.")
    parser.add_argument("--hsv04-condition-indices", default="",
                        help="Comma-separated condition indices for plot 04, using the full sorted condition order.")
    parser.add_argument("--hsv04-condition-count", type=int, default=0,
                        help="Limit plot 04 to the first K selected conditions. Default: no separate cap.")
    parser.add_argument("--reg-thresh", type=float, default=1.0,
                        help="Skip a zeta-ranked embedding plane if its cross-plane regularisation score "
                             "with any already selected plane is at least this value. Default: 1.0.")
    parser.add_argument("--only-loss", action="store_true",
                        help="Only regenerate outputs/loss_curve.png from outputs/log.csv.")
    args = parser.parse_args()
    if args.hsv04_trials_per_condition < 0:
        parser.error("--hsv04-trials-per-condition must be non-negative")
    if args.hsv04_condition_count < 0:
        parser.error("--hsv04-condition-count must be non-negative")

    run_dir = _resolve_run_dir(args.run, require_checkpoint=not args.only_loss)
    print(f"Using run: {os.path.basename(run_dir)}")

    if args.only_loss:
        plot_loss_curve(run_dir)
        return

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
    temporal_context_bins = (
        int(getattr(cfg, "temporal_context_bins", 0))
        if getattr(cfg, "temporal_filters", 0) > 0
        else 0
    )
    windows = make_windows(
        X_smooth, trial_info, time_index_s, bin_width_s,
        strategy=cfg.window_strategy, window_size=cfg.window_size,
        align_field=getattr(cfg, "align_field", "move_onset_time"),
        pre_ms=getattr(cfg, "pre_ms", 100),
        context_bins=temporal_context_bins,
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
        antisymmetric_planes=getattr(cfg, "antisymmetric_planes", 0),
        temporal_context_bins=getattr(cfg, "temporal_context_bins", 0),
    )
    model.load_state_dict(state_dict)

    make_diagnostic_plots(
        model=model,
        val_ds=val_ds,
        trial_info=trial_info,
        cfg=cfg,
        run_dir=run_dir,
        hand_windows=hand_windows,
        cond_start=args.start,
        cond_stop=args.stop,
        cond_skip=args.skip,
        hsv04_trials_per_condition=args.hsv04_trials_per_condition,
        hsv04_condition_indices=args.hsv04_condition_indices,
        hsv04_condition_count=args.hsv04_condition_count,
        hsv04_trial_seed=args.seed,
        reg_thresh=args.reg_thresh,
    )
    plot_loss_curve(run_dir, cfg)


if __name__ == "__main__":
    main()
