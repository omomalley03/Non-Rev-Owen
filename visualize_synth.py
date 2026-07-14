"""
Visualise non-reversibility quality for synthetic rotation data.

Each diagnostic plot is saved as its own PNG in the run's `outputs/` dir:

  01_raw_time_coded.png           — raw input, time-coded (top-2 input PCA)
  02_embed_planes_time_coded.png  — one subplot per 2D rotation plane, time-coded
  04_embed_planes_participant_hsv.png — participant-avg embedding, HSV by participant ID
  07_covariance_heatmap.png       — embedding correlation matrix
  08_between_within_variance.png  — trial-discriminability over time
  09_embedding_norm_distribution.png — embedding norm distribution
  14/15_*_participant_conditions.png — selected participants, colour-coded by class label
  16/17_*_participant_condition_means.png — selected participants, condition-avg trajectories
  18_val_condition_means_all_trials.png — all validation trials, condition-avg trajectories

Usage
-----
    python visualize_synth.py               # most recent run
    python visualize_synth.py --run 2       # 2nd most recent
    python visualize_synth.py --run synth_runs/x  # explicit path
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.collections import LineCollection
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset

from config import Config
from paths import SYNTH_RUNS_DIR
from model import MLP, infer_multiscale_symmetric_conv_layers
from loss import S_ratio as compute_S_ratio, _batch_rms_normalize, _whiten_2d, _plane_samples
from synth_data import load_synthetic_labels, load_synthetic_subjects, load_synthetic_windows


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


def _time_downsample(arr: np.ndarray, max_timepoints: int) -> np.ndarray:
    """Evenly downsample the last axis for plotting long recordings."""
    T = arr.shape[-1]
    if not max_timepoints or T <= max_timepoints:
        return arr
    idx = np.linspace(0, T - 1, max_timepoints).round().astype(int)
    idx = np.unique(idx)
    return arr[..., idx]


def _add_timecoded_lines(ax, xy: np.ndarray, cmap_name: str, linewidth: float, alpha: float):
    """Add all trial trajectories in one LineCollection for speed."""
    K, T, _ = xy.shape
    pts = xy.reshape(K, T, 1, 2)
    segs = np.concatenate([pts[:, :-1], pts[:, 1:]], axis=2).reshape(-1, 2, 2)
    times = np.tile(np.arange(T - 1), K)
    lc = LineCollection(
        segs,
        cmap=plt.get_cmap(cmap_name),
        norm=plt.Normalize(0, T - 1),
        linewidths=linewidth,
        alpha=alpha,
    )
    lc.set_array(times)
    ax.add_collection(lc)
    ax.autoscale()
    return lc


PHYSIONETMI_CONDITION_NAMES = {
    0: "imagined left fist",
    1: "imagined right fist",
    2: "imagined both fists",
    3: "imagined both feet",
}

PHYSIONETMI_CONDITION_COLORS = {
    0: "#1f77b4",
    1: "#d62728",
    2: "#2ca02c",
    3: "#9467bd",
}


def _condition_name(label) -> str:
    label_int = int(label)
    return PHYSIONETMI_CONDITION_NAMES.get(label_int, f"condition {label_int}")


def _condition_color(label):
    label_int = int(label)
    if label_int in PHYSIONETMI_CONDITION_COLORS:
        return PHYSIONETMI_CONDITION_COLORS[label_int]
    return plt.get_cmap("tab10")(label_int % 10)


def _add_condition_lines(ax, xy: np.ndarray, labels: np.ndarray, linewidth: float, alpha: float):
    """Add trial trajectories coloured by a per-trial class label."""
    K, T, _ = xy.shape
    if len(labels) != K:
        raise ValueError(f"labels length ({len(labels)}) must match trajectories ({K})")

    pts = xy.reshape(K, T, 1, 2)
    segs = np.concatenate([pts[:, :-1], pts[:, 1:]], axis=2).reshape(-1, 2, 2)
    colors = []
    for label in labels:
        colors.extend([_condition_color(label)] * (T - 1))
    lc = LineCollection(segs, colors=colors, linewidths=linewidth, alpha=alpha)
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def _condition_legend_handles(labels: np.ndarray):
    return [
        plt.Line2D([0], [0], color=_condition_color(label), lw=2, label=_condition_name(label))
        for label in np.unique(labels)
    ]


def _dataset_source_indices(ds) -> np.ndarray:
    """Return source-window indices for TensorDataset/Subset split objects."""
    if isinstance(ds, Subset):
        parent = _dataset_source_indices(ds.dataset)
        return parent[np.asarray(ds.indices, dtype=int)]
    return np.arange(len(ds), dtype=int)


def _embed_dataset_sample(model, ds, local_positions: np.ndarray, cfg: Config, device: torch.device) -> np.ndarray:
    """Embed selected local positions from a dataset split."""
    sample_ds = Subset(ds, local_positions.tolist())
    loader = DataLoader(sample_ds, batch_size=len(sample_ds), shuffle=False)
    (tensor,) = next(iter(loader))
    with torch.no_grad():
        F_hat_t = model(tensor.to(device))
        F_hat_t = F_hat_t - F_hat_t.mean(dim=cfg.F_mean_axis, keepdim=True)
    return F_hat_t.cpu().numpy()


def _parse_subject_ids(spec: str) -> np.ndarray:
    spec = str(spec or "").strip()
    if not spec or spec.lower() in {"all", "none"}:
        return np.array([], dtype=np.int64)
    return np.array([int(item.strip()) for item in spec.split(",") if item.strip()], dtype=np.int64)


def _select_subject_sets(
    subjects: np.ndarray,
    rng: np.random.Generator,
    subject_count: int = 0,
    subject_ids: str = "",
    holdout_subject_count: int = 0,
    holdout_subject_ids: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select train/val and held-out subject sets from available subject IDs."""
    unique_subjects = np.unique(subjects)
    explicit_subjects = _parse_subject_ids(subject_ids)

    if explicit_subjects.size:
        missing = np.setdiff1d(explicit_subjects, unique_subjects)
        if missing.size:
            raise ValueError(f"SYNTH_SUBJECT_IDS contains unknown subjects: {missing.tolist()}")
        selected_subjects = np.sort(explicit_subjects)
    elif subject_count and subject_count > 0:
        if subject_count > len(unique_subjects):
            raise ValueError(
                f"SYNTH_SUBJECT_COUNT={subject_count} exceeds available subjects={len(unique_subjects)}"
            )
        selected_subjects = np.sort(rng.choice(unique_subjects, size=subject_count, replace=False))
    else:
        selected_subjects = unique_subjects

    explicit_holdout = _parse_subject_ids(holdout_subject_ids)
    if explicit_holdout.size:
        missing = np.setdiff1d(explicit_holdout, selected_subjects)
        if missing.size:
            raise ValueError(
                "SYNTH_HOLDOUT_SUBJECT_IDS must be within the selected subject pool; "
                f"unknown or unselected subjects: {missing.tolist()}"
            )
        holdout_subjects = np.sort(explicit_holdout)
    elif holdout_subject_count and holdout_subject_count > 0:
        if holdout_subject_count >= len(selected_subjects):
            raise ValueError(
                "SYNTH_HOLDOUT_SUBJECT_COUNT must leave at least one subject for train/val; "
                f"got holdout={holdout_subject_count}, selected={len(selected_subjects)}"
            )
        holdout_subjects = np.sort(
            rng.choice(selected_subjects, size=holdout_subject_count, replace=False)
        )
    else:
        holdout_subjects = np.array([], dtype=selected_subjects.dtype)

    trainval_subjects = np.setdiff1d(selected_subjects, holdout_subjects)
    return selected_subjects, trainval_subjects, holdout_subjects


def train_val_split_synth(
    windows: np.ndarray,
    val_frac: float,
    seed: int,
    split: str = "random",
    subjects: np.ndarray | None = None,
    subject_count: int = 0,
    subject_ids: str = "",
    holdout_subject_count: int = 0,
    holdout_subject_ids: str = "",
    return_holdout: bool = False,
):
    """Split synthetic windows using the same modes as main_synth.py."""
    tensor = torch.from_numpy(windows)
    full_ds = TensorDataset(tensor)

    split = split.lower()
    if split not in {"subject_random", "participant_random"} and (
        holdout_subject_count or _parse_subject_ids(holdout_subject_ids).size
    ):
        raise ValueError("Subject holdout requires SYNTH_SPLIT=subject_random")

    if split in {"train_eq_val", "train_equals_val", "all", "none"}:
        if return_holdout:
            return full_ds, full_ds, None, None, np.array([], dtype=np.int64)
        return full_ds, full_ds
    if split in {"subject_random", "participant_random"}:
        if subjects is None:
            raise ValueError("SYNTH_SPLIT=subject_random requires SYNTH_SUBJECTS_PATH")
        if len(subjects) != len(tensor):
            raise ValueError(
                f"subject IDs length ({len(subjects)}) must match windows length ({len(tensor)})"
            )

        rng = np.random.default_rng(seed)
        _, trainval_subjects, holdout_subjects = _select_subject_sets(
            subjects,
            rng,
            subject_count=subject_count,
            subject_ids=subject_ids,
            holdout_subject_count=holdout_subject_count,
            holdout_subject_ids=holdout_subject_ids,
        )

        eligible = np.flatnonzero(np.isin(subjects, trainval_subjects))
        if len(eligible) < 2:
            raise ValueError("subject_random split requires at least two eligible trials")
        shuffled = rng.permutation(eligible)
        n_val = max(1, int(len(shuffled) * val_frac))
        n_train = len(shuffled) - n_val
        if n_train < 1:
            raise ValueError("subject_random split leaves no training trials")
        train_ds = Subset(full_ds, shuffled[:n_train].tolist())
        val_ds = Subset(full_ds, shuffled[n_train:].tolist())
        if return_holdout:
            holdout_idx = np.flatnonzero(np.isin(subjects, holdout_subjects))
            holdout_ds = Subset(full_ds, holdout_idx.tolist()) if len(holdout_idx) else None
            return train_ds, val_ds, holdout_ds, trainval_subjects, holdout_subjects
        return train_ds, val_ds
    if split != "random":
        raise ValueError("SYNTH_SPLIT must be one of: random, train_eq_val, subject_random")

    n_val = max(1, int(len(tensor) * val_frac))
    n_train = len(tensor) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)
    if return_holdout:
        return train_ds, val_ds, None, None, np.array([], dtype=np.int64)
    return train_ds, val_ds


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
    xy = phasors[idx].transpose(0, 2, 1)  # (K, T, 2)
    _add_timecoded_lines(ax, xy, cmap_name, linewidth=0.8, alpha=0.5)

    ax.axhline(0, color="k", lw=0.4, alpha=0.25)
    ax.axvline(0, color="k", lw=0.4, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=plt.Normalize(0, T - 1))
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
        xy = planes[idx, p].transpose(0, 2, 1)  # (K, T, 2)
        _add_timecoded_lines(ax, xy, cmap_name, linewidth=0.8, alpha=0.5)

        ax.axhline(0, color="k", lw=0.4, alpha=0.25)
        ax.axvline(0, color="k", lw=0.4, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        # ax.set_title("Linear embedding")
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
            xy = np.stack([F_hat[idx, x_dim], F_hat[idx, y_dim]], axis=-1)
            _add_timecoded_lines(ax, xy, cmap_name, linewidth=0.6, alpha=0.45)

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


def _plot_planes_participant_hsv(F_hat, participant_ids, s_ratio, out_path):
    """Plot participant-averaged embedding trajectories, coloured by participant ID."""
    if participant_ids is None:
        return
    participant_ids = np.asarray(participant_ids)
    if len(participant_ids) != F_hat.shape[0]:
        raise ValueError(
            f"participant_ids length ({len(participant_ids)}) must match embeddings ({F_hat.shape[0]})"
        )

    K, d, T = F_hat.shape
    D = d // 2
    if D < 1:
        return

    participants = np.unique(participant_ids)
    color_values = plt.get_cmap("hsv")(np.linspace(0, 1, len(participants), endpoint=False))
    participant_cmap = ListedColormap(color_values)
    participant_norm = BoundaryNorm(np.arange(len(participants) + 1) - 0.5, len(participants))
    participant_to_color_idx = {participant: i for i, participant in enumerate(participants)}
    planes = F_hat.reshape(K, D, 2, T)

    ncols = min(D, 4)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for p in range(D):
        ax = axes[p // ncols, p % ncols]
        for participant in participants:
            idx = np.flatnonzero(participant_ids == participant)
            mean_traj = planes[idx, p].mean(axis=0)
            color = participant_cmap(participant_to_color_idx[participant])
            ax.plot(mean_traj[0], mean_traj[1], lw=1.4, color=color, alpha=0.9)
            ax.scatter(mean_traj[0, 0], mean_traj[1, 0], color=color, s=25, zorder=5)

        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"Plane {p}  (dims {2*p}, {2*p+1})", fontsize=12)
        ax.set_xlabel(f"dim {2*p}", fontsize=12)
        ax.set_ylabel(f"dim {2*p+1}", fontsize=12)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="datalim")

    for p in range(D, nrows * ncols):
        axes[p // ncols, p % ncols].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=participant_cmap, norm=participant_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.035, pad=0.03)
    cbar.set_label("participant ID", fontsize=9)
    cbar.set_ticks(np.arange(len(participants)))
    cbar.set_ticklabels([str(int(participant)) for participant in participants])
    cbar.ax.tick_params(labelsize=7)

    counts = [int((participant_ids == participant).sum()) for participant in participants]
    fig.suptitle(
        f"Participant-averaged embedding trajectories  (ζ = {s_ratio:.2f})\n"
        f"{len(participants)} participants, {np.mean(counts):.1f} trials/participant avg",
        fontsize=11,
    )
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
    fig, axes = plt.subplots(D, D, figsize=(6, 4))
    axes = np.asarray(axes).reshape(D, D)
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
    if model.temporal_conv is None:
        return
    if hasattr(model.temporal_conv, "temporal_branches"):
        kernels = []
        titles = []
        branches = list(model.temporal_conv.temporal_branches)
        max_panels = 64
        n_per_branch = max(1, max_panels // len(branches))
        for branch_idx, branch in enumerate(branches):
            if hasattr(branch, "effective_weight"):
                weights_tensor = branch.effective_weight()
                branch_label = f"k{branch.kernel}"
            else:
                weights_tensor = branch.conv.weight
                branch_label = f"b{branch_idx}"
            weights = weights_tensor.detach().cpu().numpy()[:, 0, :]
            for dim_idx, weight in enumerate(weights[:n_per_branch]):
                kernels.append(weight)
                titles.append(f"{branch_label}:{dim_idx}")
        n_show = min(len(kernels), max_panels)
        rows = int(np.ceil(n_show ** 0.5))
        cols = int(np.ceil(n_show / rows))
        fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(2 * cols, 1.5 * rows))
        for i in range(rows):
            for j in range(cols):
                k = i * cols + j
                ax = axes[i, j]
                if k < n_show:
                    ax.plot(kernels[k])
                    ax.set_title(titles[k], fontsize=6)
                else:
                    ax.set_visible(False)
                ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved → {out_path}")
        return
    # depthwise zero-phase weights: (out_ch, 1, kernel_size); plot the effective
    # palindromic time kernel (w + flip(w)) for a sample of output channels.
    out_ch, _, kernel_size = model.temporal_conv.weight.shape
    weights = model.temporal_conv.weight.detach().cpu().numpy()[:, 0, :]   # (out_ch, k)
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


def plot_participant_split_embedding(
    model,
    train_ds,
    val_ds,
    subjects: np.ndarray,
    cfg: Config,
    out_dir: str,
    labels: np.ndarray | None = None,
):
    """Write train/val 2x2 grids of trajectories for selected participant IDs."""
    if subjects is None:
        return
    if labels is not None and len(labels) != len(subjects):
        raise ValueError(f"labels length ({len(labels)}) must match subjects length ({len(subjects)})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    max_timepoints = getattr(cfg, "synth_viz_max_timepoints", 0)

    train_source_idx = _dataset_source_indices(train_ds)
    val_source_idx = _dataset_source_indices(val_ds)
    train_subjects = subjects[train_source_idx]
    val_subjects = subjects[val_source_idx]
    train_labels = labels[train_source_idx] if labels is not None else None
    val_labels = labels[val_source_idx] if labels is not None else None

    train_unique = np.unique(train_subjects)
    val_unique = np.unique(val_subjects)
    participant_count = max(1, int(getattr(cfg, "synth_viz_participant_count", 4)))
    mode = str(getattr(cfg, "synth_viz_participant_mode", "top_zeta")).strip().lower()
    if mode not in {"top_zeta", "random"}:
        raise ValueError("SYNTH_VIZ_PARTICIPANT_MODE must be one of: top_zeta, random")

    shared_subjects = np.intersect1d(train_unique, val_unique)
    candidate_subjects = (
        shared_subjects
        if len(shared_subjects) >= participant_count
        else np.union1d(train_unique, val_unique)
    )
    if len(candidate_subjects) == 0:
        return

    n_subjects = min(participant_count, len(candidate_subjects))
    if mode == "random":
        rng = np.random.default_rng(getattr(cfg, "seed", 0) + 303)
        selected_subjects = np.sort(rng.choice(candidate_subjects, size=n_subjects, replace=False))
    else:
        ranked_subjects = []
        for subject in candidate_subjects:
            val_local_positions = np.flatnonzero(val_subjects == subject)
            if len(val_local_positions) == 0:
                continue
            F_val = _embed_dataset_sample(model, val_ds, val_local_positions, cfg, device)
            F_val = _time_downsample(F_val, max_timepoints)
            if F_val.shape[1] < 2:
                continue
            zeta_val = compute_S_ratio(_batch_rms_normalize(torch.from_numpy(F_val[:, :2, :]))).item()
            ranked_subjects.append((zeta_val, subject))

        if not ranked_subjects:
            return
        ranked_subjects.sort(key=lambda item: item[0], reverse=True)
        selected_subjects = np.array([subject for _, subject in ranked_subjects[:n_subjects]])

    def plot_split(
        name: str,
        ds,
        split_subjects: np.ndarray,
        split_labels: np.ndarray | None,
        out_path: str,
        color_by: str,
    ):
        subject_embeddings = []
        max_d = 0
        for subject in selected_subjects:
            local_positions = np.flatnonzero(split_subjects == subject)
            if len(local_positions) == 0:
                subject_embeddings.append((subject, local_positions, None))
                continue

            F_hat = _embed_dataset_sample(model, ds, local_positions, cfg, device)
            F_hat = _time_downsample(F_hat, max_timepoints)
            max_d = max(max_d, F_hat.shape[1])
            subject_embeddings.append((subject, local_positions, F_hat))

        D = max_d // 2
        if D < 1:
            return

        fig, axes = plt.subplots(
            len(selected_subjects),
            D,
            figsize=(4.0 * D, 3.2 * len(selected_subjects)),
            squeeze=False,
        )
        for row, (subject, local_positions, F_hat) in enumerate(subject_embeddings):
            for p in range(D):
                ax = axes[row, p]
                if row == 0:
                    ax.set_title(f"Plane {p}  (dims {2*p}, {2*p+1})", fontsize=10)
                if p == 0:
                    ax.set_ylabel(f"Participant {int(subject)}\ndim {2*p+1}", fontsize=9)

                if F_hat is None:
                    ax.text(0.5, 0.5, "No trials in split", ha="center", va="center")
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                if F_hat.shape[1] < 2 * (p + 1):
                    ax.text(0.5, 0.5, "Missing plane", ha="center", va="center")
                    ax.set_xticks([]); ax.set_yticks([])
                    continue

                zeta = compute_S_ratio(_batch_rms_normalize(torch.from_numpy(F_hat[:, 2*p:2*p+2, :]))).item()
                xy = F_hat[:, 2*p:2*p+2, :].transpose(0, 2, 1)
                if color_by in {"condition", "condition_mean"}:
                    if split_labels is None:
                        ax.text(0.5, 0.5, "No labels available", ha="center", va="center")
                        continue
                    local_labels = split_labels[local_positions]
                    if color_by == "condition_mean":
                        for label in np.unique(local_labels):
                            label_xy = xy[local_labels == label]
                            mean_xy = label_xy.mean(axis=0)
                            color = _condition_color(label)
                            ax.plot(mean_xy[:, 0], mean_xy[:, 1], lw=1.35, color=color, alpha=0.62)
                            ax.scatter(mean_xy[0, 0], mean_xy[0, 1], color=color, s=18, alpha=0.8, zorder=5)
                    else:
                        _add_condition_lines(ax, xy, local_labels, linewidth=0.5, alpha=0.22)
                else:
                    _add_timecoded_lines(ax, xy, "coolwarm", linewidth=0.65, alpha=0.5)
                ax.axhline(0, color="k", lw=0.35, alpha=0.25)
                ax.axvline(0, color="k", lw=0.35, alpha=0.25)
                ax.spines[["top", "right"]].set_visible(False)
                ax.text(
                    0.02, 0.98,
                    f"n={len(local_positions)}  ζ={zeta:.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                )
                ax.set_xlabel(f"dim {2*p}", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.set_aspect("equal", adjustable="datalim")

        selection_label = "random participants" if mode == "random" else "top validation-ζ participants"
        if color_by == "condition_mean":
            color_label = "condition-averaged"
        elif color_by == "condition":
            color_label = "condition-coded"
        else:
            color_label = "time-coded"
        fig.suptitle(f"{name}: {color_label} trajectories for {selection_label}", fontsize=11)
        if color_by in {"condition", "condition_mean"} and split_labels is not None:
            present_labels = np.unique(split_labels[np.isin(split_subjects, selected_subjects)])
            handles = _condition_legend_handles(present_labels)
            if handles:
                fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), fontsize=8)
        bottom = 0.08 if color_by in {"condition", "condition_mean"} else 0.03
        fig.tight_layout(rect=(0, bottom, 1, 0.96))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out_path}")

    plot_split(
        "Train split",
        train_ds,
        train_subjects,
        None,
        os.path.join(out_dir, "12_train_participant_trajectories.png"),
        color_by="time",
    )
    plot_split(
        "Validation split",
        val_ds,
        val_subjects,
        None,
        os.path.join(out_dir, "13_val_participant_trajectories.png"),
        color_by="time",
    )
    if labels is not None:
        plot_split(
            "Train split",
            train_ds,
            train_subjects,
            train_labels,
            os.path.join(out_dir, "14_train_participant_conditions.png"),
            color_by="condition",
        )
        plot_split(
            "Validation split",
            val_ds,
            val_subjects,
            val_labels,
            os.path.join(out_dir, "15_val_participant_conditions.png"),
            color_by="condition",
        )
        plot_split(
            "Train split",
            train_ds,
            train_subjects,
            train_labels,
            os.path.join(out_dir, "16_train_participant_condition_means.png"),
            color_by="condition_mean",
        )
        plot_split(
            "Validation split",
            val_ds,
            val_subjects,
            val_labels,
            os.path.join(out_dir, "17_val_participant_condition_means.png"),
            color_by="condition_mean",
        )


def plot_condition_means_all_trials(
    F_hat: np.ndarray,
    labels: np.ndarray,
    s_ratio: float,
    out_path: str,
):
    """Plot one condition-averaged embedding trajectory per plane over all trials."""
    labels = np.asarray(labels)
    if len(labels) != F_hat.shape[0]:
        raise ValueError(f"labels length ({len(labels)}) must match embeddings ({F_hat.shape[0]})")

    K, d, T = F_hat.shape
    D = d // 2
    if D < 1:
        return

    planes = F_hat.reshape(K, D, 2, T)
    present_labels = np.unique(labels)
    ncols = min(D, 4)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4 * nrows),
        squeeze=False,
    )

    for p in range(D):
        ax = axes[p // ncols, p % ncols]
        for label in present_labels:
            idx = np.flatnonzero(labels == label)
            mean_traj = planes[idx, p].mean(axis=0)
            color = _condition_color(label)
            ax.plot(mean_traj[0], mean_traj[1], lw=1.5, color=color, alpha=0.62)
            ax.scatter(mean_traj[0, 0], mean_traj[1, 0], color=color, s=24, alpha=0.8, zorder=5)

        ax.axhline(0, color="k", lw=0.4, alpha=0.25)
        ax.axvline(0, color="k", lw=0.4, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"Plane {p}  (dims {2*p}, {2*p+1})", fontsize=11)
        ax.set_xlabel(f"dim {2*p}", fontsize=9)
        ax.set_ylabel(f"dim {2*p+1}", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal", adjustable="datalim")

    for p in range(D, nrows * ncols):
        axes[p // ncols, p % ncols].set_visible(False)

    counts = [int((labels == label).sum()) for label in present_labels]
    handles = _condition_legend_handles(present_labels)
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(4, len(handles)),
            fontsize=8,
        )
    zeta_text = f"  (ζ = {s_ratio:.2f})" if np.isfinite(s_ratio) else ""
    fig.suptitle(
        f"Validation condition-averaged embedding trajectories{zeta_text}\n"
        f"{len(labels)} trials total  |  trials/condition: "
        + ", ".join(f"{_condition_name(label)}={count}" for label, count in zip(present_labels, counts)),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.12 if handles else 0.03, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── main diagnostic entry point ───────────────────────────────────────────────

def make_diagnostic_plots_synth(
    model,
    val_ds,
    cfg: Config,
    run_dir: str,
    train_ds=None,
    subjects: np.ndarray | None = None,
    labels: np.ndarray | None = None,
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
    val_source_idx = _dataset_source_indices(val_ds) if subjects is not None else None
    if labels is not None:
        source_idx = _dataset_source_indices(val_ds)
        if len(source_idx) and int(source_idx.max()) >= len(labels):
            raise ValueError("labels length must cover the validation dataset source indices")
        val_labels_plot = labels[source_idx]
    else:
        val_labels_plot = None
    max_trials = getattr(cfg, "synth_viz_max_trials", 0)
    if max_trials and val_tensor.shape[0] > max_trials:
        print(f"Using first {max_trials} validation trials for diagnostics "
              f"(of {val_tensor.shape[0]})")
        val_tensor = val_tensor[:max_trials]
        if val_source_idx is not None:
            val_source_idx = val_source_idx[:max_trials]
        if val_labels_plot is not None:
            val_labels_plot = val_labels_plot[:max_trials]
    val_participant_ids = subjects[val_source_idx] if val_source_idx is not None else None
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
        s_ratio_val = compute_S_ratio(_batch_rms_normalize(F_hat_t)).item()
        F_hat_t = F_hat_t.cpu()
        F_hat = F_hat_t.numpy()          # (K, d, T)

    max_timepoints = getattr(cfg, "synth_viz_max_timepoints", 0)
    F_hat_plot = _time_downsample(F_hat, max_timepoints)
    if F_hat_plot.shape[-1] != F_hat.shape[-1]:
        print(f"Using {F_hat_plot.shape[-1]} time bins for trajectory plots "
              f"(of {F_hat.shape[-1]})")

    # ── raw input PCA ──────────────────────────────────────────────────────────
    N_in = val_np.shape[1]
    raw_input = val_np[:, :N_in, :]

    raw_flat = raw_input.transpose(0, 2, 1).reshape(K * raw_input.shape[2], N_in)
    raw_pca_mean, raw_pca_Vh2 = _fit_pca2(raw_flat)
    phasors_raw = _windows_to_pca2(raw_input, raw_pca_mean, raw_pca_Vh2)  # (K, 2, T)
    phasors_raw_plot = _time_downsample(phasors_raw, max_timepoints)

    # ── plots ─────────────────────────────────────────────────────────────────
    _plot_all_trials_time_coded(
        phasors_raw_plot,
        title="Raw — time-coded  (top-2 PCA)",
        xlabel="PC 1", ylabel="PC 2",
        out_path=os.path.join(out_dir, "01_raw_time_coded.png"),
    )
    _plot_planes_time_coded(
        F_hat_plot, s_ratio_val,
        out_path=os.path.join(out_dir, "02_embed_planes_time_coded.png"),
    )
    _plot_dim_grid(
        F_hat_plot, s_ratio_val,
        out_path=os.path.join(out_dir, "03_dim_grid_time_coded.png"),
    )
    _plot_planes_participant_hsv(
        F_hat_plot, val_participant_ids, s_ratio_val,
        out_path=os.path.join(out_dir, "04_embed_planes_participant_hsv.png"),
    )
    if val_labels_plot is not None:
        plot_condition_means_all_trials(
            F_hat_plot,
            val_labels_plot,
            s_ratio_val,
            out_path=os.path.join(out_dir, "18_val_condition_means_all_trials.png"),
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
    if train_ds is not None and subjects is not None:
        plot_participant_split_embedding(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            subjects=subjects,
            cfg=cfg,
            out_dir=out_dir,
            labels=labels,
        )
    

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
    parser.add_argument("--participant-count", type=int, default=None,
                        help="Override SYNTH_VIZ_PARTICIPANT_COUNT saved in the checkpoint.")
    parser.add_argument("--participant-mode", choices=["top_zeta", "random"], default=None,
                        help="Override SYNTH_VIZ_PARTICIPANT_MODE saved in the checkpoint.")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Override SYNTH_VIZ_MAX_TRIALS saved in the checkpoint; 0 uses all validation trials.")
    parser.add_argument("--max-timepoints", type=int, default=None,
                        help="Override SYNTH_VIZ_MAX_TIMEPOINTS saved in the checkpoint; 0 uses all time bins.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run)
    print(f"Using run: {os.path.basename(run_dir)}")

    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    env_participant_count = os.environ.get("SYNTH_VIZ_PARTICIPANT_COUNT")
    if args.participant_count is not None:
        cfg.synth_viz_participant_count = args.participant_count
    elif env_participant_count not in {None, ""}:
        cfg.synth_viz_participant_count = int(env_participant_count)

    env_participant_mode = os.environ.get("SYNTH_VIZ_PARTICIPANT_MODE")
    if args.participant_mode is not None:
        cfg.synth_viz_participant_mode = args.participant_mode
    elif env_participant_mode not in {None, ""}:
        cfg.synth_viz_participant_mode = env_participant_mode

    env_max_trials = os.environ.get("SYNTH_VIZ_MAX_TRIALS")
    if args.max_trials is not None:
        cfg.synth_viz_max_trials = args.max_trials
    elif env_max_trials not in {None, ""}:
        cfg.synth_viz_max_trials = int(env_max_trials)

    env_max_timepoints = os.environ.get("SYNTH_VIZ_MAX_TIMEPOINTS")
    if args.max_timepoints is not None:
        cfg.synth_viz_max_timepoints = args.max_timepoints
    elif env_max_timepoints not in {None, ""}:
        cfg.synth_viz_max_timepoints = int(env_max_timepoints)

    print(
        "Participant plots: "
        f"count={getattr(cfg, 'synth_viz_participant_count', 4)} "
        f"mode={getattr(cfg, 'synth_viz_participant_mode', 'top_zeta')}"
    )
    print(
        "Validation plot caps: "
        f"max_trials={getattr(cfg, 'synth_viz_max_trials', 0)} "
        f"max_timepoints={getattr(cfg, 'synth_viz_max_timepoints', 0)}"
    )

    # data_path = args.data or getattr(cfg, "synth_data_path", "rotations.npy")
    data_path = args.data or getattr(cfg, "synth_data_path", "rotations.npy")
    print(f"Loading synthetic data from {data_path} …")
    windows = load_synthetic_windows(cfg, data_path=data_path)
    labels = load_synthetic_labels(cfg, data_path=data_path)
    subjects = load_synthetic_subjects(cfg)
    if getattr(cfg, "synth_noise_std", 0.0) > 0:
        print(f"  Added deterministic Gaussian noise: std={cfg.synth_noise_std}")
    N_in = windows.shape[1]
    print(f"  Windows shape: {windows.shape}  (K, N, T)")

    train_ds, val_ds = train_val_split_synth(
        windows,
        cfg.val_split,
        cfg.seed,
        getattr(cfg, "synth_split", "random"),
        subjects=subjects,
        subject_count=getattr(cfg, "synth_subject_count", 0),
        subject_ids=getattr(cfg, "synth_subject_ids", ""),
        holdout_subject_count=getattr(cfg, "synth_holdout_subject_count", 0),
        holdout_subject_ids=getattr(cfg, "synth_holdout_subject_ids", ""),
    )

    model = MLP(
        in_channels=N_in, d=cfg.d, hidden_dim=cfg.hidden_dim, depth=cfg.depth, dropout=cfg.dropout,
        temporal_filters=getattr(cfg, "temporal_filters", 0),
        temporal_kernel_size=getattr(cfg, "temporal_kernel_size", 31),
        temporal_frontend=getattr(cfg, "temporal_frontend", "symmetric"),
        residual_kernels=getattr(cfg, "residual_kernels", "3,7,15,31"),
        multiscale_symmetric_conv_layers=infer_multiscale_symmetric_conv_layers(
            ckpt["model_state_dict"],
            getattr(cfg, "multiscale_symmetric_conv_layers", 1),
        ),
    )
    model.load_state_dict(ckpt["model_state_dict"])

    make_diagnostic_plots_synth(
        model=model,
        val_ds=val_ds,
        cfg=cfg,
        run_dir=run_dir,
        train_ds=train_ds,
        subjects=subjects,
        labels=labels,
    )


if __name__ == "__main__":
    main()
