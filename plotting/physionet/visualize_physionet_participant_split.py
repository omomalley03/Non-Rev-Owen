"""PhysioNet participant split diagnostics for subject-holdout synth runs.

This script makes:

  - top-4 validation-plane grid for the first 4 validation participants
  - the same top-4 validation-plane grid for the first 4 test participants
  - full-validation regularisation diagnostics
  - FFT of the x coordinate for the top 8 validation planes, averaged over all
    test trials for the first test participant

The subject split is reconstructed from the checkpoint config.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model.py").is_file())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_nonrev")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import Config
from loss import _batch_rms_normalize, _pair_terms_per_plane
from model import MLP, infer_multiscale_symmetric_conv_layers
from plotting.embedding.plot_embedding_plane_timeseries_fft import save_plot as save_timeseries_fft_plot
from synth_data import apply_train_zscore, load_synthetic_labels, load_synthetic_subjects, load_synthetic_windows
from visualize import (
    plot_block_cca_plane_heatmap,
    plot_covariance_heatmap,
    plot_embedding_norm_distribution,
    plot_plane_zeta_bars,
    plot_zeta_sorted_correlation_heatmap,
    write_plane_zeta_ranking,
)
from visualize_synth import (
    _condition_color,
    _condition_name,
    _dataset_source_indices,
    _load_dataset_split_counts,
    train_val_split_synth,
)


DEFAULT_RUN = (
    "physionetmi/synth_runs/"
    "20260810_154600_d128_h256_dep2_bs128_ep75_lr3e-04_lxp0.0_lbt0.0_lcca1.0_sig10.0_s4"
)


def _load_run(run_dir: Path):
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]

    data_path = getattr(cfg, "synth_data_path", "rotations.npy")
    windows = load_synthetic_windows(
        cfg,
        data_path=data_path,
        context_bins=(
            getattr(cfg, "temporal_context_bins", 0)
            if getattr(cfg, "temporal_filters", 0) > 0
            else 0
        ),
    )
    labels = load_synthetic_labels(cfg, data_path=data_path)
    subjects = load_synthetic_subjects(cfg)
    if subjects is None:
        raise ValueError("This diagnostic requires SYNTH_SUBJECTS_PATH.")
    if labels is None:
        raise ValueError("This diagnostic requires SYNTH_LABELS_PATH.")

    dataset_split_counts = (
        _load_dataset_split_counts(data_path, len(windows))
        if str(getattr(cfg, "synth_split", "random")).lower() == "dataset"
        else None
    )
    train_ds, val_ds, test_ds, trainval_subjects, holdout_subjects = train_val_split_synth(
        windows,
        cfg.val_split,
        cfg.seed,
        getattr(cfg, "synth_split", "random"),
        subjects=subjects,
        subject_count=getattr(cfg, "synth_subject_count", 0),
        subject_ids=getattr(cfg, "synth_subject_ids", ""),
        val_subject_count=getattr(cfg, "synth_val_subject_count", 0),
        val_subject_ids=getattr(cfg, "synth_val_subject_ids", ""),
        holdout_subject_count=getattr(cfg, "synth_holdout_subject_count", 0),
        holdout_subject_ids=getattr(cfg, "synth_holdout_subject_ids", ""),
        return_holdout=True,
        dataset_split_counts=dataset_split_counts,
    )
    if test_ds is None:
        raise ValueError("No test/holdout split was produced for this run.")

    apply_train_zscore(windows, _dataset_source_indices(train_ds), getattr(cfg, "synth_normalize", "none"))

    model = MLP(
        in_channels=windows.shape[1],
        d=cfg.d,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        dropout=cfg.dropout,
        temporal_filters=getattr(cfg, "temporal_filters", 0),
        temporal_kernel_size=getattr(cfg, "temporal_kernel_size", 31),
        temporal_frontend=getattr(cfg, "temporal_frontend", "symmetric"),
        residual_kernels=getattr(cfg, "residual_kernels", "3,7,15,31"),
        multiscale_symmetric_conv_layers=infer_multiscale_symmetric_conv_layers(
            ckpt["model_state_dict"],
            getattr(cfg, "multiscale_symmetric_conv_layers", 1),
        ),
        antisymmetric_planes=getattr(cfg, "antisymmetric_planes", 0),
        temporal_context_bins=getattr(cfg, "temporal_context_bins", 0),
    )
    model.load_state_dict(ckpt["model_state_dict"])

    return cfg, model, train_ds, val_ds, test_ds, labels, subjects, trainval_subjects, holdout_subjects


def _center_embedding(F: np.ndarray, cfg: Config) -> np.ndarray:
    axes = getattr(cfg, "F_mean_axis", (0, 2))
    if axes is None or axes == ():
        return F
    if isinstance(axes, int):
        axes = (axes,)
    return F - F.mean(axis=tuple(axes), keepdims=True)


def _embed_split(model, dataset, cfg: Config, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    model = model.to(device).eval()
    with torch.no_grad():
        for (batch,) in loader:
            chunks.append(model(batch.to(device)).detach().cpu().numpy())
    if not chunks:
        raise ValueError("No embeddings were produced.")
    return _center_embedding(np.concatenate(chunks, axis=0), cfg)


def _plane_zeta(F_hat: np.ndarray) -> np.ndarray:
    K, d, T = F_hat.shape
    D = d // 2
    F_norm = _batch_rms_normalize(torch.from_numpy(F_hat))
    minus_per_plane, plus_per_plane = _pair_terms_per_plane(F_norm)
    return (minus_per_plane / (plus_per_plane + 1e-8)).cpu().numpy()


def _top_planes(F_val: np.ndarray, count: int) -> tuple[list[int], np.ndarray]:
    zeta = _plane_zeta(F_val)
    order = sorted(range(len(zeta)), key=lambda p: (-float(zeta[p]), p))
    return order[:count], zeta


def _plane_branch_labels(cfg: Config, plane_indices: list[int]) -> dict[int, str]:
    D = getattr(cfg, "d", 0) // 2
    frontend = str(getattr(cfg, "temporal_frontend", "") or "").lower()
    if frontend not in {"mixed_parity", "mixed_symmetric_antisymmetric", "mixed_sym_anti", "sym_anti"}:
        return {p: "plane" for p in plane_indices}
    odd_count = int(getattr(cfg, "antisymmetric_planes", 0))
    if odd_count < 0:
        odd_count = max(1, D // 2)
    odd_count = max(0, min(D, odd_count))
    even_count = D - odd_count
    return {p: ("even" if p < even_count else "odd") for p in plane_indices}


def _split_arrays(dataset, labels: np.ndarray, subjects: np.ndarray):
    source_idx = _dataset_source_indices(dataset)
    return labels[source_idx], subjects[source_idx]


def _first_participants(split_subjects: np.ndarray, count: int) -> np.ndarray:
    return np.unique(split_subjects)[:count]


def plot_participant_plane_grid(
    F_split: np.ndarray,
    split_labels: np.ndarray,
    split_subjects: np.ndarray,
    participants: np.ndarray,
    plane_indices: list[int],
    plane_zeta: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    if len(participants) == 0 or len(plane_indices) == 0:
        return

    fig, axes = plt.subplots(
        len(plane_indices),
        len(participants),
        figsize=(4.0 * len(participants), 3.25 * len(plane_indices)),
        squeeze=False,
    )
    planes = F_split.reshape(F_split.shape[0], F_split.shape[1] // 2, 2, F_split.shape[2])
    present_labels = np.array(sorted(np.unique(split_labels).tolist()))

    for row, plane_idx in enumerate(plane_indices):
        for col, participant in enumerate(participants):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(f"Participant {int(participant)}", fontsize=15)
            if col == 0:
                ax.set_ylabel(
                    f"Plane {plane_idx}\ndims {2 * plane_idx},{2 * plane_idx + 1}\nval zeta={plane_zeta[plane_idx]:.2f}",
                    fontsize=13,
                )

            base_idx = np.flatnonzero(split_subjects == participant)
            for label in present_labels:
                idx = base_idx[split_labels[base_idx] == label]
                if idx.size == 0:
                    continue
                mean_traj = planes[idx, plane_idx].mean(axis=0)
                color = _condition_color(label)
                ax.plot(mean_traj[0], mean_traj[1], lw=1.35, color=color, alpha=0.72)
                ax.scatter(mean_traj[0, 0], mean_traj[1, 0], color=color, s=18, alpha=0.85, zorder=5)

            ax.axhline(0, color="k", lw=0.35, alpha=0.22)
            ax.axvline(0, color="k", lw=0.35, alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel(f"dim {2 * plane_idx}", fontsize=12)
            ax.tick_params(labelsize=7)
            ax.set_aspect("equal", adjustable="datalim")

    handles = [
        Line2D([0], [0], color=_condition_color(label), lw=2, label=_condition_name(label))
        for label in present_labels
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=min(4, len(handles)),
            fontsize=12,
            frameon=False,
        )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.06 if handles else 0.02, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_first_test_participant_fft(
    F_test: np.ndarray,
    test_subjects: np.ndarray,
    participant: int,
    plane_indices: list[int],
    plane_zeta: np.ndarray,
    plane_branch: dict[int, str],
    sample_rate_hz: float,
    out_path: Path,
) -> None:
    idx = np.flatnonzero(test_subjects == participant)
    if idx.size == 0:
        raise ValueError(f"Participant {participant} has no test trials.")
    traces = F_test[idx].mean(axis=0, keepdims=True)
    save_timeseries_fft_plot(
        F=F_test,
        traces=traces,
        fft_source_indices=[idx],
        trace_labels=[f"participant {int(participant)} mean (n={idx.size})"],
        trace_colors=[None],
        sample_rate_hz=sample_rate_hz,
        plane_indices=plane_indices,
        plane_zeta={p: float(plane_zeta[p]) for p in plane_indices},
        plane_branch=plane_branch,
        coordinates="x",
        include_dc=False,
        title=(
            f"Test participant {int(participant)} x-coordinate timeseries and FFT\n"
            f"top {len(plane_indices)} validation planes; FFT averaged over {idx.size} test trials"
        ),
        out_path=out_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN, help="Run directory.")
    parser.add_argument("--out-subdir", default="physionet_participant_split", help="Subdirectory under run/outputs.")
    parser.add_argument("--participant-count", type=int, default=4)
    parser.add_argument("--grid-plane-count", type=int, default=4)
    parser.add_argument("--fft-plane-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    out_dir = run_dir / "outputs" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, model, train_ds, val_ds, test_ds, labels, subjects, trainval_subjects, holdout_subjects = _load_run(run_dir)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    val_labels, val_subjects = _split_arrays(val_ds, labels, subjects)
    test_labels, test_subjects = _split_arrays(test_ds, labels, subjects)
    val_participants = _first_participants(val_subjects, args.participant_count)
    test_participants = _first_participants(test_subjects, args.participant_count)
    if len(test_participants) == 0:
        raise ValueError("No test participants found.")

    print(f"Validation participants selected: {val_participants.tolist()}")
    print(f"Test participants selected: {test_participants.tolist()}")

    F_val = _embed_split(model, val_ds, cfg, device, args.batch_size)
    F_test = _embed_split(model, test_ds, cfg, device, args.batch_size)

    grid_planes, plane_zeta = _top_planes(F_val, args.grid_plane_count)
    fft_planes, _ = _top_planes(F_val, args.fft_plane_count)
    fft_plane_branch = _plane_branch_labels(cfg, fft_planes)
    print(f"Top grid planes by validation zeta: {grid_planes}")
    print(f"Top FFT planes by validation zeta: {fft_planes}")

    plot_participant_plane_grid(
        F_val,
        val_labels,
        val_subjects,
        val_participants,
        grid_planes,
        plane_zeta,
        "Validation participants: condition-mean trajectories on top validation planes",
        out_dir / "val_first4_participants_top4_planes.png",
    )
    plot_participant_plane_grid(
        F_test,
        test_labels,
        test_subjects,
        test_participants,
        grid_planes,
        plane_zeta,
        "Test participants: condition-mean trajectories on top validation planes",
        out_dir / "test_first4_participants_top4_planes.png",
    )

    write_plane_zeta_ranking(F_val, out_path=out_dir / "06_plane_validation_zeta_ranking.csv", cfg=cfg)
    plot_plane_zeta_bars(F_val, out_path=out_dir / "06_plane_validation_zeta_bars.png", cfg=cfg)
    plot_covariance_heatmap(F_val, out_path=out_dir / "07_covariance_heatmap.png")
    plot_zeta_sorted_correlation_heatmap(F_val, out_path=out_dir / "07_zeta_sorted_correlation_heatmap.png", cfg=cfg)
    plot_block_cca_plane_heatmap(F_val, out_path=out_dir / "07_block_cca_plane_heatmap.png", cfg=cfg)
    plot_embedding_norm_distribution(F_val, out_path=out_dir / "09_embedding_norm_distribution.png")

    sample_rate_hz = float(getattr(cfg, "eeg_fs", 160.0))
    plot_first_test_participant_fft(
        F_test,
        test_subjects,
        int(test_participants[0]),
        fft_planes,
        plane_zeta,
        fft_plane_branch,
        sample_rate_hz,
        out_dir / "test_first_participant_top8_plane_x_fft.png",
    )


if __name__ == "__main__":
    main()
