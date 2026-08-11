"""Plot embedding plane coordinates over time and their FFT spectra.

By default this plots condition-averaged x-coordinate trajectories for the same
ranked embedding planes used by visualize.py plots 02 and 04. The FFT panels
plot the mean one-sided FFT magnitude across the individual trials in each
condition, so phase-shifted trial-level frequency content is not canceled by
averaging trajectories before the FFT.

Examples
--------
    python plot_embedding_plane_timeseries_fft.py --run mcmaze/runs/...
    python plot_embedding_plane_timeseries_fft.py --run physionetmi/synth_runs/... --participants 59,44,108,79 --planes 30,19,47,39
    python plot_embedding_plane_timeseries_fft.py --checkpoint mcmaze/runs/.../checkpoints/best.pt
    python plot_embedding_plane_timeseries_fft.py --run 1 --trial-index 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_nonrev")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import load_mcmaze_cached
from plot_mcmaze_plane_spectral_redundancy import (
    _apply_centering,
    _center_stats,
    _cfg_get,
    _mean_axes,
    build_embedder,
    load_source,
    load_windows_for_run,
)
from synth_data import load_synthetic_labels, load_synthetic_subjects
from visualize import (
    _get_condition_groups,
    _mixed_parity_plane_split,
    _plane_indices_for_ranked_branch_plot,
    _plane_zeta_values,
)
from visualize_synth import (
    _condition_color as physionet_condition_color,
    _condition_name as physionet_condition_name,
    _dataset_source_indices,
)


def _kernel_fft_magnitudes(weight: torch.Tensor, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    kernel = weight.detach().cpu().float().numpy()
    if kernel.ndim == 1:
        kernel = kernel.reshape(1, 1, -1)
    n_fft = kernel.shape[-1]
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
    mag = np.abs(np.fft.rfft(kernel, axis=-1, norm="ortho"))
    mag = mag.reshape(-1, mag.shape[-1])
    return freqs, mag


def temporal_filter_fft_records(
    model,
    sample_rate_hz: float,
    individual_count: int = 0,
):
    temporal_conv = getattr(model, "temporal_conv", None)
    if temporal_conv is None:
        return []

    records = []

    def add_weight(label: str, weight: torch.Tensor, groups: int | None, kernel, layer: int) -> None:
        freqs, mags = _kernel_fft_magnitudes(weight, sample_rate_hz)
        if individual_count > 0:
            groups = int(groups or len(mags))
            filters_per_input = max(1, len(mags) // groups)
            grid_channel_count = min(groups, max(individual_count, 8))
            input_dims = np.linspace(0, groups - 1, grid_channel_count, dtype=int)[:individual_count]
            for input_dim in input_dims:
                flat_idx = int(input_dim) * filters_per_input
                filter_in_input = 0
                mag = mags[flat_idx]
                records.append(
                    {
                        "label": f"{label} input {input_dim} filter {filter_in_input}",
                        "branch_label": label,
                        "kernel": kernel,
                        "layer": layer,
                        "input_dim": input_dim,
                        "filter_index": flat_idx,
                        "filter_in_input": filter_in_input,
                        "frequency_hz": freqs,
                        "magnitude": mag,
                    }
                )
        else:
            records.append(
                {
                    "label": f"{label} mean (n={len(mags)})",
                    "branch_label": label,
                    "kernel": kernel,
                    "layer": layer,
                    "input_dim": None,
                    "filter_index": None,
                    "filter_in_input": None,
                    "frequency_hz": freqs,
                    "magnitude": mags.mean(axis=0),
                }
            )

    def add_branch(prefix: str, branch, branch_idx: int) -> None:
        kernel = getattr(branch, "kernel", "?")
        groups = getattr(branch, "groups", None)
        prefix_lower = str(prefix).lower()
        if "symmetric" in prefix_lower and "anti" not in prefix_lower:
            label = f"Symmetric kernel k={kernel}"
        elif "anti" in prefix_lower:
            label = f"Antisymmetric kernel k={kernel}"
        else:
            label = f"{prefix} kernel k={kernel}"
        if hasattr(branch, "effective_weight"):
            add_weight(label, branch.effective_weight(1), groups, kernel, 1)
            if getattr(branch, "conv2", None) is not None:
                add_weight(f"{label} layer 2", branch.effective_weight(2), groups, kernel, 2)
            return
        if hasattr(branch, "conv"):
            add_weight(label, branch.conv.weight, groups, kernel, 1)

    if hasattr(temporal_conv, "sym_conv") and hasattr(temporal_conv, "anti_conv"):
        for i, branch in enumerate(temporal_conv.sym_conv.temporal_branches):
            add_branch("sym", branch, i)
        for i, branch in enumerate(temporal_conv.anti_conv.temporal_branches):
            add_branch("anti", branch, i)
        return records

    if hasattr(temporal_conv, "temporal_branches"):
        prefix = temporal_conv.__class__.__name__.replace("Conv1d", "")
        for i, branch in enumerate(temporal_conv.temporal_branches):
            add_branch(prefix, branch, i)
        return records

    if hasattr(temporal_conv, "weight"):
        weight = temporal_conv.weight
        if hasattr(weight, "flip"):
            weight = weight + weight.flip(-1)
        add_weight(f"{temporal_conv.__class__.__name__} k={weight.shape[-1]}", weight, getattr(temporal_conv, "groups", None), weight.shape[-1], 1)

    return records


def collect_embeddings(model, dataset, cfg, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    axes = _mean_axes(cfg)
    stats = _center_stats(model, loader, axes, device)

    chunks = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            F = model(batch.to(device))
            F = _apply_centering(F, axes, stats)
            chunks.append(F.detach().cpu().numpy())
    if not chunks:
        raise ValueError("No embeddings were produced.")
    return np.concatenate(chunks, axis=0)


def parse_condition_indices(spec: str) -> list[int]:
    spec = str(spec or "").strip()
    if not spec:
        return []
    out = []
    for item in spec.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def _cfg_namespace(cfg):
    return SimpleNamespace(**cfg) if isinstance(cfg, dict) else cfg


def parse_plane_indices(spec: str | None) -> list[int]:
    spec = str(spec or "").strip()
    if not spec:
        return []
    out = []
    for item in spec.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def parse_participant_ids(spec: str | None) -> list[int]:
    return parse_plane_indices(spec)


def mcmaze_condition_groups(cfg, dataset, condition_indices: list[int]):
    _, _, trial_info, _, _ = load_mcmaze_cached(
        _cfg_get(cfg, "nwb_path"), int(_cfg_get(cfg, "bin_ms"))
    )
    if _cfg_get(cfg, "split", "dataset") == "random":
        trial_info = trial_info.drop(columns=["split"], errors="ignore")

    split_source_indices = list(dataset.indices)
    trial_info_split = trial_info.iloc[split_source_indices].reset_index(drop=True)
    groups, colors = _get_condition_groups(trial_info_split)
    keys = list(groups.keys())

    bad = [idx for idx in condition_indices if idx < 0 or idx >= len(keys)]
    if bad:
        raise ValueError(
            f"Condition indices out of range: {bad}. Valid range is 0-{len(keys) - 1}."
        )

    selected = []
    for idx in condition_indices:
        key = keys[idx]
        selected.append(
            {
                "condition_index": idx,
                "condition_key": key,
                "local_indices": np.asarray(groups[key], dtype=int),
                "color": colors[key],
                "label": f"condition {idx}",
            }
        )
    return selected


def physionet_condition_groups(
    cfg,
    dataset,
    condition_labels: list[int],
    participant_ids: list[int],
):
    cfg_obj = _cfg_namespace(cfg)
    labels = load_synthetic_labels(cfg_obj)
    if labels is None:
        raise ValueError("PhysioNet condition plotting requires SYNTH_LABELS_PATH.")
    subjects = load_synthetic_subjects(cfg_obj)

    source_indices = _dataset_source_indices(dataset)
    if np.max(source_indices, initial=-1) >= len(labels):
        raise ValueError(
            f"Loaded labels length ({len(labels)}) does not cover split source indices "
            f"up to {int(np.max(source_indices))}."
        )
    split_labels = labels[source_indices]
    mask = np.ones(len(source_indices), dtype=bool)

    selected_participants = np.asarray(participant_ids, dtype=np.int64)
    split_subjects = None
    if subjects is not None:
        if np.max(source_indices, initial=-1) >= len(subjects):
            raise ValueError(
                f"Loaded subject length ({len(subjects)}) does not cover split source indices "
                f"up to {int(np.max(source_indices))}."
            )
        split_subjects = subjects[source_indices]
    if selected_participants.size:
        if split_subjects is None:
            raise ValueError("--participants requires SYNTH_SUBJECTS_PATH for PhysioNet runs.")
        missing = np.setdiff1d(selected_participants, np.unique(split_subjects))
        if missing.size:
            raise ValueError(
                f"--participants contains IDs not present in the selected {len(dataset)}-trial split: "
                f"{missing.tolist()}"
            )
        mask &= np.isin(split_subjects, selected_participants)
    elif split_subjects is not None:
        selected_participants = np.unique(split_subjects)

    available_labels = np.unique(split_labels[mask])
    selected_labels = np.asarray(condition_labels, dtype=np.int64)
    if selected_labels.size:
        missing = np.setdiff1d(selected_labels, available_labels)
        if missing.size:
            raise ValueError(
                f"--conditions contains labels not present after participant filtering: {missing.tolist()}. "
                f"Available labels: {available_labels.tolist()}"
            )
    else:
        selected_labels = available_labels

    groups = []
    for label in selected_labels:
        idx = np.flatnonzero(mask & (split_labels == label))
        if idx.size == 0:
            continue
        participant_local_indices = []
        if split_subjects is not None:
            present_participants = (
                [p for p in selected_participants if np.any(split_subjects[idx] == p)]
                if selected_participants.size
                else np.unique(split_subjects[idx])
            )
            for participant in present_participants:
                p_idx = np.flatnonzero(
                    mask & (split_labels == label) & (split_subjects == participant)
                )
                if p_idx.size:
                    participant_local_indices.append(p_idx.astype(int, copy=False))
        groups.append(
            {
                "condition_index": int(label),
                "condition_key": int(label),
                "local_indices": idx.astype(int, copy=False),
                "participant_local_indices": participant_local_indices,
                "color": physionet_condition_color(label),
                "label": physionet_condition_name(label),
                "participant_ids": selected_participants.copy(),
            }
        )
    if not groups:
        raise ValueError("No PhysioNet trials matched the requested participants/conditions.")
    return groups


def select_traces(
    F: np.ndarray,
    trial_index: int | None,
    condition_groups=None,
) -> tuple[np.ndarray, list[np.ndarray | list[np.ndarray]], list[str], list, str]:
    if condition_groups:
        traces = []
        fft_source_indices = []
        labels = []
        colors = []
        total_trials = 0
        for group in condition_groups:
            idx = group["local_indices"]
            if len(idx) == 0:
                continue
            participant_groups = [
                np.asarray(part_idx, dtype=int)
                for part_idx in group.get("participant_local_indices", [])
                if len(part_idx) > 0
            ]
            if participant_groups:
                traces.append(
                    np.stack([F[part_idx].mean(axis=0) for part_idx in participant_groups], axis=0).mean(axis=0)
                )
                fft_source_indices.append(participant_groups)
                labels.append(f"{group['label']} n={len(idx)})")
            else:
                traces.append(F[idx].mean(axis=0))
                fft_source_indices.append(idx)
                labels.append(f"{group['label']} (n={len(idx)})")
            colors.append(group["color"])
            total_trials += len(idx)
        if not traces:
            raise ValueError("No trials matched the requested conditions.")
        return (
            np.stack(traces, axis=0),
            fft_source_indices,
            labels,
            colors,
            f"{len(traces)} condition means over {total_trials} trials",
        )
    if trial_index is None:
        return (
            F.mean(axis=0)[None, :, :],
            [np.arange(F.shape[0], dtype=int)],
            [f"split mean (n={F.shape[0]})"],
            [None],
            f"split mean over {F.shape[0]} trials",
        )
    if trial_index < 0 or trial_index >= F.shape[0]:
        raise ValueError(f"--trial-index must be in 0-{F.shape[0] - 1}, got {trial_index}")
    return (
        F[trial_index][None, :, :],
        [np.asarray([trial_index], dtype=int)],
        [f"trial {trial_index}"],
        [None],
        f"trial {trial_index}",
    )


def select_plane_indices(
    F: np.ndarray,
    cfg,
    selection: str,
    n_planes: int,
    planes_per_branch: int,
) -> tuple[list[int], dict[int, float], dict[int, str]]:
    if F.ndim != 3:
        raise ValueError(f"Expected embeddings with shape (trials, dims, time), got {F.shape}")
    K, d, T = F.shape
    if d < 2:
        raise ValueError(f"Need at least one complete 2D plane, got d={d}")
    if d % 2:
        raise ValueError(f"Expected an even embedding dimension made of 2D planes, got d={d}")

    D = d // 2
    planes = F.reshape(K, D, 2, T)
    if selection == "first":
        plane_indices = list(range(min(n_planes, D)))
        plane_zeta = _plane_zeta_values(planes, plane_indices)
        return plane_indices, plane_zeta, {p: "plane" for p in plane_indices}
    return _plane_indices_for_ranked_branch_plot(
        planes,
        d,
        cfg,
        per_branch=planes_per_branch,
    )


def explicit_plane_metadata(
    F: np.ndarray,
    cfg,
    plane_indices: list[int],
) -> tuple[dict[int, float], dict[int, str]]:
    if F.ndim != 3:
        raise ValueError(f"Expected embeddings with shape (trials, dims, time), got {F.shape}")
    K, d, T = F.shape
    if d % 2:
        raise ValueError(f"Expected an even embedding dimension made of 2D planes, got d={d}")
    D = d // 2
    bad = [p for p in plane_indices if p < 0 or p >= D]
    if bad:
        raise ValueError(
            f"Plane indices out of range for d={d}: {bad}. "
            f"Valid range is 0-{D - 1}."
        )

    planes = F.reshape(K, D, 2, T)
    plane_zeta = _plane_zeta_values(planes, plane_indices)

    split = _mixed_parity_plane_split(d, cfg)
    if split is None:
        return plane_zeta, {p: "plane" for p in plane_indices}

    even_planes, odd_planes = split
    plane_branch = {}
    for p in plane_indices:
        if p < even_planes:
            plane_branch[p] = "even"
        elif p < even_planes + odd_planes:
            plane_branch[p] = "odd"
        else:
            plane_branch[p] = "plane"
    return plane_zeta, plane_branch


EEG_BAND_REFERENCE_HZ = (8.0, 13.0, 30.0)


def _add_frequency_reference_lines(ax, freqs: np.ndarray) -> None:
    if freqs.size == 0:
        return
    f_min = float(np.nanmin(freqs))
    f_max = float(np.nanmax(freqs))
    for hz in EEG_BAND_REFERENCE_HZ:
        if f_min <= hz <= f_max:
            ax.axvline(hz, color="0.25", lw=1.2, ls=":", alpha=0.62, zorder=0)


def save_plot(
    F: np.ndarray,
    traces: np.ndarray,
    fft_source_indices: list[np.ndarray | list[np.ndarray]],
    trace_labels: list[str],
    trace_colors: list,
    sample_rate_hz: float,
    plane_indices: list[int],
    plane_zeta: dict[int, float],
    plane_branch: dict[int, str],
    coordinates: str,
    include_dc: bool,
    title: str,
    out_path: Path,
) -> None:
    n_traces, d, T = traces.shape
    max_planes = d // 2
    bad_planes = [p for p in plane_indices if p < 0 or p >= max_planes]
    if bad_planes:
        raise ValueError(
            f"Plane indices out of range for d={d}: {bad_planes}. "
            f"Valid range is 0-{max_planes - 1}."
        )
    n_planes = len(plane_indices)
    if n_planes < 1:
        raise ValueError(f"Need at least one complete 2D plane, got d={d}")

    planes = np.stack(
        [traces[:, 2 * p : 2 * p + 2, :] for p in plane_indices],
        axis=1,
    )
    t_axis = np.arange(T) / sample_rate_hz
    freqs = np.fft.rfftfreq(T, d=1.0 / sample_rate_hz) # get frequency axis

    def source_fft_mag(idx_or_groups: np.ndarray | list[np.ndarray]) -> np.ndarray:
        groups = idx_or_groups if isinstance(idx_or_groups, list) else [idx_or_groups]
        group_mags = []
        for idx in groups:
            idx = np.asarray(idx, dtype=int)
            if idx.size == 0:
                continue
            source_planes = np.stack(
                [F[idx, 2 * p : 2 * p + 2, :] for p in plane_indices],
                axis=1,
            )
            trial_fft_mag = np.abs(
                np.fft.rfft(source_planes, axis=-1, norm="ortho")
            )
            group_mags.append(trial_fft_mag.mean(axis=0))
        if not group_mags:
            raise ValueError("No trajectories were available for FFT averaging.")
        return np.stack(group_mags, axis=0).mean(axis=0)

    fft_mag = []
    for idx in fft_source_indices:
        fft_mag.append(source_fft_mag(idx))
    fft_mag = np.stack(fft_mag, axis=0)
    f_start = 0 if include_dc else 1
    fft_values = np.maximum(fft_mag[:, :, :, f_start:], 1e-12)
    fft_ymin = float(np.min(fft_values))
    fft_ymax = float(np.max(fft_values))
    if fft_ymax <= fft_ymin:
        fft_ylim = (fft_ymin / 1.25, fft_ymax * 1.25)
    else:
        log_min = np.log10(fft_ymin)
        log_max = np.log10(fft_ymax)
        margin = 0.04 * (log_max - log_min)
        fft_ylim = (10 ** (log_min - margin), 10 ** (log_max + margin))
    coord_specs = [(0, "-", "x / dim 2p")]
    if coordinates == "both":
        coord_specs.append((1, ":", "y / dim 2p+1"))
    font_scale = 1 # don't change this from 1
    title_fs = 15 * font_scale
    label_fs = 10.8 * font_scale
    tick_fs = 8 * font_scale
    legend_fs = 7 * font_scale
    suptitle_fs = 12 * font_scale

    n_blocks = 2 if n_planes > 1 else 1
    n_rows = int(np.ceil(n_planes / n_blocks))
    fig, axes = plt.subplots(
        n_rows,
        2 * n_blocks,
        figsize=(12 * n_blocks, 2.7 * n_rows),
        squeeze=False,
    )
    for p in range(n_planes):
        block = p // n_rows
        row = p % n_rows
        ax_time = axes[row, 2 * block]
        ax_fft = axes[row, 2 * block + 1]
        plane_idx = plane_indices[p]
        x_dim = 2 * plane_idx
        y_dim = 2 * plane_idx + 1
        plane_label = f"Plane {plane_idx} (dim {x_dim})"

        for trace_idx, label in enumerate(trace_labels):
            color = trace_colors[trace_idx]
            if color is None:
                coord_colors = {0: "steelblue", 1: "tomato"}
                coord_labels = {0: f"dim {x_dim}", 1: f"dim {y_dim}"}
            else:
                coord_colors = {0: color, 1: color}
                coord_labels = {0: label if p == 0 else None, 1: None}
            for coord_idx, linestyle, _ in coord_specs:
                ax_time.plot(
                    t_axis,
                    planes[trace_idx, p, coord_idx],
                    color=coord_colors[coord_idx],
                    lw=1.1,
                    ls=linestyle,
                    alpha=0.9,
                    label=coord_labels[coord_idx],
                )
        ax_time.set_title(f"{plane_label} time domain", fontsize=title_fs)
        ax_time.set_xlabel("time (s)", fontsize=label_fs)
        ax_time.set_ylabel("embedding value", fontsize=label_fs)
        ax_time.spines[["top", "right"]].set_visible(False)
        ax_time.tick_params(labelsize=tick_fs)
        if not any(color is not None for color in trace_colors) and p == 0:
            ax_time.legend(fontsize=tick_fs, frameon=False, loc="best")

        for trace_idx, label in enumerate(trace_labels):
            color = trace_colors[trace_idx]
            if color is None:
                coord_colors = {0: "steelblue", 1: "tomato"}
                coord_labels = {0: f"dim {x_dim}", 1: f"dim {y_dim}"}
            else:
                coord_colors = {0: color, 1: color}
                coord_labels = {0: label if p == 0 else None, 1: None}
            for coord_idx, linestyle, _ in coord_specs:
                mag = np.maximum(fft_mag[trace_idx, p, coord_idx, f_start:], 1e-12)
                ax_fft.plot(
                    freqs[f_start:],
                    mag,
                    color=coord_colors[coord_idx],
                    lw=1.1,
                    ls=linestyle,
                    alpha=0.9,
                    label=coord_labels[coord_idx],
                )
        ax_fft.set_title(f"{plane_label} freq domain", fontsize=title_fs)
        ax_fft.set_xlabel("frequency (Hz)", fontsize=label_fs)
        ax_fft.set_ylabel("|FFT| (log)", fontsize=label_fs)
        ax_fft.set_yscale("log")
        ax_fft.set_ylim(*fft_ylim)
        _add_frequency_reference_lines(ax_fft, freqs[f_start:])
        ax_fft.spines[["top", "right"]].set_visible(False)
        ax_fft.tick_params(labelsize=tick_fs)
        if not any(color is not None for color in trace_colors) and p == 0:
            ax_fft.legend(fontsize=tick_fs, frameon=False, loc="best")

    for p in range(n_planes, n_rows * n_blocks):
        block = p // n_rows
        row = p % n_rows
        axes[row, 2 * block].set_visible(False)
        axes[row, 2 * block + 1].set_visible(False)

    if any(color is not None for color in trace_colors):
        condition_handles = [
            plt.Line2D([0], [0], color=color, lw=1.8, ls="-", label=label)
            for label, color in zip(trace_labels, trace_colors)
        ]
        if coordinates == "both":
            style_handles = [
                plt.Line2D([0], [0], color="0.25", lw=1.2, ls=ls, label=label)
                for _, ls, label in coord_specs
            ]
            style_legend = axes[0, 0].legend(
                handles=style_handles,
                fontsize=legend_fs,
                frameon=False,
                loc="upper left",
            )
            axes[0, 0].add_artist(style_legend)
            legend_loc = "upper right"
        else:
            legend_loc = "best"
        axes[0, 0].legend(
            handles=condition_handles,
            title="condition",
            fontsize=legend_fs,
            title_fontsize=legend_fs,
            frameon=False,
            loc=legend_loc,
        )

    fig.suptitle(title, fontsize=suptitle_fs)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        out_path.with_suffix(".npz"),
        traces=traces,
        trace_labels=np.asarray(trace_labels, dtype=object),
        fft_source_trial_indices=np.asarray(fft_source_indices, dtype=object),
        plotted_planes=planes,
        plane_indices=np.asarray(plane_indices, dtype=np.int64),
        plane_zeta=np.asarray([plane_zeta.get(p, np.nan) for p in plane_indices], dtype=np.float32),
        plane_branch=np.asarray([plane_branch.get(p, "") for p in plane_indices], dtype=object),
        coordinates=coordinates,
        time_s=t_axis,
        frequency_hz=freqs,
        fft_magnitude=fft_mag,
        fft_magnitude_summary="mean trial FFT magnitude",
    )
    print(f"Saved plot: {out_path}")
    print(f"Saved plotted arrays: {out_path.with_suffix('.npz')}")


def _save_filter_npz(records: list[dict], out_path: Path) -> None:
    if not records:
        np.savez(
            out_path.with_suffix(".npz"),
            filter_labels=np.asarray([], dtype=object),
            filter_branch_labels=np.asarray([], dtype=object),
            filter_kernel=np.asarray([], dtype=object),
            filter_layer=np.asarray([], dtype=object),
            filter_input_dim=np.asarray([], dtype=object),
            filter_index=np.asarray([], dtype=object),
            filter_frequency_hz=np.asarray([], dtype=np.float32),
            filter_fft_magnitude=np.zeros((0, 0), dtype=np.float32),
        )
        return
    np.savez(
        out_path.with_suffix(".npz"),
        filter_labels=np.asarray([record["label"] for record in records], dtype=object),
        filter_branch_labels=np.asarray([record["branch_label"] for record in records], dtype=object),
        filter_kernel=np.asarray([record["kernel"] for record in records], dtype=object),
        filter_layer=np.asarray([record["layer"] for record in records], dtype=object),
        filter_input_dim=np.asarray([record["input_dim"] for record in records], dtype=object),
        filter_index=np.asarray([record["filter_index"] for record in records], dtype=object),
        filter_frequency_hz=np.asarray([record["frequency_hz"] for record in records], dtype=object),
        filter_fft_magnitude=np.asarray([record["magnitude"] for record in records], dtype=object),
    )


def save_filter_by_size_plot(records: list[dict], include_dc: bool, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No temporal filters", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        _save_filter_npz(records, out_path)
        print(f"Saved temporal filter by-size plot: {out_path}")
        return

    f_start = 0 if include_dc else 1
    branch_labels = list(dict.fromkeys(record["branch_label"] for record in records))
    positive_values = np.concatenate([
        np.maximum(record["magnitude"][f_start:], 1e-12)
        for record in records
        if len(record["magnitude"][f_start:]) > 0
    ])
    y_lim = (
        float(np.min(positive_values)) / 1.25,
        float(np.max(positive_values)) * 1.25,
    )
    fig, axes = plt.subplots(
        len(branch_labels),
        1,
        figsize=(8.5, max(2.2, 2.0 * len(branch_labels))),
        squeeze=False,
        sharex=True,
    )
    for row, branch_label in enumerate(branch_labels):
        ax = axes[row, 0]
        branch_records = [record for record in records if record["branch_label"] == branch_label]
        for record in branch_records:
            input_dim = record["input_dim"]
            label = (
                f"input {input_dim}"
                if input_dim is not None
                else record["label"]
            )
            ax.plot(
                record["frequency_hz"][f_start:],
                np.maximum(record["magnitude"][f_start:], 1e-12),
                lw=1.0,
                marker="o",
                markersize=3,
                alpha=0.85,
                label=label,
            )
        ax.set_title(branch_label, fontsize=10)
        ax.set_ylabel("|FFT| (log)", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(*y_lim)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, frameon=False, loc="best")
    axes[-1, 0].set_xlabel("frequency (Hz)", fontsize=9)
    fig.suptitle("Temporal filter FFT magnitudes by filter size", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    _save_filter_npz(records, out_path)
    print(f"Saved temporal filter by-size plot: {out_path}")
    print(f"Saved temporal filter by-size arrays: {out_path.with_suffix('.npz')}")


def save_filter_by_input_plot(records: list[dict], include_dc: bool, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    individual_records = [record for record in records if record["input_dim"] is not None]
    if not individual_records:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(
            0.5,
            0.5,
            "No individual filter records\nuse --filter-individual-count N",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        _save_filter_npz(individual_records, out_path)
        print(f"Saved temporal filter by-input plot: {out_path}")
        return

    f_start = 0 if include_dc else 1
    input_dims = list(dict.fromkeys(record["input_dim"] for record in individual_records))
    positive_values = np.concatenate([
        np.maximum(record["magnitude"][f_start:], 1e-12)
        for record in individual_records
        if len(record["magnitude"][f_start:]) > 0
    ])
    y_lim = (
        float(np.min(positive_values)) / 1.25,
        float(np.max(positive_values)) * 1.25,
    )
    fig, axes = plt.subplots(
        len(input_dims),
        1,
        figsize=(8.5, max(2.2, 2.0 * len(input_dims))),
        squeeze=False,
        sharex=True,
    )
    for row, input_dim in enumerate(input_dims):
        ax = axes[row, 0]
        input_records = [record for record in individual_records if record["input_dim"] == input_dim]
        for record in input_records:
            ax.plot(
                record["frequency_hz"][f_start:],
                np.maximum(record["magnitude"][f_start:], 1e-12),
                lw=1.0,
                marker="o",
                markersize=3,
                alpha=0.85,
                label=record["branch_label"],
            )
        ax.set_title(f"input dimension {input_dim}", fontsize=10)
        ax.set_ylabel("|FFT| (log)", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(*y_lim)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, frameon=False, loc="best")
    axes[-1, 0].set_xlabel("frequency (Hz)", fontsize=9)
    fig.suptitle("Temporal filter FFT magnitudes by input dimension", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    _save_filter_npz(individual_records, out_path)
    print(f"Saved temporal filter by-input plot: {out_path}")
    print(f"Saved temporal filter by-input arrays: {out_path.with_suffix('.npz')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None, help="Integer rank, run directory, or omit for newest run.")
    parser.add_argument("--checkpoint", default=None, help="Direct path to an embedding checkpoint .pt file.")
    parser.add_argument("--finetuned-model", default=None, help="Path to outputs/.../finetuned_model.pt.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--plane-selection",
        choices=["ranked-branch", "first"],
        default="ranked-branch",
        help=(
            "Which embedding planes to plot. Default ranked-branch matches "
            "visualize.py plots 02 and 04: top validation-zeta even planes, "
            "then top validation-zeta odd planes for mixed parity runs."
        ),
    )
    parser.add_argument(
        "--planes-per-branch",
        type=int,
        default=8,
        help="Number of even and odd planes to plot with --plane-selection ranked-branch.",
    )
    parser.add_argument(
        "--n-planes",
        type=int,
        default=4,
        help="Number of first planes to plot with --plane-selection first.",
    )
    parser.add_argument(
        "--planes",
        default="",
        help=(
            "Comma-separated explicit plane indices to plot, e.g. '18,33'. "
            "When set, this overrides --plane-selection, --n-planes, and --planes-per-branch."
        ),
    )
    parser.add_argument(
        "--trial-index",
        type=int,
        default=None,
        help=(
            "Plot one local trial from the selected split. Default: plot condition "
            "means for MC Maze and labelled PhysioNet/synth runs; otherwise plot "
            "the split-mean trace."
        ),
    )
    parser.add_argument(
        "--conditions",
        default=None,
        help=(
            "Comma-separated sorted MC Maze condition indices, or PhysioNet condition "
            "label IDs. Each condition is averaged separately and plotted as its own "
            "trace. Default: MC Maze uses 0,25,50,75,100; PhysioNet uses all labels "
            "present after participant filtering."
        ),
    )
    parser.add_argument(
        "--participants",
        default=None,
        help=(
            "Comma-separated PhysioNet participant IDs to average over. Default: "
            "SYNTH_VIZ_PARTICIPANT_IDS from the environment/config when present, "
            "otherwise all participants in the selected split."
        ),
    )
    parser.add_argument(
        "--coordinates",
        choices=["x", "both"],
        default="x",
        help="Coordinates to plot for each plane. Default x; both uses solid x and dotted y lines.",
    )
    parser.add_argument("--include-dc", action="store_true", help="Include the DC bin in the FFT panels.")
    parser.add_argument(
        "--filter-individual-count",
        type=int,
        default=4,
        help=(
            "Plot the first N linspaced input-channel filters from each filter-size branch, "
            "matching the conv-kernel grid column selection. Default 4. Use 0 to plot one "
            "mean spectrum per branch."
        ),
    )
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    args = parser.parse_args()

    if args.finetuned_model is not None and (args.run is not None or args.checkpoint is not None):
        parser.error("--finetuned-model cannot be combined with --run or --checkpoint")
    if args.run is not None and args.checkpoint is not None:
        parser.error("--run and --checkpoint are mutually exclusive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.n_planes < 1:
        parser.error("--n-planes must be positive")
    if args.planes_per_branch < 1:
        parser.error("--planes-per-branch must be positive")
    if args.filter_individual_count < 0:
        parser.error("--filter-individual-count must be non-negative")
    if args.trial_index is not None and args.conditions:
        parser.error("--trial-index and --conditions are mutually exclusive")
    if args.trial_index is not None and args.participants:
        parser.error("--trial-index and --participants are mutually exclusive")

    source = load_source(args)
    cfg = source["cfg"]
    dataset_label, in_channels, train_ds, val_ds, sample_rate_hz = load_windows_for_run(cfg)
    dataset = train_ds if args.split == "train" else val_ds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_embedder(cfg, source["state_dict"], in_channels).to(device)
    print(
        f"Computing embeddings on {dataset_label} {args.split} split: "
        f"trials={len(dataset)} fs={sample_rate_hz:g}Hz device={device}"
    )
    F = collect_embeddings(model, dataset, cfg, args.batch_size, device)

    participant_spec = (
        args.participants
        if args.participants is not None
        else os.environ.get("SYNTH_VIZ_PARTICIPANT_IDS")
        or str(_cfg_get(cfg, "synth_viz_participant_ids", "") or "")
    )
    participant_ids = parse_participant_ids(participant_spec)

    condition_spec = args.conditions
    if args.trial_index is None and condition_spec is None and dataset_label == "MC Maze":
        condition_spec = "0,25,50,75,100"
    condition_indices = parse_condition_indices(condition_spec)
    condition_groups = None
    if args.trial_index is None and dataset_label == "PhysioNet/synth":
        condition_groups = physionet_condition_groups(
            cfg,
            dataset,
            condition_indices,
            participant_ids,
        )
        counts = [(group["label"], len(group["local_indices"])) for group in condition_groups]
        participant_text = (
            ",".join(str(p) for p in participant_ids) if participant_ids else "all split participants"
        )
        print(f"Averaging PhysioNet conditions separately over participants={participant_text}: {counts}")
    elif condition_indices:
        if dataset_label != "MC Maze":
            raise ValueError("--conditions is supported only for MC Maze and labelled PhysioNet/synth runs.")
        condition_groups = mcmaze_condition_groups(cfg, dataset, condition_indices)
        counts = [len(group["local_indices"]) for group in condition_groups]
        print(f"Averaging condition indices separately: {list(zip(condition_indices, counts))}")
    traces, fft_source_indices, trace_labels, trace_colors, trace_label = select_traces(
        F,
        args.trial_index,
        condition_groups=condition_groups,
    )
    plane_spec = (
        args.planes
        or os.environ.get("SYNTH_VIZ_PLANE_INDICES", "")
        or str(_cfg_get(cfg, "synth_viz_plane_indices", "") or "")
    )
    explicit_planes = parse_plane_indices(plane_spec)
    if explicit_planes:
        plane_indices = explicit_planes
        plane_zeta, plane_branch = explicit_plane_metadata(F, cfg, plane_indices)
    else:
        plane_indices, plane_zeta, plane_branch = select_plane_indices(
            F,
            cfg,
            args.plane_selection,
            args.n_planes,
            args.planes_per_branch,
        )
    print(
        "Plotting embedding planes: "
        + ", ".join(
            f"{p} ({plane_branch.get(p, 'plane')}, zeta={plane_zeta.get(p, np.nan):.3f})"
            for p in plane_indices
        )
    )
    filter_records = temporal_filter_fft_records(
        model,
        sample_rate_hz,
        individual_count=args.filter_individual_count,
    )
    if filter_records:
        print(f"Plotting temporal filter FFT records: {len(filter_records)}")
    else:
        print("No temporal filters found on this model.")

    default_out_dir = Path(source["default_out_dir"]).parent / "plane_timeseries_fft"
    out_path = args.out or default_out_dir / "ranked_branch_plane_timeseries_fft.png"
    title = (
        f"{dataset_label} selected {len(plane_indices)} planes: "
        f"{trace_label}, {args.split} split\n{source['label']}"
    )
    title = ""
    save_plot(
        F,
        traces,
        fft_source_indices,
        trace_labels,
        trace_colors,
        sample_rate_hz,
        plane_indices,
        plane_zeta,
        plane_branch,
        args.coordinates,
        args.include_dc,
        title,
        Path(out_path),
    )
    filter_by_size_out_path = Path(out_path).with_name(
        f"{Path(out_path).stem}_temporal_filters_by_size{Path(out_path).suffix}"
    )
    filter_by_input_out_path = Path(out_path).with_name(
        f"{Path(out_path).stem}_temporal_filters_by_input_dim{Path(out_path).suffix}"
    )
    save_filter_by_size_plot(filter_records, args.include_dc, filter_by_size_out_path)
    save_filter_by_input_plot(filter_records, args.include_dc, filter_by_input_out_path)


if __name__ == "__main__":
    main()
