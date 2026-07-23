"""Plot embedding plane coordinates over time and their FFT spectra.

By default this plots the split-mean x/y trajectories for the first four
embedding planes, plus the one-sided FFT magnitude of those same time series.

Examples
--------
    python plot_embedding_plane_timeseries_fft.py --run mcmaze/runs/...
    python plot_embedding_plane_timeseries_fft.py --run physionetmi/synth_runs/...
    python plot_embedding_plane_timeseries_fft.py --checkpoint mcmaze/runs/.../checkpoints/best.pt
    python plot_embedding_plane_timeseries_fft.py --run 1 --trial-index 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
from visualize import _get_condition_groups


def _kernel_fft_magnitudes(weight: torch.Tensor, sample_rate_hz: float, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    kernel = weight.detach().cpu().float().numpy()
    if kernel.ndim == 1:
        kernel = kernel.reshape(1, 1, -1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
    mag = np.abs(np.fft.rfft(kernel, n=n_fft, axis=-1, norm="ortho")) # zero pad to the right
    mag = mag.reshape(-1, mag.shape[-1])
    return freqs, mag


def temporal_filter_fft_records(
    model,
    sample_rate_hz: float,
    n_fft: int,
    individual_count: int = 0,
):
    temporal_conv = getattr(model, "temporal_conv", None)
    if temporal_conv is None:
        return []

    records = []

    def add_weight(label: str, weight: torch.Tensor, groups: int | None, kernel, layer: int) -> None:
        freqs, mags = _kernel_fft_magnitudes(weight, sample_rate_hz, n_fft)
        if individual_count > 0:
            groups = int(groups or len(mags))
            filters_per_input = max(1, len(mags) // groups)
            for flat_idx, mag in enumerate(mags[:individual_count]):
                input_dim = flat_idx // filters_per_input
                filter_in_input = flat_idx % filters_per_input
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


def select_traces(
    F: np.ndarray,
    trial_index: int | None,
    condition_groups=None,
) -> tuple[np.ndarray, list[str], list, str]:
    if condition_groups:
        traces = []
        labels = []
        colors = []
        total_trials = 0
        for group in condition_groups:
            idx = group["local_indices"]
            if len(idx) == 0:
                continue
            traces.append(F[idx].mean(axis=0))
            labels.append(f"{group['label']} mean (n={len(idx)})")
            colors.append(group["color"])
            total_trials += len(idx)
        if not traces:
            raise ValueError("No trials matched the requested conditions.")
        return (
            np.stack(traces, axis=0),
            labels,
            colors,
            f"{len(traces)} condition means over {total_trials} trials",
        )
    if trial_index is None:
        return (
            F.mean(axis=0)[None, :, :],
            [f"split mean (n={F.shape[0]})"],
            [None],
            f"split mean over {F.shape[0]} trials",
        )
    if trial_index < 0 or trial_index >= F.shape[0]:
        raise ValueError(f"--trial-index must be in 0-{F.shape[0] - 1}, got {trial_index}")
    return F[trial_index][None, :, :], [f"trial {trial_index}"], [None], f"trial {trial_index}"


def save_plot(
    traces: np.ndarray,
    trace_labels: list[str],
    trace_colors: list,
    sample_rate_hz: float,
    n_planes: int,
    include_dc: bool,
    title: str,
    out_path: Path,
) -> None:
    n_traces, d, T = traces.shape
    max_planes = d // 2
    n_planes = min(n_planes, max_planes)
    if n_planes < 1:
        raise ValueError(f"Need at least one complete 2D plane, got d={d}")

    planes = traces[:, : 2 * n_planes].reshape(n_traces, n_planes, 2, T)
    t_axis = np.arange(T) / sample_rate_hz
    freqs = np.fft.rfftfreq(T, d=1.0 / sample_rate_hz) # get frequency axis
    fft_mag = np.abs(np.fft.rfft(planes, axis=-1, norm="ortho")) # compute one-sided real FFT mag
    f_start = 0 if include_dc else 1

    fig, axes = plt.subplots(n_planes, 2, figsize=(12, 2.7 * n_planes), squeeze=False)
    for p in range(n_planes):
        ax_time = axes[p, 0]
        ax_fft = axes[p, 1]
        x_dim = 2 * p
        y_dim = 2 * p + 1

        for trace_idx, label in enumerate(trace_labels):
            color = trace_colors[trace_idx]
            if color is None:
                x_color = "steelblue"
                y_color = "tomato"
                x_label = f"dim {x_dim}"
                y_label = f"dim {y_dim}"
            else:
                x_color = y_color = color
                x_label = label if p == 0 else None
                y_label = None

            ax_time.plot(
                t_axis,
                planes[trace_idx, p, 0],
                color=x_color,
                lw=1.1,
                ls="-",
                alpha=0.9,
                label=x_label,
            )
            ax_time.plot(
                t_axis,
                planes[trace_idx, p, 1],
                color=y_color,
                lw=1.1,
                ls="--",
                alpha=0.9,
                label=y_label,
            )
        ax_time.set_title(f"Plane {p}: coordinates vs time", fontsize=10)
        ax_time.set_xlabel("time (s)", fontsize=9)
        ax_time.set_ylabel("embedding value", fontsize=9)
        ax_time.spines[["top", "right"]].set_visible(False)
        ax_time.tick_params(labelsize=8)
        if color is None or p == 0:
            ax_time.legend(fontsize=8, frameon=False, loc="best")

        for trace_idx, label in enumerate(trace_labels):
            color = trace_colors[trace_idx]
            if color is None:
                x_color = "steelblue"
                y_color = "tomato"
                x_label = f"dim {x_dim}"
                y_label = f"dim {y_dim}"
            else:
                x_color = y_color = color
                x_label = label if p == 0 else None
                y_label = None
            ax_fft.plot(
                freqs[f_start:],
                fft_mag[trace_idx, p, 0, f_start:],
                color=x_color,
                lw=1.1,
                ls="-",
                alpha=0.9,
                label=x_label,
            )
            ax_fft.plot(
                freqs[f_start:],
                fft_mag[trace_idx, p, 1, f_start:],
                color=y_color,
                lw=1.1,
                ls="--",
                alpha=0.9,
                label=y_label,
            )
        ax_fft.set_title(f"Plane {p}: FFT magnitude", fontsize=10)
        ax_fft.set_xlabel("frequency (Hz)", fontsize=9)
        ax_fft.set_ylabel("|FFT|", fontsize=9)
        ax_fft.spines[["top", "right"]].set_visible(False)
        ax_fft.tick_params(labelsize=8)
        if color is None or p == 0:
            ax_fft.legend(fontsize=8, frameon=False, loc="best")

    if any(color is not None for color in trace_colors):
        style_handles = [
            plt.Line2D([0], [0], color="0.25", lw=1.2, ls="-", label="x / even dim"),
            plt.Line2D([0], [0], color="0.25", lw=1.2, ls="--", label="y / odd dim"),
        ]
        axes[0, 0].legend(
            handles=axes[0, 0].get_legend_handles_labels()[0] + style_handles,
            labels=axes[0, 0].get_legend_handles_labels()[1] + ["x / even dim", "y / odd dim"],
            fontsize=7,
            frameon=False,
            loc="best",
        )

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        out_path.with_suffix(".npz"),
        traces=traces,
        trace_labels=np.asarray(trace_labels, dtype=object),
        plotted_planes=planes,
        time_s=t_axis,
        frequency_hz=freqs,
        fft_magnitude=fft_mag,
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
        filter_frequency_hz=records[0]["frequency_hz"],
        filter_fft_magnitude=np.stack([record["magnitude"] for record in records], axis=0),
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
    y_max = max(float(np.max(record["magnitude"][f_start:])) for record in records)
    y_lim = (0.0, y_max * 1.05 if y_max > 0 else 1.0)
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
                record["magnitude"][f_start:],
                lw=1.0,
                alpha=0.85,
                label=label,
            )
        ax.set_title(branch_label, fontsize=10)
        ax.set_ylabel("|FFT|", fontsize=8)
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
    y_max = max(float(np.max(record["magnitude"][f_start:])) for record in individual_records)
    y_lim = (0.0, y_max * 1.05 if y_max > 0 else 1.0)
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
                record["magnitude"][f_start:],
                lw=1.0,
                alpha=0.85,
                label=record["branch_label"],
            )
        ax.set_title(f"input dimension {input_dim}", fontsize=10)
        ax.set_ylabel("|FFT|", fontsize=8)
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
    parser.add_argument("--n-planes", type=int, default=4)
    parser.add_argument(
        "--trial-index",
        type=int,
        default=None,
        help="Plot one local trial from the selected split. Default: plot the split-mean trace.",
    )
    parser.add_argument(
        "--conditions",
        default="",
        help=(
            "Comma-separated sorted MC Maze condition indices. Each condition is "
            "averaged separately and plotted as its own trace. Default: one average "
            "over all trials in the selected split."
        ),
    )
    parser.add_argument("--include-dc", action="store_true", help="Include the DC bin in the FFT panels.")
    parser.add_argument(
        "--filter-individual-count",
        type=int,
        default=4,
        help=(
            "Plot the first N individual temporal filters from each filter-size branch. "
            "Default 4. Use 0 to plot one mean spectrum per branch."
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
    if args.filter_individual_count < 0:
        parser.error("--filter-individual-count must be non-negative")
    if args.trial_index is not None and args.conditions:
        parser.error("--trial-index and --conditions are mutually exclusive")

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

    condition_indices = parse_condition_indices(args.conditions)
    condition_groups = None
    if condition_indices:
        if dataset_label != "MC Maze":
            raise ValueError("--conditions is currently supported for MC Maze runs only.")
        condition_groups = mcmaze_condition_groups(cfg, dataset, condition_indices)
        counts = [len(group["local_indices"]) for group in condition_groups]
        print(f"Averaging condition indices separately: {list(zip(condition_indices, counts))}")
    traces, trace_labels, trace_colors, trace_label = select_traces(
        F,
        args.trial_index,
        condition_groups=condition_groups,
    )
    filter_records = temporal_filter_fft_records(
        model,
        sample_rate_hz,
        n_fft=traces.shape[-1],
        individual_count=args.filter_individual_count,
    )
    if filter_records:
        print(f"Plotting temporal filter FFT records: {len(filter_records)}")
    else:
        print("No temporal filters found on this model.")

    default_out_dir = Path(source["default_out_dir"]).parent / "plane_timeseries_fft"
    out_path = args.out or default_out_dir / "first4_plane_timeseries_fft.png"
    title = (
        f"{dataset_label} first {min(args.n_planes, traces.shape[1] // 2)} planes: "
        f"{trace_label}, {args.split} split\n{source['label']}"
    )
    title = ""
    save_plot(
        traces,
        trace_labels,
        trace_colors,
        sample_rate_hz,
        args.n_planes,
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
