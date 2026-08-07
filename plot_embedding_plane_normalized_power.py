"""Plot normalized embedding-plane frequency power spectra.

This script is meant for comparing which frequency components different
embedding planes use. For each selected 2D plane, it subtracts each trial's
temporal mean, computes one-sided FFT power for each trial and coordinate,
averages power across trials and the two coordinates, then normalizes each
plane spectrum to sum to one.

Examples
--------
    python plot_embedding_plane_normalized_power.py --run mcmaze/runs/...
    python plot_embedding_plane_normalized_power.py --run 1
    python plot_embedding_plane_normalized_power.py --checkpoint mcmaze/runs/.../checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_nonrev")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_embedding_plane_timeseries_fft import collect_embeddings, select_plane_indices
from plot_mcmaze_plane_spectral_redundancy import (
    build_embedder,
    load_source,
    load_windows_for_run,
)
from visualize import _mixed_parity_plane_split, _plane_zeta_values


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


def explicit_plane_metadata(
    F: np.ndarray,
    cfg,
    plane_indices: list[int],
) -> tuple[dict[int, float], dict[int, str]]:
    K, d, T = F.shape
    if d % 2:
        raise ValueError(f"Expected an even embedding dimension, got d={d}")
    D = d // 2
    bad = [p for p in plane_indices if p < 0 or p >= D]
    if bad:
        raise ValueError(f"Plane indices out of range for d={d}: {bad}")

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


def compute_normalized_plane_power(
    F: np.ndarray,
    plane_indices: list[int],
    sample_rate_hz: float,
    include_dc: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frequencies, normalized spectra, and unnormalized plane power."""
    if F.ndim != 3:
        raise ValueError(f"Expected embeddings with shape (trials, dims, time), got {F.shape}")
    K, d, T = F.shape
    if d % 2:
        raise ValueError(f"Expected an even embedding dimension, got d={d}")

    bad = [p for p in plane_indices if p < 0 or (2 * p + 1) >= d]
    if bad:
        raise ValueError(f"Plane indices out of range for d={d}: {bad}")

    planes = np.stack(
        [F[:, 2 * p : 2 * p + 2, :] for p in plane_indices],
        axis=1,
    )  # (trials, selected_planes, 2, time)
    planes = planes - planes.mean(axis=-1, keepdims=True)

    freqs = np.fft.rfftfreq(T, d=1.0 / sample_rate_hz)
    power = np.abs(np.fft.rfft(planes, axis=-1, norm="ortho")) ** 2
    plane_power = power.mean(axis=(0, 2))  # (selected_planes, frequency)

    start = 0 if include_dc else 1
    freqs = freqs[start:]
    plane_power = plane_power[:, start:]
    if plane_power.shape[1] == 0:
        raise ValueError("No frequency bins available after omitting DC.")

    denom = np.maximum(plane_power.sum(axis=1, keepdims=True), 1e-12)
    normalized_power = plane_power / denom
    return freqs, normalized_power, plane_power


def cumulative_frequency(freqs: np.ndarray, normalized_power: np.ndarray, q: float) -> np.ndarray:
    cdf = np.cumsum(normalized_power, axis=1)
    idx = np.argmax(cdf >= q, axis=1)
    return freqs[idx]


def write_metrics(
    out_path: Path,
    plane_indices: list[int],
    plane_zeta: dict[int, float],
    plane_branch: dict[int, str],
    freqs: np.ndarray,
    normalized_power: np.ndarray,
    plane_power: np.ndarray,
) -> None:
    dominant_idx = np.argmax(normalized_power, axis=1)
    centroid = (normalized_power * freqs[None, :]).sum(axis=1)
    f50 = cumulative_frequency(freqs, normalized_power, 0.50)
    f80 = cumulative_frequency(freqs, normalized_power, 0.80)
    f95 = cumulative_frequency(freqs, normalized_power, 0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "plane",
                "branch",
                "zeta",
                "total_power",
                "dominant_frequency_hz",
                "dominant_power_fraction",
                "spectral_centroid_hz",
                "f50_hz",
                "f80_hz",
                "f95_hz",
            ],
        )
        writer.writeheader()
        for rank, plane in enumerate(plane_indices, start=1):
            row = rank - 1
            writer.writerow(
                {
                    "rank": rank,
                    "plane": plane,
                    "branch": plane_branch.get(plane, "plane"),
                    "zeta": plane_zeta.get(plane, np.nan),
                    "total_power": float(plane_power[row].sum()),
                    "dominant_frequency_hz": float(freqs[dominant_idx[row]]),
                    "dominant_power_fraction": float(normalized_power[row, dominant_idx[row]]),
                    "spectral_centroid_hz": float(centroid[row]),
                    "f50_hz": float(f50[row]),
                    "f80_hz": float(f80[row]),
                    "f95_hz": float(f95[row]),
                }
            )


def plot_normalized_power(
    out_path: Path,
    plane_indices: list[int],
    plane_zeta: dict[int, float],
    plane_branch: dict[int, str],
    freqs: np.ndarray,
    normalized_power: np.ndarray,
    max_freq: float | None,
) -> None:
    keep_freq = np.ones_like(freqs, dtype=bool)
    if max_freq is not None:
        keep_freq &= freqs <= max_freq
    plot_freqs = freqs[keep_freq]
    plot_power = normalized_power[:, keep_freq]

    branches = list(dict.fromkeys(plane_branch.get(p, "plane") for p in plane_indices))
    ncols = len(branches)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(6.2 * ncols, 4.4),
        squeeze=False,
        sharey=True,
    )

    cmap = plt.get_cmap("tab10")
    for col, branch in enumerate(branches):
        ax = axes[0, col]
        branch_rows = [
            row
            for row, plane in enumerate(plane_indices)
            if plane_branch.get(plane, "plane") == branch
        ]
        for j, row in enumerate(branch_rows):
            plane = plane_indices[row]
            ax.plot(
                plot_freqs,
                plot_power[row],
                lw=1.5,
                color=cmap(j % 10),
                label=fr"rank {row + 1}, plane {plane}, $\zeta={plane_zeta.get(plane, np.nan):.2f}$",
            )
        ax.set_title(branch, fontsize=11)
        ax.set_xlabel("frequency (Hz)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=7, frameon=False)

    axes[0, 0].set_ylabel("fraction of plane power", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="Integer rank, run directory, or omit for newest run.")
    parser.add_argument("--checkpoint", default=None, help="Direct path to an embedding checkpoint .pt file.")
    parser.add_argument("--finetuned-model", default=None, help="Path to outputs/.../finetuned_model.pt.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--plane-selection",
        choices=["ranked-branch", "first"],
        default="ranked-branch",
        help="Default ranked-branch selects top-zeta even planes, then top-zeta odd planes.",
    )
    parser.add_argument("--planes-per-branch", type=int, default=8)
    parser.add_argument("--n-planes", type=int, default=8)
    parser.add_argument(
        "--planes",
        default="",
        help=(
            "Comma-separated explicit plane indices to plot, e.g. '18,33'. "
            "When set, this overrides --plane-selection, --n-planes, and --planes-per-branch."
        ),
    )
    parser.add_argument("--include-dc", action="store_true", help="Include the DC bin.")
    parser.add_argument("--max-freq", type=float, default=15.0, help="Highest frequency shown in the plot.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    args = parser.parse_args()

    if args.finetuned_model is not None and (args.run is not None or args.checkpoint is not None):
        parser.error("--finetuned-model cannot be combined with --run or --checkpoint")
    if args.run is not None and args.checkpoint is not None:
        parser.error("--run and --checkpoint are mutually exclusive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.planes_per_branch < 1:
        parser.error("--planes-per-branch must be positive")
    if args.n_planes < 1:
        parser.error("--n-planes must be positive")

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

    explicit_planes = parse_plane_indices(args.planes)
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
        "Plotting normalized spectra for planes: "
        + ", ".join(
            f"{p} ({plane_branch.get(p, 'plane')}, zeta={plane_zeta.get(p, np.nan):.3f})"
            for p in plane_indices
        )
    )

    freqs, normalized_power, plane_power = compute_normalized_plane_power(
        F,
        plane_indices,
        sample_rate_hz,
        args.include_dc,
    )

    default_out_dir = Path(source["default_out_dir"]).parent / "plane_frequency_power"
    out_path = args.out or default_out_dir / "normalized_plane_power_spectra.png"
    plot_normalized_power(
        out_path,
        plane_indices,
        plane_zeta,
        plane_branch,
        freqs,
        normalized_power,
        args.max_freq,
    )
    write_metrics(
        out_path.with_name(f"{out_path.stem}_metrics.csv"),
        plane_indices,
        plane_zeta,
        plane_branch,
        freqs,
        normalized_power,
        plane_power,
    )
    np.savez(
        out_path.with_suffix(".npz"),
        plane_indices=np.asarray(plane_indices, dtype=np.int64),
        plane_zeta=np.asarray([plane_zeta.get(p, np.nan) for p in plane_indices], dtype=np.float32),
        plane_branch=np.asarray([plane_branch.get(p, "plane") for p in plane_indices], dtype=object),
        frequency_hz=freqs,
        normalized_power=normalized_power,
        plane_power=plane_power,
        preprocessing="per-trial temporal mean subtraction; no taper",
        power_summary="mean trials and coordinates of |rFFT|^2, normalized per plane",
    )
    print(f"Saved plot: {out_path}")
    print(f"Saved metrics: {out_path.with_name(f'{out_path.stem}_metrics.csv')}")
    print(f"Saved arrays: {out_path.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
