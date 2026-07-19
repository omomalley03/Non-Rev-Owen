"""Plot MC Maze embedding-plane spectral redundancy.

For each 2D embedding plane ``p = (x_p, y_p)``, this computes the average
one-sided power spectrum

    P_p(f) = mean_trials((|FFT(x_p)|^2 + |FFT(y_p)|^2) / 2)

then normalizes each plane's spectrum to unit sum. Plane redundancy is measured
as cosine similarity between these normalized spectral fingerprints. This is a
different diagnostic from the block-CCA regularizer: it asks whether different
planes use the same temporal frequencies, not whether their raw trajectories are
linearly dependent.

Examples
--------
    python plot_mcmaze_plane_spectral_redundancy.py --run mcmaze/runs/...
    python plot_mcmaze_plane_spectral_redundancy.py --checkpoint mcmaze/runs/.../checkpoints/best.pt
    python plot_mcmaze_plane_spectral_redundancy.py --finetuned-model mcmaze/runs/.../outputs/.../finetuned_model.pt
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
from torch.utils.data import DataLoader

from data import gaussian_smooth, load_mcmaze_cached, make_windows, soft_normalize, train_val_split
from model import MLP, infer_multiscale_symmetric_conv_layers
from paths import RUNS_BASE, RUNS_DIR


ROOT = Path(__file__).resolve().parent


def _cfg_get(cfg, name: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _unique_existing(paths: list[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        path = path.resolve()
        if path not in seen and path.is_dir():
            seen.add(path)
            out.append(path)
    return out


def _completed_runs() -> list[Path]:
    roots = _unique_existing(
        [
            Path(RUNS_DIR),
            Path(RUNS_BASE) / "mcmaze" / "runs",
            ROOT / "mcmaze" / "runs",
            ROOT / "runs",
        ]
    )
    completed = []
    for root in roots:
        for run_dir in root.iterdir():
            ckpt = run_dir / "checkpoints" / "best.pt"
            if ckpt.is_file():
                completed.append((ckpt.stat().st_mtime, run_dir))
    return [run_dir for _, run_dir in sorted(completed, reverse=True)]


def resolve_run_dir(arg_run: str | None) -> Path:
    completed = _completed_runs()
    if arg_run is None:
        if not completed:
            raise FileNotFoundError("No completed MC Maze runs with checkpoints/best.pt were found.")
        print("Using most recent run. Available runs:")
        for i, run_dir in enumerate(completed[:10], 1):
            print(f"  [{i}] {run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir}")
        if len(completed) > 10:
            print(f"  ... {len(completed) - 10} more")
        print()
        return completed[0]
    if arg_run.isdigit():
        idx = int(arg_run) - 1
        if idx < 0 or idx >= len(completed):
            raise ValueError(f"--run {arg_run} out of range (1-{len(completed)})")
        return completed[idx]
    return Path(arg_run).expanduser().resolve()


def infer_temporal_frontend(cfg, state_dict: dict) -> str:
    if "temporal_conv.weight" in state_dict:
        return "symmetric"
    if any(k.startswith("temporal_conv.temporal_branches.0.norm.") for k in state_dict):
        return "residual"
    if any(k.startswith("temporal_conv.temporal_branches.0.conv.") for k in state_dict):
        return "multiscale_symmetric"
    return _cfg_get(cfg, "temporal_frontend", "symmetric")


def build_embedder(cfg, state_dict: dict, in_channels: int) -> MLP:
    model = MLP(
        in_channels=in_channels,
        d=int(_cfg_get(cfg, "d")),
        hidden_dim=int(_cfg_get(cfg, "hidden_dim")),
        depth=int(_cfg_get(cfg, "depth")),
        dropout=float(_cfg_get(cfg, "dropout", 0.0)),
        temporal_filters=int(_cfg_get(cfg, "temporal_filters", 0)),
        temporal_kernel_size=int(_cfg_get(cfg, "temporal_kernel_size", 31)),
        temporal_frontend=infer_temporal_frontend(cfg, state_dict),
        residual_kernels=_cfg_get(cfg, "residual_kernels", "3,7,15,31"),
        multiscale_symmetric_conv_layers=infer_multiscale_symmetric_conv_layers(
            state_dict,
            _cfg_get(cfg, "multiscale_symmetric_conv_layers", 1),
        ),
        antisymmetric_planes=_cfg_get(cfg, "antisymmetric_planes", 0),
    )
    model.load_state_dict(state_dict)
    return model


def load_source(args):
    if args.finetuned_model is not None:
        path = Path(args.finetuned_model).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"No finetuned model at {path}")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "cfg": saved["config"],
            "state_dict": saved["embedder_state_dict"],
            "default_out_dir": path.parent / "plane_spectral_redundancy",
            "label": f"finetuned: {path.name}",
        }

    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint).expanduser().resolve()
    else:
        run_dir = resolve_run_dir(args.run)
        ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
    return {
        "cfg": saved["config"],
        "state_dict": saved["model_state_dict"],
        "default_out_dir": run_dir / "outputs" / "plane_spectral_redundancy",
        "label": f"checkpoint: {ckpt_path.name}",
    }


def load_mcmaze_windows(cfg):
    spikes_raw, bin_width_s, trial_info, time_index_s, _ = load_mcmaze_cached(
        _cfg_get(cfg, "nwb_path"), int(_cfg_get(cfg, "bin_ms"))
    )
    sigma_samples = round((float(_cfg_get(cfg, "sigma_ms")) * 1e-3) / bin_width_s)
    X_smooth = gaussian_smooth(spikes_raw, sigma_samples)
    softnorm = _cfg_get(cfg, "softnorm_method", "none")
    if softnorm and softnorm != "none":
        X_smooth = soft_normalize(X_smooth, method=softnorm)

    windows = make_windows(
        X_smooth,
        trial_info,
        time_index_s,
        bin_width_s,
        strategy=_cfg_get(cfg, "window_strategy", "trial_aligned"),
        window_size=int(_cfg_get(cfg, "window_size")),
        align_field=_cfg_get(cfg, "align_field", "move_onset_time"),
        pre_ms=int(_cfg_get(cfg, "pre_ms", 100)),
    )
    if _cfg_get(cfg, "split", "dataset") == "random":
        trial_info = trial_info.drop(columns=["split"], errors="ignore")
    train_ds, val_ds = train_val_split(
        windows,
        trial_info,
        float(_cfg_get(cfg, "val_split")),
        int(_cfg_get(cfg, "seed")),
    )
    return spikes_raw, train_ds, val_ds


def _mean_axes(cfg) -> tuple[int, ...]:
    axes = _cfg_get(cfg, "F_mean_axis", (0, 2))
    if axes is None:
        return ()
    if isinstance(axes, int):
        axes = (axes,)
    return tuple(int(axis) for axis in axes)


def _center_stats(model, loader, axes: tuple[int, ...], device):
    if 1 in axes:
        raise ValueError("F_mean_axis including embedding dim 1 is not supported for this diagnostic.")
    if 0 not in axes:
        return None

    model.eval()
    total = None
    count = 0
    with torch.no_grad():
        for (batch,) in loader:
            F = model(batch.to(device))
            if 2 in axes:
                chunk = F.sum(dim=(0, 2))
                chunk_count = F.shape[0] * F.shape[2]
            else:
                chunk = F.sum(dim=0)
                chunk_count = F.shape[0]
            total = chunk.detach() if total is None else total + chunk.detach()
            count += chunk_count
    return total / max(count, 1)


def _apply_centering(F: torch.Tensor, axes: tuple[int, ...], stats):
    if not axes:
        return F
    if 0 in axes and 2 in axes:
        return F - stats.reshape(1, -1, 1)
    if 0 in axes:
        return F - stats.unsqueeze(0)
    if 2 in axes:
        return F - F.mean(dim=2, keepdim=True)
    return F


def compute_plane_psd(model, dataset, cfg, batch_size: int, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    axes = _mean_axes(cfg)
    stats = _center_stats(model, loader, axes, device)

    model.eval()
    power_sum = None
    n_trials = 0
    T = None
    with torch.no_grad():
        for (batch,) in loader:
            F = model(batch.to(device))
            F = _apply_centering(F, axes, stats)
            B, d, T = F.shape
            if d % 2 != 0:
                raise ValueError(f"Embedding dimension must be even, got d={d}")
            D = d // 2
            planes = F.reshape(B, D, 2, T)
            fft = torch.fft.rfft(planes, dim=-1, norm="ortho")
            power = fft.abs().pow(2).mean(dim=2)  # (B, D, F)
            chunk = power.sum(dim=0).detach().cpu()
            power_sum = chunk if power_sum is None else power_sum + chunk
            n_trials += B
    if power_sum is None or T is None:
        raise ValueError("No embeddings were produced.")
    return (power_sum / max(n_trials, 1)).numpy(), int(T), int(n_trials)


def spectral_metrics(avg_power: np.ndarray, freqs: np.ndarray, include_dc: bool):
    start = 0 if include_dc else 1
    selected_power = avg_power[:, start:].astype(np.float64)
    selected_freqs = freqs[start:].astype(np.float64)
    if selected_power.shape[1] == 0:
        raise ValueError("No non-DC frequency bins are available.")

    total_power = selected_power.sum(axis=1)
    fingerprints = selected_power / np.maximum(total_power[:, None], 1e-12)
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    similarity = (fingerprints @ fingerprints.T) / np.maximum(norms @ norms.T, 1e-12)
    np.fill_diagonal(similarity, 1.0)

    peak_idx = np.argmax(fingerprints, axis=1)
    entropy = -(fingerprints * np.log(np.maximum(fingerprints, 1e-12))).sum(axis=1)
    entropy = entropy / np.log(fingerprints.shape[1]) if fingerprints.shape[1] > 1 else entropy * 0.0
    centroid = (fingerprints * selected_freqs[None, :]).sum(axis=1)
    if len(similarity) > 1:
        mean_redundancy = (similarity.sum(axis=1) - 1.0) / (len(similarity) - 1)
    else:
        mean_redundancy = np.zeros(len(similarity), dtype=np.float64)

    rows = []
    for plane in range(len(fingerprints)):
        rows.append(
            {
                "plane": plane,
                "total_power": float(total_power[plane]),
                "dominant_frequency_hz": float(selected_freqs[peak_idx[plane]]),
                "dominant_fraction": float(fingerprints[plane, peak_idx[plane]]),
                "spectral_centroid_hz": float(centroid[plane]),
                "spectral_entropy": float(entropy[plane]),
                "mean_spectral_redundancy": float(mean_redundancy[plane]),
            }
        )
    return selected_freqs, fingerprints, similarity, rows


def _limit_plot_items(arr, max_items: int):
    if max_items <= 0 or len(arr) <= max_items:
        return arr, np.arange(len(arr))
    idx = np.linspace(0, len(arr) - 1, max_items).round().astype(int)
    return arr[idx], idx


def save_outputs(out_dir: Path, freqs, fingerprints, similarity, rows, title: str, max_plot_planes: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "plane_spectral_fingerprints.npy", fingerprints)
    np.save(out_dir / "plane_spectral_similarity.npy", similarity)
    np.save(out_dir / "plane_spectral_frequencies_hz.npy", freqs)

    metrics_path = out_dir / "plane_spectral_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        fieldnames = [
            "plane",
            "total_power",
            "dominant_frequency_hz",
            "dominant_fraction",
            "spectral_centroid_hz",
            "spectral_entropy",
            "mean_spectral_redundancy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = {key: np.asarray([row[key] for row in rows]) for key in rows[0] if key != "plane"}
    sim_plot, sim_idx = _limit_plot_items(similarity, max_plot_planes)
    if sim_plot.shape[1] != len(sim_idx):
        sim_plot = sim_plot[:, sim_idx]
    fp_plot, fp_idx = _limit_plot_items(fingerprints, max_plot_planes)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    im = ax.imshow(sim_plot, vmin=0, vmax=1, cmap="viridis", aspect="auto", origin="lower")
    ax.set_title("Plane spectral redundancy")
    ax.set_xlabel("plane")
    ax.set_ylabel("plane")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")

    ax = axes[0, 1]
    extent = [float(freqs[0]), float(freqs[-1]), int(fp_idx[0]), int(fp_idx[-1])] if len(fp_idx) else None
    im = ax.imshow(fp_plot, cmap="magma", aspect="auto", origin="lower", extent=extent)
    ax.set_title("Normalized plane spectra")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("plane")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of plane power")

    plane = np.arange(len(rows))
    axes[0, 2].plot(plane, metrics["mean_spectral_redundancy"], linewidth=1.2)
    axes[0, 2].set_title("Mean redundancy to other planes")
    axes[0, 2].set_xlabel("plane")
    axes[0, 2].set_ylabel("mean cosine similarity")

    axes[1, 0].plot(plane, metrics["dominant_frequency_hz"], linewidth=1.2)
    axes[1, 0].set_title("Dominant frequency")
    axes[1, 0].set_xlabel("plane")
    axes[1, 0].set_ylabel("Hz")

    axes[1, 1].plot(plane, metrics["spectral_entropy"], linewidth=1.2)
    axes[1, 1].set_title("Spectral entropy")
    axes[1, 1].set_xlabel("plane")
    axes[1, 1].set_ylabel("0=narrow, 1=broad")

    axes[1, 2].plot(plane, metrics["total_power"], linewidth=1.2)
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_title("Total spectral power")
    axes[1, 2].set_xlabel("plane")
    axes[1, 2].set_ylabel("FFT power")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    png_path = out_dir / "plane_spectral_redundancy.png"
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {png_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved fingerprints: {out_dir / 'plane_spectral_fingerprints.npy'}")
    print(f"Saved similarity: {out_dir / 'plane_spectral_similarity.npy'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None, help="Integer rank, run directory, or omit for newest MC Maze run.")
    parser.add_argument("--checkpoint", default=None, help="Direct path to an embedding checkpoint .pt file.")
    parser.add_argument("--finetuned-model", default=None, help="Path to outputs/.../finetuned_model.pt.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--include-dc", action="store_true", help="Include the DC frequency bin in fingerprints.")
    parser.add_argument("--max-plot-planes", type=int, default=512,
                        help="Max planes shown in image panels; 0 means all. Metrics are always all planes.")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.finetuned_model is not None and (args.run is not None or args.checkpoint is not None):
        parser.error("--finetuned-model cannot be combined with --run or --checkpoint")
    if args.run is not None and args.checkpoint is not None:
        parser.error("--run and --checkpoint are mutually exclusive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    source = load_source(args)
    cfg = source["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spikes_raw, train_ds, val_ds = load_mcmaze_windows(cfg)
    dataset = train_ds if args.split == "train" else val_ds
    model = build_embedder(cfg, source["state_dict"], spikes_raw.shape[0]).to(device)

    print(f"Computing plane spectra on {args.split} split: trials={len(dataset)} device={device}")
    avg_power, T, n_trials = compute_plane_psd(model, dataset, cfg, args.batch_size, device)
    bin_ms = float(_cfg_get(cfg, "bin_ms"))
    freqs = np.fft.rfftfreq(T, d=bin_ms * 1e-3)
    selected_freqs, fingerprints, similarity, rows = spectral_metrics(avg_power, freqs, args.include_dc)

    out_dir = args.out_dir or source["default_out_dir"]
    title = (
        f"MC Maze plane spectral redundancy ({args.split}, n={n_trials}, "
        f"{'with' if args.include_dc else 'no'} DC)\n{source['label']}"
    )
    save_outputs(Path(out_dir), selected_freqs, fingerprints, similarity, rows, title, args.max_plot_planes)


if __name__ == "__main__":
    main()
