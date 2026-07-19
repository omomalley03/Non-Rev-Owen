"""Plot temporal convolution kernels for MC Maze embedding checkpoints.

Examples
--------
    source mcmaze_config.sh
    python plot_mcmaze_conv_kernels.py
    python plot_mcmaze_conv_kernels.py --run 2
    python plot_mcmaze_conv_kernels.py --run mcmaze/runs/20260706_141117_...
    python plot_mcmaze_conv_kernels.py --checkpoint mcmaze/runs/.../checkpoints/best.pt
    python plot_mcmaze_conv_kernels.py --kernel-mode raw
    python plot_mcmaze_conv_kernels.py --kernel-mode symmetric
    python plot_mcmaze_conv_kernels.py --kernel-mode antisymmetric
    python plot_mcmaze_conv_kernels.py --all
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_nonrev")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import MLP, infer_multiscale_symmetric_conv_layers
from paths import RUNS_BASE, RUNS_DIR


KERNEL_MODES = ("auto", "raw", "symmetric", "antisymmetric")
SYMMETRIC_FRONTENDS = {"symmetric", "multiscale_symmetric", "symmetric_multiscale"}
ANTISYMMETRIC_FRONTENDS = {"multiscale_antisymmetric", "antisymmetric_multiscale"}
MULTISCALE_FRONTENDS = SYMMETRIC_FRONTENDS | ANTISYMMETRIC_FRONTENDS


def _unique_existing(paths: list[str]) -> list[str]:
    seen = set()
    out = []
    for path in paths:
        path = os.path.abspath(path)
        if path not in seen and os.path.isdir(path):
            seen.add(path)
            out.append(path)
    return out


def _completed_runs() -> list[str]:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    roots = _unique_existing(
        [
            RUNS_DIR,
            os.path.join(RUNS_BASE, "mcmaze", "runs"),
            os.path.join(repo_root, "mcmaze", "runs"),
            os.path.join(repo_root, "runs"),
        ]
    )
    completed = []
    for root in roots:
        for name in os.listdir(root):
            run_dir = os.path.join(root, name)
            ckpt = os.path.join(run_dir, "checkpoints", "best.pt")
            if os.path.isfile(ckpt):
                completed.append((os.path.getmtime(ckpt), run_dir))
    return [run_dir for _, run_dir in sorted(completed, reverse=True)]


def resolve_run_dir(arg_run: str | None) -> str:
    completed = _completed_runs()
    repo_root = os.path.dirname(os.path.abspath(__file__))

    if arg_run is None:
        if not completed:
            raise FileNotFoundError("No completed MC Maze runs with checkpoints/best.pt were found.")
        print("Using most recent run. Available runs:")
        for i, run_dir in enumerate(completed[:10], 1):
            print(f"  [{i}] {os.path.relpath(run_dir, repo_root)}")
        if len(completed) > 10:
            print(f"  ... {len(completed) - 10} more")
        print()
        return completed[0]

    if arg_run.isdigit():
        idx = int(arg_run) - 1
        if idx < 0 or idx >= len(completed):
            raise ValueError(f"--run {arg_run} out of range (1-{len(completed)})")
        return completed[idx]

    run_dir = os.path.abspath(arg_run)
    if os.path.isfile(run_dir):
        raise ValueError("Use --checkpoint for a direct checkpoint path.")
    return run_dir


def resolve_checkpoint(run: str | None, checkpoint: str | None) -> tuple[str, str | None]:
    if run is not None and checkpoint is not None:
        raise ValueError("Use either --run or --checkpoint, not both.")
    if checkpoint is not None:
        ckpt_path = os.path.abspath(checkpoint)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"No checkpoint at {ckpt_path!r}")
        run_dir = os.path.dirname(os.path.dirname(ckpt_path))
        return ckpt_path, run_dir if os.path.basename(os.path.dirname(ckpt_path)) == "checkpoints" else None

    run_dir = resolve_run_dir(run)
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path!r}")
    return ckpt_path, run_dir


def configured_temporal_frontend(cfg) -> str | None:
    frontend = getattr(cfg, "temporal_frontend", None)
    if not isinstance(frontend, str):
        return None
    frontend = frontend.lower()
    if frontend in MULTISCALE_FRONTENDS or frontend == "residual":
        return frontend
    return None


def infer_temporal_frontend(state_dict: dict, cfg) -> str:
    cfg_frontend = configured_temporal_frontend(cfg)
    if "temporal_conv.weight" in state_dict:
        return "symmetric"
    if any(k.startswith("temporal_conv.temporal_branches.0.norm.") for k in state_dict):
        return "residual"
    if any(k.startswith("temporal_conv.temporal_branches.0.conv.") for k in state_dict):
        return cfg_frontend or "multiscale_symmetric"
    return cfg_frontend or "symmetric"


def resolve_kernel_mode(requested_mode: str, temporal_frontend: str) -> str:
    requested_mode = requested_mode.lower()
    if requested_mode != "auto":
        return requested_mode

    return "effective"


def transform_kernel_weight(weight: torch.Tensor, kernel_mode: str) -> torch.Tensor:
    if kernel_mode == "raw":
        return weight
    if kernel_mode == "symmetric":
        return weight + weight.flip(-1)
    if kernel_mode == "antisymmetric":
        return weight - weight.flip(-1)
    raise ValueError(f"Unknown kernel mode: {kernel_mode!r}")


def first_linear_input_dim(state_dict: dict) -> int:
    candidates = []
    for key, value in state_dict.items():
        if not (key.startswith("net.") and key.endswith(".weight")):
            continue
        if getattr(value, "ndim", None) != 2:
            continue
        parts = key.split(".")
        layer_idx = int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else 10**9
        candidates.append((layer_idx, int(value.shape[1]), key))
    if not candidates:
        raise ValueError("Could not find the first MLP linear layer in the checkpoint state dict.")
    _, in_dim, _ = min(candidates)
    return in_dim


def infer_input_channels(cfg, state_dict: dict) -> int:
    in_dim = first_linear_input_dim(state_dict)
    temporal_filters = int(getattr(cfg, "temporal_filters", 0))
    if temporal_filters <= 0:
        return in_dim
    if in_dim % temporal_filters == 0:
        return in_dim // temporal_filters
    if in_dim % (temporal_filters + 1) == 0:
        return in_dim // (temporal_filters + 1)
    raise ValueError(
        "Could not infer input channel count from checkpoint. "
        f"first_linear_in={in_dim}, temporal_filters={temporal_filters}"
    )


def build_model_for_kernels(cfg, state_dict: dict) -> tuple[MLP, str]:
    temporal_filters = int(getattr(cfg, "temporal_filters", 0))
    if temporal_filters <= 0:
        raise ValueError("This checkpoint has TEMPORAL_FILTERS=0, so there are no conv kernels to plot.")

    temporal_frontend = infer_temporal_frontend(state_dict, cfg)
    model = MLP(
        in_channels=infer_input_channels(cfg, state_dict),
        d=int(getattr(cfg, "d")),
        hidden_dim=int(getattr(cfg, "hidden_dim")),
        depth=int(getattr(cfg, "depth")),
        dropout=float(getattr(cfg, "dropout", 0.0)),
        temporal_filters=temporal_filters,
        temporal_kernel_size=int(getattr(cfg, "temporal_kernel_size", 31)),
        temporal_frontend=temporal_frontend,
        residual_kernels=getattr(cfg, "residual_kernels", "3,7,15,31"),
        multiscale_symmetric_conv_layers=infer_multiscale_symmetric_conv_layers(
            state_dict,
            getattr(cfg, "multiscale_symmetric_conv_layers", 1),
        ),
    )

    current = model.state_dict()
    loaded = []
    for key, value in state_dict.items():
        if not key.startswith("temporal_conv."):
            continue
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            current[key] = value
            loaded.append(key)
    if not loaded:
        raise ValueError("No temporal_conv weights in the checkpoint matched the rebuilt model.")
    model.load_state_dict(current)
    return model, temporal_frontend


def _collect_kernels(temporal_conv, max_panels: int, kernel_mode: str) -> tuple[list[np.ndarray], list[str]]:
    if hasattr(temporal_conv, "temporal_branches"):
        kernels = []
        titles = []
        branches = list(temporal_conv.temporal_branches)
        n_per_branch = max(1, max_panels // max(len(branches), 1))
        for branch_idx, branch in enumerate(branches):
            if kernel_mode == "effective" and hasattr(branch, "effective_weight"):
                layer_weights = [(branch.effective_weight(), f"k{branch.kernel}:l1")]
                if getattr(branch, "conv_layers", 1) == 2:
                    layer_weights.append((branch.effective_weight(2), f"k{branch.kernel}:l2"))
            elif hasattr(branch, "conv"):
                layer_weights = [(transform_kernel_weight(branch.conv.weight, kernel_mode), f"k{branch.kernel}:l1")]
                if getattr(branch, "conv_layers", 1) == 2:
                    layer_weights.append(
                        (transform_kernel_weight(branch.conv2.weight, kernel_mode), f"k{branch.kernel}:l2")
                    )
            else:
                layer_weights = [(transform_kernel_weight(branch.conv.weight, kernel_mode), f"b{branch_idx}")]
            for weights_tensor, branch_label in layer_weights:
                weights = weights_tensor.detach().cpu().numpy().mean(axis=1)
                for dim_idx, weight in enumerate(weights[:n_per_branch]):
                    kernels.append(weight)
                    titles.append(f"{branch_label}:{dim_idx}")
        return kernels[:max_panels], titles[:max_panels]

    weights = temporal_conv.weight.detach().cpu().numpy()[:, 0, :]
    if kernel_mode in {"effective", "symmetric"}:
        weights = weights + np.flip(weights, axis=-1)
    elif kernel_mode == "antisymmetric":
        weights = weights - np.flip(weights, axis=-1)
    elif kernel_mode != "raw":
        raise ValueError(f"Unknown kernel mode: {kernel_mode!r}")
    n_show = min(len(weights), max_panels)
    return [weights[i] for i in range(n_show)], [str(i) for i in range(n_show)]


def plot_conv_kernels(
    model: MLP,
    out_path: str,
    max_panels: int = 64,
    dpi: int = 150,
    ylim: tuple[float, float] | None = None,
    kernel_mode: str = "symmetric",
) -> None:
    if model.temporal_conv is None:
        raise ValueError("Model has no temporal convolution frontend.")

    kernels, titles = _collect_kernels(model.temporal_conv, max_panels=max_panels, kernel_mode=kernel_mode)
    if not kernels:
        raise ValueError("No temporal convolution kernels were found.")

    n_show = len(kernels)
    cols = int(np.ceil(np.sqrt(n_show)))
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(2.0 * cols, 1.5 * rows))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i, j]
            if idx < n_show:
                ax.plot(kernels[idx])
                # ax.set_title(titles[idx], fontsize=6)
                ax.axhline(0, color="0.8", linewidth=0.5)
                if ylim is not None:
                    ax.set_ylim(*ylim)
                ax.locator_params(axis="y", nbins=3)
                ax.tick_params(axis="y", labelsize=6, length=2)
            else:
                ax.set_visible(False)
            ax.set_xticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def fixed_ylim_path(out_path: str, ylim: tuple[float, float]) -> str:
    root, ext = os.path.splitext(out_path)
    lo, hi = ylim
    return f"{root}_ylim_{lo:g}_to_{hi:g}{ext or '.png'}"


def default_out_path(ckpt_path: str, run_dir: str | None, kernel_mode: str, explicit_mode: str) -> str:
    if run_dir is not None:
        out_dir = os.path.join(run_dir, "outputs")
        base = "11_conv_kernels.png" if explicit_mode == "auto" else f"11_conv_kernels_{kernel_mode}.png"
        return os.path.join(out_dir, base)
    base = "conv_kernels.png" if explicit_mode == "auto" else f"conv_kernels_{kernel_mode}.png"
    return os.path.join(os.path.dirname(ckpt_path), base)


def plot_checkpoint(
    ckpt_path: str,
    run_dir: str | None,
    out_path: str | None,
    max_panels: int,
    dpi: int,
    requested_kernel_mode: str,
) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    state_dict = ckpt["model_state_dict"]
    model, temporal_frontend = build_model_for_kernels(cfg, state_dict)
    kernel_mode = resolve_kernel_mode(requested_kernel_mode, temporal_frontend)

    if out_path is None:
        out_path = default_out_path(ckpt_path, run_dir, kernel_mode, requested_kernel_mode)
    print(f"Kernel mode: {requested_kernel_mode} -> {kernel_mode} ({temporal_frontend})")
    plot_conv_kernels(model, out_path, max_panels=max_panels, dpi=dpi, kernel_mode=kernel_mode)
    plot_conv_kernels(
        model,
        fixed_ylim_path(out_path, (-1.0, 1.0)),
        max_panels=max_panels,
        dpi=dpi,
        ylim=(-1.0, 1.0),
        kernel_mode=kernel_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None, help="Integer rank, run directory, or omit for newest MC Maze run.")
    parser.add_argument("--checkpoint", default=None, help="Direct path to a checkpoint .pt file.")
    parser.add_argument("--out", default=None, help="Output PNG path. Defaults to <run>/outputs/11_conv_kernels.png.")
    parser.add_argument("--all", action="store_true", help="Plot kernels for every completed MC Maze run.")
    parser.add_argument("--max-panels", type=int, default=64, help="Maximum number of kernels to show.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--kernel-mode",
        choices=KERNEL_MODES,
        default="auto",
        help=(
            "Which kernel weights to plot. auto rebuilds the configured frontend and plots "
            "the effective kernel used by the current model code."
        ),
    )
    parser.add_argument("--skip-no-conv", action="store_true", help="With --all, skip runs without temporal filters.")
    args = parser.parse_args()

    if args.max_panels < 1:
        parser.error("--max-panels must be positive")
    if args.all and (args.run is not None or args.checkpoint is not None or args.out is not None):
        parser.error("--all cannot be combined with --run, --checkpoint, or --out")

    if args.all:
        runs = _completed_runs()
        if not runs:
            raise FileNotFoundError("No completed MC Maze runs with checkpoints/best.pt were found.")
        for run_dir in runs:
            ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
            try:
                plot_checkpoint(ckpt_path, run_dir, None, args.max_panels, args.dpi, args.kernel_mode)
            except ValueError as exc:
                if args.skip_no_conv and "TEMPORAL_FILTERS=0" in str(exc):
                    print(f"Skipped no-conv run: {run_dir}")
                    continue
                raise
        return

    ckpt_path, run_dir = resolve_checkpoint(args.run, args.checkpoint)
    plot_checkpoint(ckpt_path, run_dir, args.out, args.max_panels, args.dpi, args.kernel_mode)


if __name__ == "__main__":
    main()
