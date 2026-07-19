from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import gaussian_smooth, load_mcmaze_cached, make_windows, soft_normalize, train_val_split
from loss import S_ratio as compute_S_ratio, _batch_rms_normalize
from model import MLP, infer_multiscale_symmetric_conv_layers
from visualize import _get_condition_groups, _plot_planes_time_coded


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    ROOT
    / "mcmaze/runs/20260706_084343_d256_h64_dep1_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca1.0_sig10.0_s1/checkpoints/best.pt"
)
DEFAULT_OUT = ROOT / "mcmaze/pretrained_first8_planes_condition_hsv.png"


def _cfg_get(cfg, name: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def load_pretrained_embeddings(checkpoint_path: Path):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = saved["config"]

    spikes_raw, bin_width_s, trial_info, time_index_s, _ = load_mcmaze_cached(
        _cfg_get(cfg, "nwb_path"), _cfg_get(cfg, "bin_ms")
    )
    sigma_samples = round((_cfg_get(cfg, "sigma_ms") * 1e-3) / bin_width_s)
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
        pre_ms=_cfg_get(cfg, "pre_ms", 100),
    )
    if _cfg_get(cfg, "split", "dataset") == "random":
        trial_info = trial_info.drop(columns=["split"], errors="ignore")

    _, val_ds = train_val_split(
        windows,
        trial_info,
        _cfg_get(cfg, "val_split"),
        _cfg_get(cfg, "seed"),
    )
    val_info = trial_info.iloc[list(val_ds.indices)].reset_index(drop=True)
    groups, colors = _get_condition_groups(val_info)

    model = MLP(
        in_channels=spikes_raw.shape[0],
        d=_cfg_get(cfg, "d"),
        hidden_dim=_cfg_get(cfg, "hidden_dim"),
        depth=_cfg_get(cfg, "depth"),
        dropout=_cfg_get(cfg, "dropout"),
        temporal_filters=_cfg_get(cfg, "temporal_filters", 0),
        temporal_kernel_size=_cfg_get(cfg, "temporal_kernel_size", 31),
        temporal_frontend=_cfg_get(cfg, "temporal_frontend", "symmetric"),
        residual_kernels=_cfg_get(cfg, "residual_kernels", "3,7,15,31"),
        multiscale_symmetric_conv_layers=infer_multiscale_symmetric_conv_layers(
            saved["model_state_dict"],
            _cfg_get(cfg, "multiscale_symmetric_conv_layers", 1),
        ),
        antisymmetric_planes=_cfg_get(cfg, "antisymmetric_planes", 0),
    )
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    (val_tensor,) = next(iter(DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)))
    with torch.no_grad():
        F = model(val_tensor)
        mean_axis = _cfg_get(cfg, "F_mean_axis", (0, 2))
        if mean_axis:
            F = F - F.mean(dim=mean_axis, keepdim=True)
        zeta = compute_S_ratio(_batch_rms_normalize(F)).item()

    return F.cpu().numpy(), groups, colors, zeta


def plot_first_planes(F_hat, groups, colors, out_path: Path, n_planes: int = 8):
    _, d, _ = F_hat.shape
    max_planes = d // 2
    n_planes = min(n_planes, max_planes)
    planes = F_hat.reshape(F_hat.shape[0], max_planes, 2, F_hat.shape[2])

    ncols = 4
    nrows = int(np.ceil(n_planes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows), squeeze=False)

    for p in range(n_planes):
        ax = axes[p // ncols, p % ncols]
        for cond_key, idx_list in groups.items():
            mean_traj = planes[idx_list, p].mean(axis=0)
            ax.plot(mean_traj[0], mean_traj[1], lw=1.4, color=colors[cond_key], alpha=0.9)
            ax.scatter(mean_traj[0, 0], mean_traj[1, 0], color=colors[cond_key], s=18, zorder=5)

        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"Plane {p}", fontsize=12)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="datalim")

    for p in range(n_planes, nrows * ncols):
        axes[p // ncols, p % ncols].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def first_n_planes_embedding(F_hat: np.ndarray, n_planes: int) -> np.ndarray:
    """Return an embedding containing only the first n 2D planes."""
    _, d, _ = F_hat.shape
    n_planes = min(int(n_planes), d // 2)
    return F_hat[:, : 2 * n_planes, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_arg",
        type=Path,
        nargs="?",
        help="Optional checkpoint path. Equivalent to --checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        "--model",
        dest="checkpoint_flag",
        type=Path,
        default=None,
        help="Embedding checkpoint path. Defaults to the built-in pretrained checkpoint.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--time-coded-out",
        type=Path,
        default=None,
        help="Optional output path for the first-N-plane 02-style time-coded embedding plot.",
    )
    parser.add_argument("--n-planes", type=int, default=8)
    args = parser.parse_args()

    if args.checkpoint_arg is not None and args.checkpoint_flag is not None:
        parser.error("pass the checkpoint either positionally or with --checkpoint/--model, not both")

    checkpoint = args.checkpoint_flag or args.checkpoint_arg or DEFAULT_CHECKPOINT
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint}")

    F_hat, groups, colors, zeta = load_pretrained_embeddings(checkpoint)
    plot_first_planes(F_hat, groups, colors, args.out, args.n_planes)
    time_coded_out = args.time_coded_out
    if time_coded_out is None:
        time_coded_out = args.out.with_name(f"{args.out.stem}_02_time_coded{args.out.suffix}")
    time_coded_out.parent.mkdir(parents=True, exist_ok=True)
    _plot_planes_time_coded(
        first_n_planes_embedding(F_hat, args.n_planes),
        groups,
        zeta,
        out_path=str(time_coded_out),
    )
    print(args.out)
    print(time_coded_out)
    print(f"zeta={zeta:.6f}")


if __name__ == "__main__":
    main()
