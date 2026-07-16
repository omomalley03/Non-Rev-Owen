from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config
from data import gaussian_smooth, load_mcmaze_cached, make_windows, soft_normalize, train_val_split
from model import MLP, infer_multiscale_symmetric_conv_layers
from visualize import plot_covariance_heatmap


def make_plot07(run_dir: Path) -> Path:
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]

    spikes_raw, bin_width_s, trial_info, time_index_s, _ = load_mcmaze_cached(
        cfg.nwb_path, cfg.bin_ms
    )
    sigma_samples = round((cfg.sigma_ms * 1e-3) / bin_width_s)
    X_smooth = gaussian_smooth(spikes_raw, sigma_samples)
    softnorm = getattr(cfg, "softnorm_method", "none")
    if softnorm and softnorm != "none":
        X_smooth = soft_normalize(X_smooth, method=softnorm)

    windows = make_windows(
        X_smooth,
        trial_info,
        time_index_s,
        bin_width_s,
        strategy=cfg.window_strategy,
        window_size=cfg.window_size,
        align_field=getattr(cfg, "align_field", "move_onset_time"),
        pre_ms=getattr(cfg, "pre_ms", 100),
    )
    if cfg.split == "random":
        trial_info = trial_info.drop(columns=["split"], errors="ignore")
    _, val_ds = train_val_split(windows, trial_info, cfg.val_split, cfg.seed)

    model = MLP(
        in_channels=spikes_raw.shape[0],
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
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.cpu().eval()

    loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    (val_tensor,) = next(iter(loader))
    with torch.no_grad():
        F_hat_t = model(val_tensor)
        F_hat_t = F_hat_t - F_hat_t.mean(dim=cfg.F_mean_axis, keepdim=True)
        F_hat = F_hat_t.numpy()

    out_path = run_dir / "outputs" / "07_covariance_heatmap.png"
    plot_covariance_heatmap(F_hat, out_path=str(out_path))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(make_plot07(args.run))


if __name__ == "__main__":
    main()
