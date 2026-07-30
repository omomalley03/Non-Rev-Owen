"""Save an imshow of one MC Maze neural window with temporal context padding."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from data import gaussian_smooth, load_mcmaze_cached, make_windows, soft_normalize


def main() -> None:
    cfg = Config()
    spikes_raw, bin_width_s, trial_info, time_index_s, _ = load_mcmaze_cached(
        cfg.nwb_path,
        cfg.bin_ms,
    )

    sigma_samples = round((cfg.sigma_ms * 1e-3) / bin_width_s)
    x_smooth = gaussian_smooth(spikes_raw, sigma_samples)
    if cfg.softnorm_method and cfg.softnorm_method != "none":
        x_smooth = soft_normalize(x_smooth, method=cfg.softnorm_method)

    context_bins = int(cfg.temporal_context_bins) if cfg.temporal_filters > 0 else 0
    windows = make_windows(
        x_smooth,
        trial_info,
        time_index_s,
        bin_width_s,
        strategy=cfg.window_strategy,
        window_size=cfg.window_size,
        align_field=cfg.align_field,
        pre_ms=cfg.pre_ms,
        context_bins=context_bins,
    )

    trial_index = 0
    channel_count = 30
    image = windows[trial_index, :channel_count, :]

    out_dir = Path("mcmaze") / "misc_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"mcmaze_trial{trial_index}_channels{channel_count}_"
        f"timesteps{image.shape[1]}_imshow.png"
    )

    time_start = -context_bins
    time_stop = cfg.window_size + context_bins

    fig, ax = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
    im = ax.imshow(
        image,
        aspect="auto",
        cmap="gray_r",
        interpolation="nearest",
        extent=(time_start, time_stop, channel_count - 0.5, -0.5),
    )
    if context_bins > 0:
        ax.axvline(0, color="red", linestyle=":", linewidth=2.4, alpha=0.95)
        ax.axvline(cfg.window_size, color="red", linestyle=":", linewidth=2.4, alpha=0.95)
    ax.set_title(
        f"MC Maze trial {trial_index}: {channel_count} channels, "
        f"{cfg.window_size}+2*{context_bins}={image.shape[1]} bins"
    )
    ax.set_xlabel("time bin relative to window start")
    ax.set_ylabel("channel")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(out_path)
    print(f"window_shape={windows.shape}")
    print(f"plotted_shape={image.shape}")


if __name__ == "__main__":
    main()
