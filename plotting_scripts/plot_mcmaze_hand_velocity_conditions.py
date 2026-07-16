from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from data import load_mcmaze_cached, make_windows
from predict_mcmaze_velocity import load_hand_velocity_resampled
from visualize import _get_condition_groups, _hand_windows_from_raw


def _plot_condition_mean_xy(ax, windows, groups, colors, xlabel, ylabel, title):
    for cond_key in groups:
        idx_list = groups[cond_key]
        mean_xy = windows[idx_list].mean(axis=0)
        color = colors[cond_key]
        ax.plot(mean_xy[0], mean_xy[1], lw=0.9, color=color, alpha=0.85)
        ax.scatter(mean_xy[0, 0], mean_xy[1, 0], color=color, s=18, zorder=5)

    # ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot condition-averaged MC Maze hand position trajectories next to "
            "condition-averaged hand velocity trajectories."
        )
    )
    parser.add_argument(
        "--out",
        default="mcmaze/hand_trajectory_condition_averages.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--nwb-path",
        default=None,
        help="Explicit NWB path. Usually set by sourcing mcmaze_config.sh.",
    )
    parser.add_argument(
        "--include-velocity",
        action="store_true",
        help="Also plot condition-averaged hand velocity next to hand position.",
    )
    parser.add_argument(
        "--velocity-scale",
        choices=["stored", "si"],
        default="stored",
        help="'stored' gives mm/s for this NWB; 'si' applies the NWB conversion to m/s.",
    )
    args = parser.parse_args()

    cfg = Config()
    nwb_path = args.nwb_path or cfg.nwb_path
    spikes_raw, bin_width_s, trial_info, time_index_s, hand_pos_raw = load_mcmaze_cached(
        nwb_path, cfg.bin_ms
    )
    del spikes_raw

    hand_pos_windows = _hand_windows_from_raw(
        hand_pos_raw, cfg, trial_info, time_index_s, bin_width_s
    )
    if hand_pos_windows is None:
        raise RuntimeError("This NWB/cache does not contain hand_pos data.")

    groups, colors = _get_condition_groups(trial_info)
    cond_sizes = [len(v) for v in groups.values()]

    if args.include_velocity:
        if not os.path.exists(nwb_path):
            raise FileNotFoundError(
                f"NWB file not found: {nwb_path}\n"
                "Source mcmaze_config.sh first or pass --nwb-path /path/to/file.nwb"
            )
        hand_vel_raw, vel_meta = load_hand_velocity_resampled(
            nwb_path, time_index_s, args.velocity_scale
        )
        hand_vel_windows = make_windows(
            hand_vel_raw,
            trial_info,
            time_index_s,
            bin_width_s,
            strategy=cfg.window_strategy,
            window_size=cfg.window_size,
            align_field=getattr(cfg, "align_field", "move_onset_time"),
            pre_ms=getattr(cfg, "pre_ms", 100),
        )
        velocity_units = "mm/s" if args.velocity_scale == "stored" else str(vel_meta["units"])
        fig, (ax_pos, ax_vel) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax_pos = plt.subplots(1, 1, figsize=(7, 6))

    _plot_condition_mean_xy(
        ax_pos,
        hand_pos_windows,
        groups,
        colors,
        xlabel="hand_x [mm]",
        ylabel="hand_y [mm]",
        title="Condition-averaged hand trajectories",
    )
    if args.include_velocity:
        _plot_condition_mean_xy(
            ax_vel,
            hand_vel_windows,
            groups,
            colors,
            xlabel=f"hand velocity x ({velocity_units})",
            ylabel=f"hand velocity y ({velocity_units})",
            title="Condition-averaged hand velocities",
        )

    # fig.suptitle(
    #     f"MC Maze condition averages ({len(groups)} conditions, {hand_pos_windows.shape[0]} trials)\n"
    #     f"trials/condition: min={min(cond_sizes)}, median={int(np.median(cond_sizes))}, "
    #     f"max={max(cond_sizes)}",
    #     fontsize=11,
    # )
    fig.tight_layout()

    out_path = Path(args.out)
    os.makedirs(out_path.parent, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
