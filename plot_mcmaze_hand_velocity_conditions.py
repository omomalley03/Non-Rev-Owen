from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_nonrev")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from config import Config
from data import load_mcmaze_cached, make_windows
from predict_mcmaze_velocity import load_hand_velocity_resampled
from visualize import _get_condition_groups, _hand_windows_from_raw


def _condition_label(cond_key) -> str:
    if isinstance(cond_key, tuple) and len(cond_key) == 2:
        return f"trial_type={cond_key[0]}, version={cond_key[1]}"
    return f"condition={cond_key}"


def _add_condition_legend(ax, groups, colors, condition_numbers=None):
    condition_numbers = condition_numbers or {}
    handles = [
        Line2D(
            [0],
            [0],
            color=colors[cond_key],
            lw=1.8,
            marker="o",
            markersize=3.5,
            label=f"Condition {condition_numbers.get(cond_key, i)}",#: {_condition_label(cond_key)}",
        )
        for i, cond_key in enumerate(groups)
    ]
    if not handles:
        return None

    n_conditions = len(handles)
    ncols = max(1, int(np.ceil(n_conditions / 32)))
    return ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=ncols,
        fontsize=6,
        frameon=False,
        title="Condition colours",
        title_fontsize=7,
    )


def _parse_condition_indices(spec: str) -> list[int]:
    spec = str(spec or "").strip()
    if not spec:
        return []
    indices = []
    for item in spec.split(","):
        item = item.strip()
        if item:
            indices.append(int(item))
    return indices


def _filter_conditions(groups, colors, indices: list[int]):
    if not indices:
        return groups, colors

    keys = list(groups.keys())
    bad = [idx for idx in indices if idx < 0 or idx >= len(keys)]
    if bad:
        raise ValueError(
            f"Condition indices out of range: {bad}. Valid range is 0-{len(keys) - 1}."
        )

    keep = [keys[idx] for idx in indices]
    return {k: groups[k] for k in keep}, {k: colors[k] for k in keep}


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
    parser.add_argument(
        "--conditions",
        default="",
        help=(
            "Comma-separated condition indices to plot, using the sorted condition order "
            "shown in the legend. Default: plot all conditions."
        ),
    )
    parser.add_argument(
        "--legend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the condition-color legend. Use --no-legend to hide it.",
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
    condition_numbers = {cond_key: i for i, cond_key in enumerate(groups)}
    condition_indices = _parse_condition_indices(args.conditions)
    if condition_indices:
        groups, colors = _filter_conditions(groups, colors, condition_indices)
        print(f"Plotting condition indices: {condition_indices}")
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
    legend = _add_condition_legend(ax_pos, groups, colors, condition_numbers) if args.legend else None
    fig.tight_layout()

    out_path = Path(args.out)
    os.makedirs(out_path.parent, exist_ok=True)
    extra_artists = (legend,) if legend is not None else None
    fig.savefig(out_path, dpi=180, bbox_inches="tight", bbox_extra_artists=extra_artists)
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
