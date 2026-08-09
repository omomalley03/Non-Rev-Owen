# """
# Visualise how window_size and pre_ms affect trial-aligned windowing.

# Produces a grid of plots (rows = window_size, cols = pre_ms) using the same
# parameterisation as main.py: make_windows receives window_size and pre_ms
# directly.

# Usage
# -----
#     python visualize_windowing.py
#     python visualize_windowing.py --pre 0 50 100 200 --ws 40 60 80 120
# """

# import argparse
# import os

# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from config import Config
# from data import load_mcmaze_cached, gaussian_smooth, soft_normalize, make_windows


# def _windows_for(X, trial_info, time_index_s, bin_width_s, cfg,
#                  pre_ms, window_size):
#     return make_windows(
#         X, trial_info, time_index_s, bin_width_s,
#         strategy=cfg.window_strategy, window_size=window_size,
#         align_field=cfg.align_field, pre_ms=pre_ms,
#     )


# def plot_hand_grid(hand_pos_raw, trial_info, time_index_s, bin_width_s,
#                    cfg, pre_values, ws_values, out_path):
#     n_ws = len(ws_values)
#     n_pre = len(pre_values)
#     fig, axes = plt.subplots(n_ws, n_pre, figsize=(4.5 * n_pre, 4 * n_ws),
#                              squeeze=False)

#     for i, ws in enumerate(ws_values):
#         for j, pre in enumerate(pre_values):
#             ax = axes[i, j]
#             hw = _windows_for(hand_pos_raw, trial_info, time_index_s,
#                               bin_width_s, cfg, pre, ws)
#             mean_hand = hw.mean(axis=0)  # (2, T)
#             T = mean_hand.shape[1]
#             pre_bins = int(round(pre / cfg.bin_ms))

#             cmap = plt.get_cmap("coolwarm")
#             for t in range(T - 1):
#                 ax.plot(mean_hand[0, t:t+2], mean_hand[1, t:t+2],
#                         color=cmap(t / max(T - 1, 1)), lw=1.2, alpha=0.9)

#             if 0 <= pre_bins < T:
#                 ax.scatter(mean_hand[0, pre_bins], mean_hand[1, pre_bins],
#                            color="k", s=60, zorder=10, marker="x", linewidths=2)
#             ax.scatter(mean_hand[0, 0], mean_hand[1, 0],
#                        color="green", s=30, zorder=9)

#             post_ms = ws * cfg.bin_ms - pre
#             ax.set_title(f"T={ws}  pre={pre}  (post={post_ms})", fontsize=9)
#             ax.set_aspect("equal", adjustable="datalim")
#             ax.tick_params(labelsize=7)
#             ax.spines[["top", "right"]].set_visible(False)

#             if i == n_ws - 1:
#                 ax.set_xlabel("hand_x", fontsize=8)
#             if j == 0:
#                 ax.set_ylabel("hand_y", fontsize=8)

#     fig.suptitle(
#         f"Trial-avg hand trajectory  (rows=window_size, cols=pre_ms)\n"
#         f"align={cfg.align_field}  |  x = alignment event  |  "
#         f"blue=early, red=late",
#         fontsize=11,
#     )
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved -> {out_path}")


# def plot_neural_grid(X_smooth, trial_info, time_index_s, bin_width_s,
#                      cfg, pre_values, ws_values, out_path, n_neurons=100):
#     ch_var = X_smooth.var(axis=1)
#     top_ch = np.argsort(ch_var)[-n_neurons:][::-1]

#     n_ws = len(ws_values)
#     n_pre = len(pre_values)
#     fig, axes = plt.subplots(n_ws, n_pre, figsize=(5 * n_pre, 3 * n_ws),
#                              squeeze=False)

#     for i, ws in enumerate(ws_values):
#         for j, pre in enumerate(pre_values):
#             ax = axes[i, j]
#             wins = _windows_for(X_smooth, trial_info, time_index_s,
#                                 bin_width_s, cfg, pre, ws)
#             mean_pop = wins[:, top_ch, :].mean(axis=0)
#             T = mean_pop.shape[1]

#             t_ms = np.arange(T) * cfg.bin_ms - pre
#             ax.imshow(mean_pop, aspect="auto", cmap="viridis",
#                       extent=[t_ms[0], t_ms[-1], n_neurons - 0.5, -0.5],
#                       interpolation="nearest")
#             ax.axvline(0, color="white", lw=1.2, ls="--", alpha=0.8)

#             post_ms = ws * cfg.bin_ms - pre
#             ax.set_title(f"T={ws}  pre={pre}  (post={post_ms})", fontsize=9)
#             ax.tick_params(labelsize=7)
#             if i == n_ws - 1:
#                 ax.set_xlabel("time from alignment (ms)", fontsize=8)
#             if j == 0:
#                 ax.set_ylabel(f"neuron (top {n_neurons})", fontsize=8)

#     fig.suptitle(
#         f"Trial-avg population activity  (rows=window_size, cols=pre_ms)\n"
#         f"align={cfg.align_field}  |  dashed = alignment  |  "
#         f"top {n_neurons} neurons by variance",
#         fontsize=11,
#     )
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved -> {out_path}")


# def plot_single_trial_examples(X_smooth, trial_info, time_index_s, bin_width_s,
#                                cfg, pre_values, ws_values, out_path, n_trials=5):
#     rng = np.random.default_rng(42)
#     trial_idx = rng.choice(len(trial_info), size=min(n_trials, len(trial_info)),
#                            replace=False)
#     trial_idx.sort()

#     ch_var = X_smooth.var(axis=1)
#     best_ch = int(np.argmax(ch_var))

#     n_trials_show = len(trial_idx)
#     fig, axes = plt.subplots(n_trials_show, 1,
#                              figsize=(12, 2.5 * n_trials_show), squeeze=False)

#     n_combos = len(pre_values) * len(ws_values)
#     colors = plt.cm.tab10(np.linspace(0, 1, min(n_combos, 10)))

#     align_col = (cfg.align_field if cfg.align_field in trial_info.columns
#                  else "start_time")

#     for row, ti in enumerate(trial_idx):
#         ax = axes[row, 0]
#         info = trial_info.iloc[ti]
#         t_start = float(info["start_time"])
#         t_end = float(info.get("end_time", info.get("stop_time", t_start + 2.0)))
#         t_align = float(info[align_col])

#         mask = (time_index_s >= t_start - 0.1) & (time_index_s <= t_end + 0.1)
#         idx = np.where(mask)[0]
#         if len(idx) == 0:
#             continue
#         t_local = (time_index_s[idx] - t_align) * 1000
#         neural = X_smooth[best_ch, idx]

#         ax.plot(t_local, neural, color="k", lw=0.8, alpha=0.6, label="full trial")
#         ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.6)

#         ci = 0
#         for ws in ws_values:
#             for pre in pre_values:
#                 win_start_ms = -pre
#                 win_end_ms = ws * cfg.bin_ms - pre
#                 ax.axvspan(win_start_ms, win_end_ms, alpha=0.08,
#                            color=colors[ci % len(colors)],
#                            label=f"T={ws} pre={pre}")
#                 ax.axvline(win_start_ms, color=colors[ci % len(colors)],
#                            lw=0.7, ls="--", alpha=0.5)
#                 ax.axvline(win_end_ms, color=colors[ci % len(colors)],
#                            lw=0.7, ls="--", alpha=0.5)
#                 ci += 1

#         ax.set_ylabel(f"ch {best_ch}", fontsize=8)
#         ax.tick_params(labelsize=7)
#         ax.spines[["top", "right"]].set_visible(False)
#         if row == 0:
#             ax.legend(fontsize=6, ncol=3, loc="upper right")
#         if row == n_trials_show - 1:
#             ax.set_xlabel("time from alignment (ms)", fontsize=9)
#         ax.set_title(f"trial {ti}", fontsize=8, loc="left")

#     fig.suptitle(
#         f"Window placement on individual trials (ch {best_ch})\n"
#         f"align={cfg.align_field}  |  dotted = alignment event",
#         fontsize=11,
#     )
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved -> {out_path}")


# def main():
#     parser = argparse.ArgumentParser(
#         description=__doc__,
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#     )
#     parser.add_argument("--pre", type=int, nargs="+", default=[0, 50, 100, 200],
#                         help="pre_ms values to sweep (grid columns)")
#     parser.add_argument("--ws", type=int, nargs="+", default=[40, 60, 80, 120],
#                         help="window_size values to sweep (grid rows)")
#     args = parser.parse_args()

#     cfg = Config()
#     pre_values = args.pre
#     ws_values = args.ws

#     print("Loading data...")
#     spikes_raw, bin_width_s, trial_info, time_index_s, hand_pos_raw = \
#         load_mcmaze_cached(cfg.nwb_path, cfg.bin_ms)

#     sigma_samples = round((cfg.sigma_ms * 1e-3) / bin_width_s)
#     X_smooth = gaussian_smooth(spikes_raw, sigma_samples)
#     if cfg.softnorm_method and cfg.softnorm_method != "none":
#         X_smooth = soft_normalize(X_smooth, method=cfg.softnorm_method)

#     out_dir = "windowing_sweep"
#     os.makedirs(out_dir, exist_ok=True)

#     print(f"Sweeping window_size={ws_values}, pre_ms={pre_values}")
#     print(f"  align_field={cfg.align_field}  bin_ms={cfg.bin_ms}")

#     if hand_pos_raw is not None:
#         plot_hand_grid(
#             hand_pos_raw, trial_info, time_index_s, bin_width_s,
#             cfg, pre_values, ws_values,
#             out_path=os.path.join(out_dir, "hand_trajectory_grid.png"),
#         )
#     else:
#         print("Skipping hand grid: no hand_pos available.")

#     plot_neural_grid(
#         X_smooth, trial_info, time_index_s, bin_width_s,
#         cfg, pre_values, ws_values,
#         out_path=os.path.join(out_dir, "neural_heatmap_grid.png"),
#     )

#     plot_single_trial_examples(
#         X_smooth, trial_info, time_index_s, bin_width_s,
#         cfg, pre_values, ws_values,
#         out_path=os.path.join(out_dir, "single_trial_windows.png"),
#     )

#     print(f"\nAll plots saved to {out_dir}/")


# if __name__ == "__main__":
#     main()

"""
Visualise how pre- and post-alignment durations affect trial-aligned windowing.

The hand-velocity grid uses rows = pre_ms and cols = post_ms. The total
window_size passed to make_windows is calculated from pre_ms + post_ms.

Usage
-----
    python visualize_windowing.py
    python visualize_windowing.py --pre 300 200 100 --post 600 700 800
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from config import Config
from data import load_mcmaze_cached, gaussian_smooth, soft_normalize, make_windows
from visualize import _get_condition_groups


def _windows_for(X, trial_info, time_index_s, bin_width_s, cfg,
                 pre_ms, window_size):
    return make_windows(
        X, trial_info, time_index_s, bin_width_s,
        strategy=cfg.window_strategy, window_size=window_size,
        align_field=cfg.align_field, pre_ms=pre_ms,
    )


def _window_size(cfg, pre_ms, post_ms):
    return int(round((pre_ms + post_ms) / cfg.bin_ms))


def plot_hand_velocity_grid(hand_velocity_raw, trial_info, time_index_s,
                            bin_width_s, cfg, pre_values, post_values, out_path):
    n_post = len(post_values)
    n_pre = len(pre_values)
    fig, axes = plt.subplots(n_pre, n_post,
                             figsize=(4.5 * n_post, 4 * n_pre),
                             squeeze=False)
    cmap = plt.get_cmap("coolwarm")
    norm = Normalize(vmin=0, vmax=max(pre_values) + max(post_values))
    all_groups, _ = _get_condition_groups(trial_info)
    condition_indices = [0, 25, 50, 75, 100]
    group_keys = list(all_groups)
    groups = [all_groups[group_keys[idx]] for idx in condition_indices]

    for i, post in enumerate(post_values):
        for j, pre in enumerate(pre_values):
            ax = axes[j, i]
            ws = _window_size(cfg, pre, post)
            velocity_windows = _windows_for(
                hand_velocity_raw, trial_info, time_index_s,
                bin_width_s, cfg, pre, ws,
            )
            T = velocity_windows.shape[2]
            pre_bins = int(round(pre / cfg.bin_ms))
            t_ms = np.arange(T) * cfg.bin_ms

            for idx_list in groups:
                mean_velocity = velocity_windows[idx_list].mean(axis=0)
                points = mean_velocity.T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=cmap, norm=norm,
                                    linewidth=0.9, alpha=0.75)
                lc.set_array(t_ms[:-1])
                ax.add_collection(lc)

            if 0 <= pre_bins < T:
                onset_velocity = np.stack([
                    velocity_windows[idx_list].mean(axis=0)[:, pre_bins]
                    for idx_list in groups
                ])
                # ax.scatter(onset_velocity[:, 0], onset_velocity[:, 1],
                        #    color="k", s=10, zorder=10, marker="x", linewidths=0.8)
            start_velocity = np.stack([
                velocity_windows[idx_list].mean(axis=0)[:, 0]
                for idx_list in groups
            ])
            # ax.scatter(start_velocity[:, 0], start_velocity[:, 1],
            #            color="green", s=8, zorder=9)
            ax.autoscale_view()

            ax.axhline(0, color="k", lw=0.8, ls=":", alpha=0.35)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"pre={pre} ms, post={post} ms", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

            if j == n_pre - 1:
                ax.set_xlabel("hand velocity x (mm/s)", fontsize=8)
            if i == 0:
                ax.set_ylabel("hand velocity y (mm/s)", fontsize=8)

    fig.subplots_adjust(right=0.84)
    fig.suptitle("Trial-averaged hand velocity: conditions 0, 25, 50, 75, 100",
                 fontsize=11)
    cax = fig.add_axes([0.88, 0.18, 0.03, 0.64])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="time in window (ms)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_hand_speed_summaries(hand_velocity_raw, trial_info, time_index_s,
                              cfg, time_out_path, hist_out_path):
    sample_ms = np.arange(0, 1001, 50)
    align_times = trial_info[cfg.align_field].to_numpy(dtype=float)
    end_times = trial_info["end_time"].to_numpy(dtype=float)
    sample_times = align_times[:, None] + sample_ms[None, :] * 1e-3

    sample_idx = np.searchsorted(time_index_s, sample_times)
    valid = sample_idx < len(time_index_s)
    # Keep the velocity derivative away from zero-filled inter-trial gaps.
    valid &= sample_times <= end_times[:, None] - 0.05
    sample_idx = np.clip(sample_idx, 0, len(time_index_s) - 1)

    speed = np.linalg.norm(hand_velocity_raw[:, sample_idx], axis=0)
    speed[~valid] = np.nan

    mean_speed = np.nanmean(speed, axis=0)
    std_speed = np.nanstd(speed, axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(sample_ms, mean_speed, fmt="o-",
                color="tab:blue", lw=1.5, markersize=4, capsize=3,
                # label="mean ± 1 SD"
                )
    ax.axhline(0, color="k", lw=0.8, ls=":", alpha=0.35)
    ax.set_xlabel("time after movement onset (ms)")
    ax.set_ylabel("hand velocity magnitude (mm/s)")
    ax.set_xticks(np.arange(0, 1001, 100))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(time_out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {time_out_path}")

    hist_ms = [600, 700, 800, 900, 1000]
    hist_columns = [int(ms / 50) for ms in hist_ms]
    pooled_speed = speed[:, hist_columns]
    bins = np.linspace(0, np.nanmax(pooled_speed), 50)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(hist_ms)))
    for ms, column, color in zip(hist_ms, hist_columns, colors):
        values = speed[:, column]
        values = values[np.isfinite(values)]
        ax.hist(values, bins=bins, density=True, histtype="step",
                linewidth=1.5, color=color, label=f"{ms} ms")
    ax.set_xlabel("hand velocity magnitude (mm/s)")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(hist_out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {hist_out_path}")


def plot_neural_grid(X_smooth, trial_info, time_index_s, bin_width_s,
                     cfg, pre_values, post_values, out_path, n_neurons=100):
    ch_var = X_smooth.var(axis=1)
    top_ch = np.argsort(ch_var)[-n_neurons:][::-1]

    n_post = len(post_values)
    n_pre = len(pre_values)
    fig, axes = plt.subplots(n_post, n_pre,
                             figsize=(5 * n_pre, 3 * n_post),
                             squeeze=False)

    mean_pops = {}
    for i, post in enumerate(post_values):
        for j, pre in enumerate(pre_values):
            ws = _window_size(cfg, pre, post)
            wins = _windows_for(X_smooth, trial_info, time_index_s,
                                bin_width_s, cfg, pre, ws)
            mean_pops[i, j] = wins[:, top_ch, :].mean(axis=0)
    norm = Normalize(
        vmin=min(mean_pop.min() for mean_pop in mean_pops.values()),
        vmax=max(mean_pop.max() for mean_pop in mean_pops.values()),
    )

    for i, post in enumerate(post_values):
        for j, pre in enumerate(pre_values):
            ax = axes[i, j]
            mean_pop = mean_pops[i, j]
            T = mean_pop.shape[1]

            t_ms = np.arange(T) * cfg.bin_ms - pre
            ax.imshow(mean_pop, aspect="auto", cmap="viridis",
                      norm=norm,
                      extent=[t_ms[0], t_ms[-1], n_neurons - 0.5, -0.5],
                      interpolation="nearest")
            ax.axvline(0, color="white", lw=1.2, ls="--", alpha=0.8)

            ax.tick_params(labelsize=7)
            if i == n_post - 1:
                ax.set_xlabel("time from alignment (ms)", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"neuron (top {n_neurons} by variance)", fontsize=8)

    # fig.suptitle(
    #     f"Trial-avg population activity  (rows=post_ms, cols=pre_ms)\n"
    #     f"align={cfg.align_field}  |  dashed = alignment  |  "
    #     f"top {n_neurons} neurons by variance",
    #     fontsize=11,
    # )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_neural_heatmap_trials(
        X_smooth, trial_info, time_index_s, bin_width_s, cfg, trial_indices,
        out_path, pre_ms=200, post_ms=700, n_neurons=100):
    ch_var = X_smooth.var(axis=1)
    top_ch = np.argsort(ch_var)[-n_neurons:][::-1]
    ws = _window_size(cfg, pre_ms, post_ms)
    trial_windows = _windows_for(
        X_smooth, trial_info.iloc[trial_indices], time_index_s,
        bin_width_s, cfg, pre_ms, ws,
    )[:, top_ch, :]
    mean_activity = trial_windows.mean(axis=0)

    t_ms = np.arange(mean_activity.shape[1]) * cfg.bin_ms - pre_ms
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(
        mean_activity, aspect="auto", cmap="viridis",
        extent=[t_ms[0], t_ms[-1], n_neurons - 0.5, -0.5],
        interpolation="nearest",
    )
    ax.axvline(0, color="white", lw=1.2, ls="--", alpha=0.8)
    ax.set_xlabel("time from alignment (ms)", fontsize=8)
    ax.set_ylabel(f"neuron (top {n_neurons} by variance)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_single_trial_examples(X_smooth, trial_info, time_index_s, bin_width_s,
                               cfg, pre_values, post_values, out_path, n_trials=5):
    rng = np.random.default_rng(42)
    trial_idx = rng.choice(len(trial_info), size=min(n_trials, len(trial_info)),
                           replace=False)
    trial_idx.sort()

    ch_var = X_smooth.var(axis=1)
    best_ch = int(np.argmax(ch_var))

    n_trials_show = len(trial_idx)
    fig, axes = plt.subplots(1, n_trials_show,
                             figsize=(12, 2.5 * n_trials_show), squeeze=False)

    n_combos = len(pre_values) * len(post_values)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_combos, 10)))

    align_col = (cfg.align_field if cfg.align_field in trial_info.columns
                 else "start_time")

    for row, ti in enumerate(trial_idx):
        ax = axes[0, row]
        info = trial_info.iloc[ti]
        t_start = float(info["start_time"])
        t_end = float(info.get("end_time", info.get("stop_time", t_start + 2.0)))
        t_align = float(info[align_col])

        mask = (time_index_s >= t_start - 0.1) & (time_index_s <= t_end + 0.1)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        t_local = (time_index_s[idx] - t_align) * 1000
        neural = X_smooth[best_ch, idx]

        ax.plot(t_local, neural, color="k", lw=0.8, alpha=0.6, label="full trial")
        ax.axhline(0, color="k", lw=0.8, ls=":", alpha=0.35)
        ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.6)

        ci = 0
        for post in post_values:
            for pre in pre_values:
                win_start_ms = -pre
                win_end_ms = post
                ax.axvspan(win_start_ms, win_end_ms, alpha=0.08,
                           color=colors[ci % len(colors)],
                           label=f"pre={pre}, post={post} ms")
                ax.axvline(win_start_ms, color=colors[ci % len(colors)],
                           lw=0.7, ls="--", alpha=0.5)
                ax.axvline(win_end_ms, color=colors[ci % len(colors)],
                           lw=0.7, ls="--", alpha=0.5)
                ci += 1

        ax.set_ylabel(f"ch {best_ch}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.legend(fontsize=6, ncol=3, loc="upper right")
        if row == n_trials_show - 1:
            ax.set_xlabel("time from alignment (ms)", fontsize=9)
        ax.set_title(f"trial {ti}", fontsize=8, loc="left")

    # fig.suptitle(
    #     f"Window placement on individual trials (ch {best_ch})\n"
    #     f"align={cfg.align_field}  |  dotted = alignment event",
    #     fontsize=11,
    # )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pre", type=int, nargs="+", default=[200],
                        help="pre_ms values to sweep")
    parser.add_argument("--post", type=int, nargs="+", default=[600, 700, 800],
                        help="post_ms values to sweep")
    args = parser.parse_args()

    cfg = Config()
    pre_values = args.pre
    post_values = args.post

    print("Loading data...")
    spikes_raw, bin_width_s, trial_info, time_index_s, hand_pos_raw = \
        load_mcmaze_cached(cfg.nwb_path, cfg.bin_ms)

    sigma_samples = round((cfg.sigma_ms * 1e-3) / bin_width_s)
    X_smooth = gaussian_smooth(spikes_raw, sigma_samples)
    if cfg.softnorm_method and cfg.softnorm_method != "none":
        X_smooth = soft_normalize(X_smooth, method=cfg.softnorm_method)

    out_dir = "windowing_sweep"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Sweeping pre_ms={pre_values}, post_ms={post_values}")
    print(f"  align_field={cfg.align_field}  bin_ms={cfg.bin_ms}")

    if hand_pos_raw is not None:
        hand_velocity_raw = np.gradient(hand_pos_raw, bin_width_s, axis=1)
        plot_hand_velocity_grid(
            hand_velocity_raw, trial_info, time_index_s, bin_width_s,
            cfg, pre_values, post_values,
            out_path=os.path.join(out_dir, "hand_velocity_grid.png"),
        )
        plot_hand_speed_summaries(
            hand_velocity_raw, trial_info, time_index_s, cfg,
            time_out_path=os.path.join(
                out_dir, "hand_velocity_magnitude_vs_time.png"
            ),
            hist_out_path=os.path.join(
                out_dir, "hand_velocity_magnitude_histogram.png"
            ),
        )
    else:
        print("Skipping hand velocity grid: no hand_pos available.")

    plot_neural_grid(
        X_smooth, trial_info, time_index_s, bin_width_s,
        cfg, [200], [700],
        out_path=os.path.join(out_dir, "neural_heatmap_grid.png"),
    )
    condition_groups, _ = _get_condition_groups(trial_info)
    condition_zero_trials = next(iter(condition_groups.values()))
    plot_neural_heatmap_trials(
        X_smooth, trial_info, time_index_s, bin_width_s, cfg,
        trial_indices=[condition_zero_trials[0]],
        out_path=os.path.join(
            out_dir, "neural_heatmap_condition_0_first_trial.png"
        ),
    )
    plot_neural_heatmap_trials(
        X_smooth, trial_info, time_index_s, bin_width_s, cfg,
        trial_indices=condition_zero_trials,
        out_path=os.path.join(out_dir, "neural_heatmap_condition_0.png"),
    )

    plot_single_trial_examples(
        X_smooth, trial_info, time_index_s, bin_width_s,
        cfg, pre_values, post_values,
        out_path=os.path.join(out_dir, "single_trial_windows.png"),
    )

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
