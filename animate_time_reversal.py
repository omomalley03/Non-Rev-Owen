"""Save forward-time and reversed-time animations for circular motion.

This script intentionally uses Pillow GIF output because it is available in the
project environment, while ffmpeg may not be installed.
"""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/poc_mlp_mplconfig")

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


T = 90
FPS = 20
OUTPUT_DIR = Path("animations")


def circle_data():
    ts = np.linspace(0, 2 * np.pi, T)
    xs = np.cos(ts)
    ys = np.sin(ts)
    return ts, xs, ys


def add_colored_trajectory(ax, xs, ys, ts, cmap, norm):
    points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(ts[:-1])
    lc.set_linewidth(3)
    ax.add_collection(lc)
    return lc


def pi_over_four_ticks():
    tick_positions = np.arange(0, 9) * np.pi / 4
    tick_labels = [
        "0",
        r"$\pi/4$",
        r"$\pi/2$",
        r"$3\pi/4$",
        r"$\pi$",
        r"$5\pi/4$",
        r"$3\pi/2$",
        r"$7\pi/4$",
        r"$2\pi$",
    ]
    return tick_positions, tick_labels


def make_animation(direction, output_path):
    ts, xs, ys = circle_data()
    order = np.arange(T) if direction == 1 else np.arange(T - 1, -1, -1)
    direction_label = "Forward time" if direction == 1 else "Reversed time"

    cmap = plt.get_cmap("coolwarm")
    norm = Normalize(vmin=ts.min(), vmax=ts.max())

    fig, (ax_signals, ax_phase) = plt.subplots(1, 2, figsize=(10, 4.6))
    fig.suptitle(f"Circular motion (pure non-reversibility)", fontsize=14)

    ax_signals.plot(ts, xs, color="steelblue", lw=2, label="x(t) = cos(t)")
    ax_signals.plot(ts, ys, color="tomato", lw=2, label="y(t) = sin(t)")
    ax_signals.set_xlim(ts.min(), ts.max())
    ax_signals.set_ylim(-1.2, 1.2)
    tick_positions, tick_labels = pi_over_four_ticks()
    ax_signals.set_xticks(tick_positions)
    ax_signals.set_xticklabels(tick_labels)
    ax_signals.set_xlabel("time")
    ax_signals.set_ylabel("value")
    ax_signals.grid(alpha=0.25)
    ax_signals.legend(loc="lower left", frameon=False)

    dot_x, = ax_signals.plot([], [], "o", ms=7)
    dot_y, = ax_signals.plot([], [], "o", ms=7)
    cursor = ax_signals.axvline(ts[order[0]], color="black", ls="--", lw=1, alpha=0.65)

    # ax_phase.set_title("Phase portrait")
    ax_phase.set_xlim(-1.2, 1.2)
    ax_phase.set_ylim(-1.2, 1.2)
    ax_phase.set_xlabel("x(t)")
    ax_phase.set_ylabel("y(t)")
    ax_phase.set_aspect("equal", adjustable="box")
    ax_phase.grid(alpha=0.25)
    lc = add_colored_trajectory(ax_phase, xs, ys, ts, cmap, norm)
    dot_phase, = ax_phase.plot([], [], "o", ms=9)

    cbar = fig.colorbar(lc, ax=ax_phase, fraction=0.046, pad=0.04)
    cbar.set_label("time")

    def init():
        dot_x.set_data([], [])
        dot_y.set_data([], [])
        dot_phase.set_data([], [])
        cursor.set_xdata([ts[order[0]], ts[order[0]]])
        return dot_x, dot_y, dot_phase, cursor

    def update(frame):
        i = order[frame]
        color = cmap(norm(ts[i]))

        dot_x.set_data([ts[i]], [xs[i]])
        dot_y.set_data([ts[i]], [ys[i]])
        dot_phase.set_data([xs[i]], [ys[i]])
        dot_x.set_color(color)
        dot_y.set_color(color)
        dot_phase.set_color(color)
        cursor.set_xdata([ts[i], ts[i]])

        return dot_x, dot_y, dot_phase, cursor

    ani = FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        interval=1000 / FPS,
        blit=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"saving {output_path}")
    ani.save(output_path, writer=PillowWriter(fps=FPS))
    plt.close(fig)


def main():
    make_animation(direction=1, output_path=OUTPUT_DIR / "circle_forward_time.gif")
    make_animation(direction=-1, output_path=OUTPUT_DIR / "circle_reversed_time.gif")
    print(f"saved {OUTPUT_DIR / 'circle_forward_time.gif'}")
    print(f"saved {OUTPUT_DIR / 'circle_reversed_time.gif'}")


if __name__ == "__main__":
    main()
