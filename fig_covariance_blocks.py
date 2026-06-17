"""Standalone covariance-block figures with in-panel labels.

This reproduces the covariance, time-transposed covariance, covariance minus
its time transpose, and covariance plus its time transpose from play2.ipynb.
Each figure is saved separately as PNG and SVG in the project root.
"""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/poc_mlp_mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


OUTPUTS = {
    "cov": Path("fig_covariance_blocks_cov"),
    "sigma": Path("fig_covariance_blocks_sigma"),
    "minus": Path("fig_covariance_blocks_sigma_minus"),
    "plus": Path("fig_covariance_blocks_sigma_plus"),
}

AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 20


def lighten_cmap(name="viridis", amount=0.58):
    base = plt.get_cmap(name)
    colors = base(np.linspace(0, 1, 256))
    colors[:, :3] = colors[:, :3] * (1 - amount) + amount
    return LinearSegmentedColormap.from_list(f"{name}_light", colors)


def make_covariance(K=1000, T=100, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.linspace(0, 2 * np.pi, T, endpoint=False)
    phis = rng.random((K, 1)) * 2 * np.pi

    xs = np.cos(ts[None, :] + phis)
    ys = np.sin(ts[None, :] + phis)
    X = np.stack((xs, ys), axis=1)
    Z = X.reshape(K, T * 2)
    return np.einsum("ki,kj->ij", Z, Z) / K


def sigma_cov(cov, T, d=2):
    return cov.reshape(d, T, d, T).swapaxes(1, 3).reshape(T * d, T * d)


def block_labels(template):
    return {
        (0, 0): template.format(a="x", b="x"),
        (0, 1): template.format(a="x", b="y"),
        (1, 0): template.format(a="y", b="x"),
        (1, 1): template.format(a="y", b="y"),
    }


def plot_covariance_blocks(M, T, labels_inside, vlim):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.6))
    extent = [0, 2 * np.pi, 2 * np.pi, 0]
    ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    tick_labels = ["0", "π/2", "π", "3π/2", "2π"]
    cmap = lighten_cmap("viridis", amount=0.58)

    for (r, c), label in labels_inside.items():
        sub = M[r * T:(r + 1) * T, c * T:(c + 1) * T]
        ax = axes[r, c]
        ax.imshow(sub, vmin=-vlim, vmax=vlim, extent=extent, cmap=cmap)
        ax.text(
            0.5,
            0.5,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=26,
            color="black",
        )
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        # ax.set_xlabel(r"time $s$", fontsize=AXIS_LABEL_SIZE, labelpad=8)
        # ax.set_ylabel(r"time $t$", fontsize=AXIS_LABEL_SIZE, labelpad=8)

    fig.subplots_adjust(wspace=0.25, hspace=0.2)
    return fig


def save_figure(fig, output_base):
    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png_path}")
    print(f"saved {svg_path}")


def main():
    T = 100
    cov = make_covariance(K=1000, T=T, seed=0)
    cov_sigma = sigma_cov(cov, T, d=2)
    cov_minus = cov - cov_sigma
    cov_plus = cov + cov_sigma

    figures = [
        (
            cov,
            block_labels(r"$C_{{{a}{b}}}(t,t')$"),
            0.5,
            OUTPUTS["cov"],
        ),
        (
            cov_sigma,
            block_labels(r"$C_{{{a}{b}}}(t',t)$"),
            0.5,
            OUTPUTS["sigma"],
        ),
        (
            cov_minus,
            block_labels(r"$C^-_{{{a}{b}}}(t,t')$"),
            1.0,
            OUTPUTS["minus"],
        ),
        (
            cov_plus,
            block_labels(r"$C^+_{{{a}{b}}}(t,t')$"),
            1.0,
            OUTPUTS["plus"],
        ),
    ]

    for matrix, labels, vlim, output_base in figures:
        fig = plot_covariance_blocks(matrix, T, labels, vlim)
        save_figure(fig, output_base)


if __name__ == "__main__":
    main()
