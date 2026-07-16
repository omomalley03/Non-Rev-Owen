"""Plot validation non-reversibility components from a run log.

The full per-epoch plot requires training logs with these columns:

    val_s, val_c_plus, val_zeta

Older runs usually have only val_s. For those runs this script can still plot
C- / validation S, but it cannot reconstruct per-epoch C+ or ζ without saved
per-epoch checkpoints.

Examples
--------
    source mcmaze_config.sh
    python plot_nonrev_components.py
    python plot_nonrev_components.py --run mcmaze/runs/20260710_064929_...
    python plot_nonrev_components.py --run 2 --out /tmp/nonrev_components.png
"""

from __future__ import annotations

import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_nonrev")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import RUNS_BASE, RUNS_DIR


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
            os.path.join(RUNS_BASE, "physionetmi", "synth_runs"),
            os.path.join(repo_root, "mcmaze", "runs"),
            os.path.join(repo_root, "physionetmi", "synth_runs"),
            os.path.join(repo_root, "runs"),
            os.path.join(repo_root, "synth_runs"),
        ]
    )
    completed = []
    for root in roots:
        for name in os.listdir(root):
            run_dir = os.path.join(root, name)
            log_path = os.path.join(run_dir, "outputs", "log.csv")
            if os.path.isfile(log_path):
                completed.append((os.path.getmtime(log_path), run_dir))
    return [run_dir for _, run_dir in sorted(completed, reverse=True)]


def resolve_run_dir(arg_run: str | None) -> str:
    completed = _completed_runs()
    repo_root = os.path.dirname(os.path.abspath(__file__))

    if arg_run is None:
        if not completed:
            raise FileNotFoundError("No runs with outputs/log.csv were found.")
        print("Using most recent run with outputs/log.csv. Available runs:")
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

    return os.path.abspath(arg_run)


def _float_or_nan(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def read_log(log_path: str) -> tuple[list[str], dict[str, np.ndarray]]:
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError(f"No rows found in {log_path!r}")

    data = {}
    for name in fieldnames:
        values = []
        for row in rows:
            if name == "epoch":
                values.append(int(float(row[name])))
            else:
                values.append(_float_or_nan(row.get(name, "")))
        data[name] = np.asarray(values)
    return fieldnames, data


def _plot_series(ax, x, y, label, color):
    ax.plot(x, y, lw=1.8, color=color, label=label)
    best_idx = int(np.nanargmax(y)) if np.isfinite(y).any() else None
    if best_idx is not None:
        ax.scatter([x[best_idx]], [y[best_idx]], s=42, color=color, edgecolors="black", zorder=5)
        ax.annotate(
            f"max @ epoch {int(x[best_idx])}",
            xy=(x[best_idx], y[best_idx]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.legend(loc="best", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def plot_components(run_dir: str, out_path: str | None = None) -> str:
    log_path = os.path.join(run_dir, "outputs", "log.csv")
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"No log at {log_path!r}")

    fields, data = read_log(log_path)
    if "epoch" not in data:
        raise ValueError(f"{log_path!r} has no epoch column")
    if "val_s" not in data:
        raise ValueError(f"{log_path!r} has no val_s column")

    epoch = data["epoch"]
    has_c_plus = "val_c_plus" in fields
    has_zeta = "val_zeta" in fields

    if out_path is None:
        out_path = os.path.join(run_dir, "outputs", "nonrev_components.png")

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)
    _plot_series(
        axes[0],
        epoch,
        data["val_s"],
        "validation C- / S mean per plane",
        "steelblue",
    )
    axes[0].set_ylabel("C- / S")

    if has_c_plus:
        _plot_series(
            axes[1],
            epoch,
            data["val_c_plus"],
            "validation C+ mean per plane",
            "darkorange",
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "val_c_plus was not saved in this run.\n"
            "Per-epoch C+ cannot be reconstructed without per-epoch checkpoints.",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    axes[1].set_ylabel("C+")
    axes[1].spines[["top", "right"]].set_visible(False)

    if has_zeta:
        _plot_series(
            axes[2],
            epoch,
            data["val_zeta"],
            "validation ζ = C- / C+",
            "seagreen",
        )
    else:
        axes[2].text(
            0.5,
            0.5,
            "val_zeta was not saved in this run.\n"
            "Per-epoch ζ cannot be reconstructed without per-epoch checkpoints.",
            transform=axes[2].transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    axes[2].set_ylabel("ζ")
    axes[2].set_xlabel("Epoch")
    axes[2].spines[["top", "right"]].set_visible(False)

    title = os.path.basename(os.path.normpath(run_dir))
    fig.suptitle(f"Validation non-reversibility components\n{title}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {out_path}")
    if not (has_c_plus and has_zeta):
        missing = []
        if not has_c_plus:
            missing.append("val_c_plus")
        if not has_zeta:
            missing.append("val_zeta")
        print(
            "Warning: "
            + ", ".join(missing)
            + " not found in outputs/log.csv; only saved columns were plotted."
        )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None, help="Integer rank, run directory, or omit for newest log.")
    parser.add_argument("--out", default=None, help="Output PNG path. Defaults to <run>/outputs/nonrev_components.png.")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    print(f"Using run: {run_dir}")
    plot_components(run_dir, args.out)


if __name__ == "__main__":
    main()
