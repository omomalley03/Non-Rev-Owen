"""Run an MC Maze embedding-dimension sweep for the no-conv MLP config.

For each embedding dimension, this script:
  1. Sources the current ``mcmaze_config.sh`` settings.
  2. Forces ``TEMPORAL_FILTERS=0`` and overrides only ``D``.
  3. Trains the MC Maze embedder with ``main.py``.
  4. Trains the frozen-embedding MLP velocity decoder with
     ``predict_mcmaze_velocity.py``.
  5. Writes a CSV and an RMSE-vs-dimension plot.

Run from the repo root:

    micromamba run -n nonrev python -u run_mcmaze_no_conv_dim_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_DIMS = (16, 32, 64, 128, 256, 512, 1024)
LOG_DIR = ROOT / "mcmaze" / "sweep_logs" / "mlp_no_conv_current"
OUT_CSV = ROOT / "mcmaze" / "mlp_no_conv_embedding_dim_velocity_rmse.csv"
OUT_PLOT = ROOT / "mcmaze" / "mlp_no_conv_embedding_dim_velocity_rmse.png"


def source_mcmaze_config() -> dict[str, str]:
    """Return the environment produced by sourcing mcmaze_config.sh."""
    cmd = ". ./mcmaze_config.sh >/dev/null 2>&1; env"
    proc = subprocess.run(
        ["sh", "-lc", cmd],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    env = os.environ.copy()
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key] = value
    return env


def run_logged(cmd: list[str], env: dict[str, str], log_path: Path) -> str:
    """Run a subprocess, teeing combined stdout/stderr to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output = []
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            output.append(line)
            log.write(line)
            log.flush()
            print(line, end="")
        rc = proc.wait()
    text = "".join(output)
    if rc != 0:
        raise RuntimeError(f"command failed with exit code {rc}: {' '.join(cmd)}")
    return text


def parse_run_dir(train_output: str) -> str:
    match = re.search(r"^Run directory:\s*(.+)$", train_output, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("could not find 'Run directory:' in training output")
    return match.group(1).strip()


def read_mlp_metrics(run_dir: str, horizon_ms: int) -> dict[str, str]:
    path = Path(run_dir) / "outputs" / f"velocity_prediction_{horizon_ms}ms" / "metrics.csv"
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == "mlp":
                return row
    raise RuntimeError(f"No MLP row found in {path}")


def existing_records(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        try:
            dim = int(row["dimension"])
        except (KeyError, ValueError):
            continue
        metrics_path = Path(row.get("metrics_path", ""))
        if metrics_path.is_file():
            out[dim] = row
    return out


def write_outputs(records: list[dict[str, object]]) -> None:
    records = sorted(records, key=lambda r: int(r["dimension"]))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dimension",
        "rmse_mean",
        "rmse_x",
        "rmse_y",
        "r2_mean",
        "corr_x",
        "corr_y",
        "best_val_mse_z",
        "run",
        "metrics_path",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        [int(r["dimension"]) for r in records],
        [float(r["rmse_mean"]) for r in records],
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="MLP, no temporal conv",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(DEFAULT_DIMS))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Mean hand velocity RMSE (mm/s)")
    ax.set_title("MC Maze velocity prediction: no-conv MLP sweep")
    ax.grid(True, which="major", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=list(DEFAULT_DIMS))
    parser.add_argument("--horizon-ms", type=int, default=100)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows already present in the output CSV.")
    args = parser.parse_args()

    completed = existing_records(OUT_CSV) if args.resume else {}
    records: list[dict[str, object]] = [completed[d] for d in args.dims if d in completed]

    base_env = source_mcmaze_config()
    base_env["DATASET_NAME"] = "mcmaze"
    base_env["TEMPORAL_FILTERS"] = "0"
    base_env["PYTHONUNBUFFERED"] = "1"
    base_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_nonrev")

    for dim in args.dims:
        if dim in completed:
            print(f"Reusing completed D={dim} from {completed[dim]['run']}")
            continue

        env = base_env.copy()
        env["D"] = str(dim)

        print(f"\n=== Training no-conv MC Maze embedder: D={dim} ===")
        train_log = LOG_DIR / f"d{dim}_train.log"
        train_output = run_logged([sys.executable, "-u", "main.py"], env, train_log)
        run_dir = parse_run_dir(train_output)

        print(f"\n=== Training frozen MLP velocity decoder: D={dim} ===")
        decoder_log = LOG_DIR / f"d{dim}_decoder.log"
        run_logged(
            [
                sys.executable,
                "-u",
                "predict_mcmaze_velocity.py",
                "--run",
                run_dir,
                "--horizon-ms",
                str(args.horizon_ms),
                "--mlp-epochs",
                str(args.mlp_epochs),
            ],
            env,
            decoder_log,
        )

        metrics = read_mlp_metrics(run_dir, args.horizon_ms)
        row = {
            "dimension": dim,
            "rmse_mean": metrics["rmse_mean"],
            "rmse_x": metrics["rmse_x"],
            "rmse_y": metrics["rmse_y"],
            "r2_mean": metrics["r2_mean"],
            "corr_x": metrics["corr_x"],
            "corr_y": metrics["corr_y"],
            "best_val_mse_z": metrics["best_val_mse_z"],
            "run": run_dir,
            "metrics_path": str(Path(run_dir) / "outputs" / f"velocity_prediction_{args.horizon_ms}ms" / "metrics.csv"),
        }
        records.append(row)
        write_outputs(records)
        print(f"D={dim} MLP RMSE mean: {float(metrics['rmse_mean']):.4f}")
        print(f"Updated {OUT_CSV}")
        print(f"Updated {OUT_PLOT}")


if __name__ == "__main__":
    main()
