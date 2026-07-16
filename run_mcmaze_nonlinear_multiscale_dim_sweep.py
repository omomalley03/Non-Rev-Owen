"""Run MC Maze nonlinear multiscale-symmetric embedding dimension sweep.

For each embedding dimension, this script:
  1. Uses the MC Maze config/data settings.
  2. Forces a nonlinear multiscale symmetric embedding model:
       TEMPORAL_FRONTEND=symmetric_multiscale, TEMPORAL_FILTERS=4,
       MULTISCALE_SYMMETRIC_CONV_LAYERS=2, DEPTH=2, HIDDEN_DIM=256.
  3. Trains the embedder. ``train.py`` saves checkpoints/best.pt by validation
     zeta, so downstream decoding uses the best-zeta embedding checkpoint.
  4. Freezes embeddings and trains the MLP hand-velocity decoder.
  5. Doubles embedding dimension. By default it stops when decoder RMSE is worse
     than the best previous value for two consecutive dimensions; set
     ``--patience 0`` to force all requested dimensions.

Run from the repo root:

    /home/omo26/micromamba/envs/nonrev/bin/python -u run_mcmaze_nonlinear_multiscale_dim_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_nonrev")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = ROOT / "mcmaze" / "embedding_dim_vs_rmse_experiment" / "nonlinear_multiscale_zeta"
LOG_DIR = EXPERIMENT_DIR / "logs"
OUT_CSV = EXPERIMENT_DIR / "results.csv"
OUT_MSE_PLOT = EXPERIMENT_DIR / "embedding_dim_vs_velocity_mse_with_zeta.png"
OUT_RMSE_PLOT = EXPERIMENT_DIR / "embedding_dim_vs_velocity_rmse_with_zeta.png"


def source_mcmaze_config() -> dict[str, str]:
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


def normalize_frontend(value: str) -> str:
    value = (value or "").strip().lower()
    if value == "multiscale_symmetric":
        return "symmetric_multiscale"
    return value


def cfg_matches(cfg, dim: int, args) -> bool:
    return (
        int(getattr(cfg, "d", -1)) == int(dim)
        and int(getattr(cfg, "hidden_dim", -1)) == int(args.hidden_dim)
        and int(getattr(cfg, "depth", -1)) == int(args.depth)
        and int(getattr(cfg, "temporal_filters", -1)) == int(args.temporal_filters)
        and normalize_frontend(getattr(cfg, "temporal_frontend", "")) == normalize_frontend(args.temporal_frontend)
        and int(getattr(cfg, "multiscale_symmetric_conv_layers", 1)) == int(args.multiscale_symmetric_conv_layers)
        and str(getattr(cfg, "residual_kernels", "")) == str(args.residual_kernels)
        and int(getattr(cfg, "epochs", -1)) == int(args.epochs)
        and float(getattr(cfg, "lambda_block_cca", -1.0)) == float(args.lambda_block_cca)
        and int(getattr(cfg, "seed", -1)) == int(args.seed)
    )


def checkpoint_metadata(run_dir: str) -> dict[str, object] | None:
    ckpt_path = Path(run_dir) / "checkpoints" / "best.pt"
    if not ckpt_path.is_file():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "path": str(ckpt_path),
        "config": ckpt["config"],
        "epoch": ckpt.get("epoch"),
        "checkpoint_selection": ckpt.get("checkpoint_selection"),
        "val_s": ckpt.get("val_s"),
        "val_zeta": ckpt.get("val_zeta"),
        "val_c_plus": ckpt.get("val_c_plus"),
        "val_loss": ckpt.get("val_loss"),
    }


def training_log_complete(run_dir: str, expected_epochs: int) -> bool:
    path = Path(run_dir) / "outputs" / "log.csv"
    if not path.is_file():
        return False
    with path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f)) >= int(expected_epochs)


def find_existing_run(dim: int, args) -> str | None:
    candidates = []
    runs_root = ROOT / "mcmaze" / "runs"
    if not runs_root.is_dir():
        return None
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        meta = checkpoint_metadata(str(run_dir))
        if meta is None:
            continue
        if meta["checkpoint_selection"] != "best_val_zeta":
            continue
        if not cfg_matches(meta["config"], dim, args):
            continue
        if not training_log_complete(str(run_dir), args.epochs):
            continue
        candidates.append((run_dir.stat().st_mtime, str(run_dir)))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def read_mlp_metrics(run_dir: str, horizon_ms: int) -> dict[str, str] | None:
    path = Path(run_dir) / "outputs" / f"velocity_prediction_{horizon_ms}ms" / "metrics.csv"
    if not path.is_file():
        return None
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == "mlp":
                row["metrics_path"] = str(path)
                return row
    return None


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


def _plot_metric_with_zeta(
    records: list[dict[str, object]],
    metric_key: str,
    ylabel: str,
    title: str,
    color: str,
    path: Path,
) -> None:
    dims = [int(r["dimension"]) for r in records]
    zetas = [float(r["embedding_val_zeta"]) for r in records]

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    line_metric = ax.plot(
        dims,
        [float(r[metric_key]) for r in records],
        marker="o",
        linewidth=2,
        color=color,
        label=ylabel,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Embedding dimension [log scale]")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.25)
    ax.spines["top"].set_visible(False)

    ax_zeta = ax.twinx()
    line_zeta = ax_zeta.plot(
        dims,
        zetas,
        marker="s",
        linewidth=1.8,
        color="seagreen",
        label="Best validation ζ",
    )
    ax_zeta.set_ylabel("Best validation ζ")
    ax_zeta.spines["top"].set_visible(False)

    handles = line_metric + line_zeta
    ax.legend(handles, [h.get_label() for h in handles], loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_outputs(records: list[dict[str, object]]) -> None:
    records = sorted(records, key=lambda r: int(r["dimension"]))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dimension",
        "mse_mean",
        "mse_x",
        "mse_y",
        "rmse_mean",
        "rmse_x",
        "rmse_y",
        "r2_mean",
        "corr_x",
        "corr_y",
        "best_val_mse_z",
        "checkpoint_epoch",
        "checkpoint_selection",
        "embedding_val_s",
        "embedding_val_zeta",
        "embedding_val_c_plus",
        "embedding_val_loss",
        "run",
        "metrics_path",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    _plot_metric_with_zeta(
        records,
        metric_key="mse_mean",
        ylabel="Mean hand velocity MSE",
        title="MC Maze nonlinear multiscale embeddings: decoder MSE",
        color="#1f77b4",
        path=OUT_MSE_PLOT,
    )
    _plot_metric_with_zeta(
        records,
        metric_key="rmse_mean",
        ylabel="Mean hand velocity RMSE [mm/s]",
        title="MC Maze nonlinear multiscale embeddings: decoder RMSE",
        color="#1f77b4",
        path=OUT_RMSE_PLOT,
    )


def row_from_run(dim: int, run_dir: str, metrics: dict[str, str]) -> dict[str, object]:
    meta = checkpoint_metadata(run_dir)
    if meta is None:
        raise RuntimeError(f"No best.pt metadata found for {run_dir}")
    mse_mean = metrics.get("mse_mean")
    if mse_mean is None:
        mse_mean = (float(metrics["mse_x"]) + float(metrics["mse_y"])) / 2.0
    return {
        "dimension": dim,
        "mse_mean": mse_mean,
        "mse_x": metrics["mse_x"],
        "mse_y": metrics["mse_y"],
        "rmse_mean": metrics["rmse_mean"],
        "rmse_x": metrics["rmse_x"],
        "rmse_y": metrics["rmse_y"],
        "r2_mean": metrics["r2_mean"],
        "corr_x": metrics["corr_x"],
        "corr_y": metrics["corr_y"],
        "best_val_mse_z": metrics["best_val_mse_z"],
        "checkpoint_epoch": meta["epoch"],
        "checkpoint_selection": meta["checkpoint_selection"],
        "embedding_val_s": meta["val_s"],
        "embedding_val_zeta": meta["val_zeta"],
        "embedding_val_c_plus": meta["val_c_plus"],
        "embedding_val_loss": meta["val_loss"],
        "run": run_dir,
        "metrics_path": metrics["metrics_path"],
    }


def decoder_command(run_dir: str, args) -> list[str]:
    return [
        sys.executable,
        "-u",
        "predict_mcmaze_velocity.py",
        "--run",
        run_dir,
        "--horizon-ms",
        str(args.horizon_ms),
        "--mlp-hidden-dim",
        str(args.decoder_hidden_dim),
        "--mlp-depth",
        str(args.decoder_depth),
        "--mlp-dropout",
        str(args.decoder_dropout),
        "--mlp-epochs",
        str(args.decoder_epochs),
        "--mlp-batch-size",
        str(args.decoder_batch_size),
        "--mlp-lr",
        str(args.decoder_lr),
        "--mlp-weight-decay",
        str(args.decoder_weight_decay),
        "--seed",
        str(args.decoder_seed),
    ]


def train_env(base_env: dict[str, str], dim: int, args) -> dict[str, str]:
    env = base_env.copy()
    env.update(
        {
            "DATASET_NAME": "mcmaze",
            "D": str(dim),
            "HIDDEN_DIM": str(args.hidden_dim),
            "DEPTH": str(args.depth),
            "TEMPORAL_FRONTEND": args.temporal_frontend,
            "TEMPORAL_FILTERS": str(args.temporal_filters),
            "MULTISCALE_SYMMETRIC_CONV_LAYERS": str(args.multiscale_symmetric_conv_layers),
            "RESIDUAL_KERNELS": args.residual_kernels,
            "EPOCHS": str(args.epochs),
            "LAMBDA_BLOCK_CCA": str(args.lambda_block_cca),
            "SEED": str(args.seed),
            "VAL_CHECKPOINT_METRIC": "zeta",
            "VAL_ZETA_CHECKPOINTS": "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
            "PYTHONUNBUFFERED": "1",
            "QUIET_TRAIN": "1",
            "SKIP_DIAGNOSTICS": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib_nonrev",
        }
    )
    return env


def should_stop(records: list[dict[str, object]], patience: int, metric_key: str) -> bool:
    if patience < 1 or len(records) <= patience:
        return False
    ordered = sorted(records, key=lambda r: int(r["dimension"]))
    best_before = min(float(r[metric_key]) for r in ordered[:-patience])
    recent = ordered[-patience:]
    return all(float(r[metric_key]) >= best_before for r in recent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-dim", type=int, default=16)
    parser.add_argument("--max-dim", type=int, default=2048)
    parser.add_argument("--dims", type=int, nargs="+", default=None,
                        help="Explicit dimensions. Otherwise double from --start-dim to --max-dim.")
    parser.add_argument("--patience", type=int, default=2,
                        help="Stop after this many consecutive dimensions fail to improve the selected decoder metric.")
    parser.add_argument("--stop-metric", choices=["rmse_mean", "mse_mean"], default="rmse_mean",
                        help="Decoder metric used for early stopping across embedding dimensions.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows in the output CSV.")
    parser.add_argument("--reuse-existing-runs", action="store_true",
                        help="Reuse compatible best-zeta runs even if they are not in the CSV.")
    parser.add_argument("--horizon-ms", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--temporal-frontend", default="symmetric_multiscale")
    parser.add_argument("--temporal-filters", type=int, default=4)
    parser.add_argument("--multiscale-symmetric-conv-layers", type=int, default=2,
                        help="Number of symmetric conv layers in each multiscale branch.")
    parser.add_argument("--residual-kernels", default="7,15,31,61")
    parser.add_argument("--lambda-block-cca", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--decoder-dropout", type=float, default=0.1)
    parser.add_argument("--decoder-epochs", type=int, default=50)
    parser.add_argument("--decoder-batch-size", type=int, default=4096)
    parser.add_argument("--decoder-lr", type=float, default=1e-3)
    parser.add_argument("--decoder-weight-decay", type=float, default=1e-4)
    parser.add_argument("--decoder-seed", type=int, default=0)
    args = parser.parse_args()

    if args.dims is None:
        dims = []
        dim = args.start_dim
        while dim <= args.max_dim:
            dims.append(dim)
            dim *= 2
    else:
        dims = args.dims

    completed = existing_records(OUT_CSV) if args.resume else {}
    records: list[dict[str, object]] = [completed[d] for d in dims if d in completed]
    base_env = source_mcmaze_config()

    for dim in dims:
        if dim in completed:
            print(f"Reusing completed CSV row for D={dim}: {completed[dim]['run']}")
            write_outputs(records)
            if should_stop(records, args.patience, args.stop_metric):
                print(
                    f"Stopping: decoder {args.stop_metric} failed to improve for "
                    f"{args.patience} consecutive dimensions after D={dim}."
                )
                break
            continue

        env = train_env(base_env, dim, args)
        run_dir = find_existing_run(dim, args) if args.reuse_existing_runs else None

        if run_dir is None:
            print(f"\n=== Training nonlinear multiscale MC Maze embedder: D={dim} ===")
            train_log = LOG_DIR / f"d{dim}_train.log"
            train_output = run_logged([sys.executable, "-u", "main.py"], env, train_log)
            run_dir = parse_run_dir(train_output)
        else:
            print(f"\n=== Reusing compatible best-zeta embedder: D={dim} ===")
            print(run_dir)

        metrics = read_mlp_metrics(run_dir, args.horizon_ms)
        if metrics is None:
            print(f"\n=== Training frozen MLP velocity decoder: D={dim} ===")
            decoder_log = LOG_DIR / f"d{dim}_decoder.log"
            run_logged(decoder_command(run_dir, args), env, decoder_log)
            metrics = read_mlp_metrics(run_dir, args.horizon_ms)
        if metrics is None:
            raise RuntimeError(f"No MLP decoder metrics found for {run_dir}")

        row = row_from_run(dim, run_dir, metrics)
        records.append(row)
        write_outputs(records)
        print(
            f"D={dim} MLP MSE mean: {float(row['mse_mean']):.4f}  "
            f"RMSE mean: {float(row['rmse_mean']):.4f}  "
            f"embedding ζ: {float(row['embedding_val_zeta']):.4f}"
        )
        print(f"Updated {OUT_CSV}")
        print(f"Updated {OUT_MSE_PLOT}")
        print(f"Updated {OUT_RMSE_PLOT}")

        if should_stop(records, args.patience, args.stop_metric):
            print(
                f"Stopping: decoder {args.stop_metric} failed to improve for "
                f"{args.patience} consecutive dimensions after D={dim}."
            )
            break


if __name__ == "__main__":
    main()
