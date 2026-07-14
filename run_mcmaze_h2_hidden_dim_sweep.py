"""Run MC Maze D sweep with a 2-wide embedder hidden layer and hidden decoder.

For each embedding dimension D=16..2048 by powers of two, this script:
  1. Sources mcmaze_config.sh for the baseline MC Maze settings.
  2. Overrides only D and HIDDEN_DIM=2 for the embedder.
  3. Trains main.py, whose checkpoints/best.pt is selected by validation zeta.
  4. Removes the embedder's final linear layer and trains the hand-velocity MLP
     decoder from the penultimate hidden features.

Dimension-scaling diagnostics stay disabled through SKIP_DIAGNOSTICS=1. Training
loss curves and decoder loss curves are still written.

Run from the repo root:

    /home/omo26/micromamba/envs/nonrev/bin/python -u run_mcmaze_h2_hidden_dim_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = ROOT / "mcmaze" / "embedding_dim_vs_rmse_experiment" / "h2_hidden_best_zeta"
LOG_DIR = EXPERIMENT_DIR / "logs"
OUT_CSV = EXPERIMENT_DIR / "results.csv"


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


def cfg_matches(cfg, dim: int, hidden_dim: int, base_env: dict[str, str]) -> bool:
    return (
        int(getattr(cfg, "d", -1)) == int(dim)
        and int(getattr(cfg, "hidden_dim", -1)) == int(hidden_dim)
        and int(getattr(cfg, "depth", -1)) == int(base_env["DEPTH"])
        and int(getattr(cfg, "temporal_filters", -1)) == int(base_env["TEMPORAL_FILTERS"])
        and str(getattr(cfg, "temporal_frontend", "")) == str(base_env["TEMPORAL_FRONTEND"])
        and int(getattr(cfg, "multiscale_symmetric_conv_layers", 1)) == int(
            base_env.get("MULTISCALE_SYMMETRIC_CONV_LAYERS", "1")
        )
        and str(getattr(cfg, "residual_kernels", "")) == str(base_env["RESIDUAL_KERNELS"])
        and int(getattr(cfg, "epochs", -1)) == int(base_env["EPOCHS"])
        and float(getattr(cfg, "lambda_block_cca", -1.0)) == float(base_env["LAMBDA_BLOCK_CCA"])
        and int(getattr(cfg, "seed", -1)) == int(base_env.get("SEED", "1"))
    )


def find_existing_run(dim: int, hidden_dim: int, base_env: dict[str, str]) -> str | None:
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
        if not cfg_matches(meta["config"], dim, hidden_dim, base_env):
            continue
        if not training_log_complete(str(run_dir), int(base_env["EPOCHS"])):
            continue
        candidates.append((run_dir.stat().st_mtime, str(run_dir)))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def read_mlp_metrics(run_dir: str, horizon_ms: int) -> dict[str, str] | None:
    path = Path(run_dir) / "outputs" / f"velocity_prediction_{horizon_ms}ms_hidden" / "metrics.csv"
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
        loss_curve_path = Path(row.get("decoder_loss_curve_path", ""))
        run_dir = Path(row.get("run", ""))
        if metrics_path.is_file() and loss_curve_path.is_file() and run_dir.is_dir():
            out[dim] = row
    return out


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
        "feature_layer",
        "feature_dim",
        "checkpoint_epoch",
        "checkpoint_selection",
        "embedding_val_s",
        "embedding_val_zeta",
        "embedding_val_c_plus",
        "embedding_val_loss",
        "run",
        "metrics_path",
        "training_loss_curve_path",
        "decoder_loss_curve_path",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def row_from_run(dim: int, run_dir: str, metrics: dict[str, str], horizon_ms: int) -> dict[str, object]:
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
        "feature_layer": metrics.get("feature_layer", "hidden"),
        "feature_dim": metrics.get("feature_dim", ""),
        "checkpoint_epoch": meta["epoch"],
        "checkpoint_selection": meta["checkpoint_selection"],
        "embedding_val_s": meta["val_s"],
        "embedding_val_zeta": meta["val_zeta"],
        "embedding_val_c_plus": meta["val_c_plus"],
        "embedding_val_loss": meta["val_loss"],
        "run": run_dir,
        "metrics_path": metrics["metrics_path"],
        "training_loss_curve_path": str(Path(run_dir) / "outputs" / "loss_curve.png"),
        "decoder_loss_curve_path": str(
            Path(run_dir) / "outputs" / f"velocity_prediction_{horizon_ms}ms_hidden" / "mlp_loss_curve.png"
        ),
    }


def decoder_command(run_dir: str, args) -> list[str]:
    return [
        sys.executable,
        "-u",
        "predict_mcmaze_velocity.py",
        "--run",
        run_dir,
        "--feature-layer",
        "hidden",
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
        "--disable-plots",
    ]


def train_env(base_env: dict[str, str], dim: int, hidden_dim: int) -> dict[str, str]:
    env = base_env.copy()
    env.update(
        {
            "DATASET_NAME": "mcmaze",
            "D": str(dim),
            "HIDDEN_DIM": str(hidden_dim),
            "VAL_CHECKPOINT_METRIC": "zeta",
            "PYTHONUNBUFFERED": "1",
            "QUIET_TRAIN": "1",
            "SKIP_DIAGNOSTICS": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib_nonrev",
        }
    )
    return env


def verify_row(row: dict[str, object]) -> None:
    for key in ("metrics_path", "training_loss_curve_path", "decoder_loss_curve_path"):
        path = Path(str(row[key]))
        if not path.is_file():
            raise RuntimeError(f"expected output does not exist: {path}")
    if row.get("checkpoint_selection") != "best_val_zeta":
        raise RuntimeError(f"expected best_val_zeta checkpoint, got {row.get('checkpoint_selection')!r}")
    if row.get("feature_layer") != "hidden":
        raise RuntimeError(f"expected hidden decoder features, got {row.get('feature_layer')!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-dim", type=int, default=16)
    parser.add_argument("--max-dim", type=int, default=2048)
    parser.add_argument("--dims", type=int, nargs="+", default=None,
                        help="Explicit dimensions. Otherwise double from --start-dim to --max-dim.")
    parser.add_argument("--hidden-dim", type=int, default=2,
                        help="Embedder hidden width. Default is the requested h=2 sweep.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows in the output CSV.")
    parser.add_argument("--reuse-existing-runs", action="store_true",
                        help="Reuse compatible completed h2 best-zeta runs if present.")
    parser.add_argument("--horizon-ms", type=int, default=100)
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
            verify_row(completed[dim])
            write_outputs(records)
            continue

        env = train_env(base_env, dim, args.hidden_dim)
        run_dir = find_existing_run(dim, args.hidden_dim, base_env) if args.reuse_existing_runs else None

        if run_dir is None:
            print(f"\n=== Training MC Maze embedder: D={dim}, HIDDEN_DIM={args.hidden_dim} ===")
            train_log = LOG_DIR / f"d{dim}_h{args.hidden_dim}_train.log"
            train_output = run_logged([sys.executable, "-u", "main.py"], env, train_log)
            run_dir = parse_run_dir(train_output)
        else:
            print(f"\n=== Reusing compatible MC Maze embedder: D={dim}, HIDDEN_DIM={args.hidden_dim} ===")
            print(run_dir)

        metrics = read_mlp_metrics(run_dir, args.horizon_ms)
        if metrics is None:
            print(f"\n=== Training hidden-feature MLP velocity decoder: D={dim} ===")
            decoder_log = LOG_DIR / f"d{dim}_h{args.hidden_dim}_hidden_decoder.log"
            run_logged(decoder_command(run_dir, args), env, decoder_log)
            metrics = read_mlp_metrics(run_dir, args.horizon_ms)
        if metrics is None:
            raise RuntimeError(f"No hidden-feature MLP decoder metrics found for {run_dir}")

        row = row_from_run(dim, run_dir, metrics, args.horizon_ms)
        verify_row(row)
        records.append(row)
        write_outputs(records)
        print(
            f"D={dim} hidden-feature MLP RMSE mean: {float(row['rmse_mean']):.4f}  "
            f"embedding ζ: {float(row['embedding_val_zeta']):.4f}  "
            f"best epoch: {row['checkpoint_epoch']}"
        )
        print(f"Updated {OUT_CSV}")


if __name__ == "__main__":
    main()
