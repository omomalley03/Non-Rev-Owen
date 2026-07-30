"""Run PhysioNetMI condition decoders for the embedding-dimension sweep.

Uses the newest matching d32..d2048 PhysioNetMI embedding checkpoints from the
50-epoch multiscale-symmetric sweep.  Each job runs
``predict_physionet_condition_finetune.py`` once, which trains both the frozen
decoder baseline and the finetuned embedder+decoder model.

Defaults:
  - dimensions: 32, 64, 128, 256, 512, 1024, 2048
  - decoder seeds: 0, 1, 2, 3, 4
  - feature layer: hidden
  - decoder LR: 1e-2
  - epochs: 50

Usage:
    micromamba run -n nonrev python -u sweep_scripts/run_physionet_embedding_dim_condition_decoder_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "physionetmi" / "synth_runs"
SWEEP_DIR = REPO_ROOT / "physionetmi" / "embedding_dim_multiscale_symmetric_decoder_sweep"
DEFAULT_DIMS = (32, 64, 128, 256, 512, 1024, 2048)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def parse_int_list(spec: str, name: str) -> list[int]:
    values = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    return values


def run_dim(path: Path) -> int | None:
    match = re.search(r"_d(\d+)_", path.name)
    return int(match.group(1)) if match else None


def find_embedding_runs(dims: list[int]) -> dict[int, Path]:
    runs: dict[int, Path] = {}
    missing = []
    for dim in dims:
        pattern = (
            f"*_d{dim}_h256_dep2_bs128_ep50_lr1e-03"
            "_lxp0.0_lbt0.0_lcca1.0_sig10.0_s4"
        )
        candidates = [
            path
            for path in RUNS_ROOT.glob(pattern)
            if (path / "checkpoints" / "best.pt").is_file()
        ]
        if not candidates:
            missing.append(dim)
            continue
        runs[dim] = max(candidates, key=lambda path: (path / "checkpoints" / "best.pt").stat().st_mtime)
    if missing:
        raise FileNotFoundError(f"No matching 50-epoch PhysioNet embedding run found for dims: {missing}")
    return runs


def output_dir(run_dir: Path, dim: int, seed: int) -> Path:
    return run_dir / "outputs" / f"condition_prediction_finetune_dim{dim}_seed{seed}_hidden"


def metrics_complete(metrics_path: Path) -> bool:
    if not metrics_path.is_file():
        return False
    with metrics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    models = {row.get("model") for row in rows}
    return {"frozen_temporal_conv", "finetuned_temporal_conv"}.issubset(models)


def planned_jobs(args: argparse.Namespace) -> list[tuple[int, int, Path]]:
    dims = parse_int_list(args.dims, "--dims")
    seeds = parse_int_list(args.seeds, "--seeds")
    runs = find_embedding_runs(dims)
    return [(dim, seed, runs[dim]) for dim in dims for seed in seeds]


def run_jobs(args: argparse.Namespace) -> None:
    jobs = planned_jobs(args)
    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    python_exe = args.python or sys.executable

    print(f"Planned decoder jobs: {len(jobs)}")
    for idx, (dim, seed, run_dir) in enumerate(jobs, 1):
        metrics_path = output_dir(run_dir, dim, seed) / "metrics.csv"
        if args.skip_existing and metrics_complete(metrics_path):
            print(f"[{idx}/{len(jobs)}] skip complete dim={dim} seed={seed}")
            continue

        cmd = [
            python_exe,
            "-u",
            str(REPO_ROOT / "predict_physionet_condition_finetune.py"),
            "--run",
            str(run_dir),
            "--epochs",
            str(args.epochs),
            "--decoder-lr",
            str(args.decoder_lr),
            "--feature-layer",
            "hidden",
            "--seed",
            str(seed),
            "--output-suffix",
            f"dim{dim}_seed{seed}",
        ]
        log_path = log_dir / f"d{dim}_seed{seed}.log"
        print(
            f"[{idx}/{len(jobs)}] run dim={dim} seed={seed} "
            f"run={run_dir.name} log={log_path}",
            flush=True,
        )
        if args.dry_run:
            print(" ".join(cmd))
            continue
        with log_path.open("w") as log_file:
            subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    for dim, seed, run_dir in planned_jobs(args):
        metrics_path = output_dir(run_dir, dim, seed) / "metrics.csv"
        if not metrics_path.is_file():
            continue
        with metrics_path.open(newline="") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "embedding_dim": dim,
                        "decoder_seed": seed,
                        "run_name": run_dir.name,
                        "output_dir": str(output_dir(run_dir, dim, seed).relative_to(REPO_ROOT)),
                        **row,
                    }
                )

    args.results.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "embedding_dim",
        "decoder_seed",
        "run_name",
        "output_dir",
        "model",
        "embedder_init",
        "feature_layer",
        "feature_dim",
        "flat_feature_dim",
        "decoder_type",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "best_val_acc",
        "best_val_ce",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
    ]
    with args.results.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda row: (row["embedding_dim"], row["decoder_seed"], row["model"]))
        )
    print(f"Wrote {len(rows)} rows to {args.results}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", default=",".join(str(dim) for dim in DEFAULT_DIMS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--decoder-lr", type=float, default=1e-2)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default="")
    parser.add_argument("--log-dir", type=Path, default=SWEEP_DIR / "logs")
    parser.add_argument("--results", type=Path, default=SWEEP_DIR / "results.csv")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.decoder_lr <= 0:
        parser.error("--decoder-lr must be positive")

    if not args.aggregate_only:
        run_jobs(args)
    aggregate(args)


if __name__ == "__main__":
    main()
