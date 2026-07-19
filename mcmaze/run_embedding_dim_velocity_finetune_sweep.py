from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    32: "20260713_093518_d32_h256_dep2_bs64_ep200_lr1e-03_lxp0.0_lbt0.0_lcca1.0_sig10.0_s1",
    64: "20260713_093604_d64_h256_dep2_bs64_ep100_lr1e-03_lxp0.0_lbt0.0_lcca1.0_sig10.0_s1",
    128: "20260713_093632_d128_h256_dep2_bs64_ep100_lr1e-03_lxp0.0_lbt0.0_lcca1.0_sig10.0_s1",
    256: "20260713_093710_d256_h256_dep2_bs64_ep100_lr1e-03_lxp0.0_lbt0.0_lcca1.0_sig10.0_s1",
    512: "20260713_094453_d512_h256_dep2_bs64_ep100_lr1e-03_lxp0.0_lbt0.0_lcca1.0_sig10.0_s1",
    1024: "20260713_094633_d1024_h256_dep2_bs64_ep100_lr1e-03_lxp0.0_lbt0.0_lcca1.0_sig10.0_s1",
    2048: "20260713_094944_d2048_h256_dep2_bs64_ep100_lr1e-03_lxp0.0_lbt0.0_lcca1.0_sig10.0_s1",
}
LAYERS = ("output", "hidden")


def output_dir(run_dir: Path, dim: int, seed: int, layer: str, horizon_ms: int) -> Path:
    suffix = "_hidden" if layer == "hidden" else ""
    return run_dir / "outputs" / f"velocity_prediction_finetune_dim{dim}_seed{seed}_{horizon_ms}ms{suffix}"


def metrics_complete(metrics_path: Path) -> bool:
    if not metrics_path.is_file():
        return False
    with metrics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return any(row.get("model") == "finetuned_mlp" for row in rows)


def planned_jobs() -> list[tuple[int, int, str, str]]:
    jobs = []
    for dim, run_name in RUNS.items():
        seeds = [4] if dim == 32 else list(range(5))
        for seed in seeds:
            for layer in LAYERS:
                jobs.append((dim, seed, layer, run_name))
    return jobs


def run_jobs(args: argparse.Namespace) -> None:
    jobs = planned_jobs()
    log_dir = args.results.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Planned jobs: {len(jobs)}")
    for idx, (dim, seed, layer, run_name) in enumerate(jobs, 1):
        run_dir = REPO_ROOT / "mcmaze" / "runs" / run_name
        metrics_path = output_dir(run_dir, dim, seed, layer, args.horizon_ms) / "metrics.csv"
        if args.skip_existing and metrics_complete(metrics_path):
            print(f"[{idx}/{len(jobs)}] skip complete dim={dim} seed={seed} layer={layer}")
            continue

        cmd = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "predict_mcmaze_velocity_finetune.py"),
            "--run",
            str(run_dir),
            "--epochs",
            str(args.epochs),
            "--decoder-lr",
            str(args.decoder_lr),
            "--seed",
            str(seed),
            "--feature-layer",
            layer,
            "--output-suffix",
            f"dim{dim}_seed{seed}",
        ]
        log_path = log_dir / f"finetune_dim{dim}_seed{seed}_{layer}.log"
        print(f"[{idx}/{len(jobs)}] run dim={dim} seed={seed} layer={layer} log={log_path}", flush=True)
        with log_path.open("w") as log_file:
            subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )


def parse_dim_seed(path: Path) -> tuple[int | None, int | None]:
    match = re.search(r"dim(\d+)_seed(\d+)", str(path))
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"_d(\d+)_", str(path))
    return (int(match.group(1)), None) if match else (None, None)


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    expected = [
        output_dir(REPO_ROOT / "mcmaze" / "runs" / run_name, dim, seed, layer, args.horizon_ms)
        for dim, seed, layer, run_name in planned_jobs()
    ]
    discovered = []
    if args.include_discovered:
        discovered = sorted((REPO_ROOT / "mcmaze" / "runs").glob("*/outputs/velocity_prediction_finetune*"))
    for out_dir in sorted(set(expected + discovered)):
        metrics_path = out_dir / "metrics.csv"
        if not metrics_path.is_file():
            continue
        dim, seed = parse_dim_seed(out_dir)
        if dim not in RUNS:
            continue
        with metrics_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("embedder_init") != "pretrained":
                    continue
                layer = row.get("feature_layer", "output")
                if layer not in LAYERS:
                    continue
                enriched = {
                    "embedding_dim": dim,
                    "seed": seed if seed is not None else "",
                    "run_name": out_dir.parents[1].name,
                    "output_dir": str(out_dir.relative_to(REPO_ROOT)),
                    **row,
                    "feature_layer": layer,
                }
                rows.append(enriched)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "embedding_dim",
        "seed",
        "run_name",
        "output_dir",
        "model",
        "embedder_init",
        "feature_layer",
        "feature_dim",
        "mse_x",
        "mse_y",
        "rmse_x",
        "rmse_y",
        "rmse_mean",
        "r2_x",
        "r2_y",
        "r2_mean",
        "corr_x",
        "corr_y",
        "best_val_mse_z",
    ]
    with args.results.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["embedding_dim"], str(r["seed"]), r["feature_layer"], r["model"])))
    print(f"Wrote {len(rows)} rows to {args.results}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--decoder-lr", type=float, default=1e-2)
    parser.add_argument("--horizon-ms", type=int, default=100)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--include-discovered", action="store_true")
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "mcmaze" / "embedding_dim_vs_rmse_experiment" / "finetune_dim32_seed4_dim64_2048_seeds0_4_results.csv",
    )
    args = parser.parse_args()

    if not args.aggregate_only:
        run_jobs(args)
    aggregate(args)


if __name__ == "__main__":
    main()
