"""Compare MC Maze fine-tuning from pretrained vs random embedder init.

Pass a completed seed-sweep folder. The script reads ``results.csv``, selects
one condition group by 1-based index, and runs ``predict_mcmaze_velocity_finetune.py``
for every seed run in that condition with:

* ``--embedder-init pretrained``: start from the frozen checkpoint and initialize
  the fine-tune decoder from the frozen decoder baseline.
* ``--embedder-init random``: train the embedder and decoder end-to-end from
  random initialization.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model.py").is_file())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import csv
import sys
from pathlib import Path

from sweep_scripts.common.finetune_init_comparison_common import (
    REPO_ROOT,
    append_summary_rows,
    copied_source_fields,
    existing_completed,
    metrics_complete,
    metric_row,
    print_condition_listing,
    read_rows,
    run_logged,
    select_condition_rows,
    write_ci95_summary,
    write_paired_summary,
)


METRIC_FIELDS = (
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
)
FIELDNAMES = [
    "source_condition",
    "source_temporal_frontend",
    "source_odd_plane_fraction",
    "source_antisymmetric_planes",
    "source_symmetric_planes",
    "source_first_symmetric_planes",
    "source_second_symmetric_planes",
    "source_n_planes",
    "source_dimension",
    "source_hidden_dim",
    "source_lambda_start_frac",
    "source_lambda_block_cca",
    "seed",
    "source_run_dir",
    "embedder_init",
    "model",
    "feature_layer",
    "feature_dim",
    "metrics_path",
    "log_path",
    *METRIC_FIELDS,
]


def metrics_path_for(run_dir: Path, suffix: str, horizon_ms: int, feature_layer: str) -> Path:
    feature_suffix = "_hidden" if feature_layer == "hidden" else ""
    return run_dir / "outputs" / f"velocity_prediction_finetune_{suffix}_{horizon_ms}ms{feature_suffix}" / "metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_folder", type=Path)
    parser.add_argument("--condition", type=int, default=1, help="1-based condition group from sweep results.csv.")
    parser.add_argument("--list-conditions", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--feature-layer", choices=["output", "hidden"], default="hidden")
    parser.add_argument("--horizon-ms", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--velocity-scale", choices=["stored", "si"], default="stored")
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--trial-batch-size", type=int, default=64)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--mlp-dropout", type=float, default=0.3)
    parser.add_argument("--decoder-lr", type=float, default=1e-3)
    parser.add_argument("--embedder-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-trials", type=int, default=0)
    parser.add_argument("--max-val-trials", type=int, default=0)
    parser.add_argument("--max-plot-points", type=int, default=5000)
    args = parser.parse_args()

    sweep_folder = args.sweep_folder.resolve()
    source_csv = sweep_folder / "results.csv"
    if not source_csv.is_file():
        raise FileNotFoundError(f"missing sweep results CSV: {source_csv}")

    if args.list_conditions:
        print_condition_listing(source_csv)
        return

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else sweep_folder / f"finetune_init_comparison_condition{args.condition}"
    )
    summary_csv = output_dir / "results.csv"
    ci_csv = output_dir / "results_ci95.csv"
    paired_csv = output_dir / "paired_ttests.csv"
    log_dir = output_dir / "logs"

    if args.aggregate_only:
        write_ci95_summary(summary_csv, ci_csv, METRIC_FIELDS)
        write_paired_summary(summary_csv, paired_csv, METRIC_FIELDS)
        print(f"Wrote 95% CI summary to {ci_csv}")
        print(f"Wrote paired t-test summary to {paired_csv}")
        return

    source_rows = select_condition_rows(source_csv, args.condition)
    completed = existing_completed(summary_csv) if args.resume else set()
    print(f"Selected condition {args.condition}: {source_rows[0].get('condition', '')}")
    print(f"Seed runs: {len(source_rows)}")

    for source_row in source_rows:
        run_dir = Path(source_row["run_dir"])
        seed = source_row.get("seed", "")
        if not run_dir.is_dir():
            raise FileNotFoundError(f"missing run_dir from sweep row: {run_dir}")
        for init in ("pretrained", "random"):
            suffix = f"initcmp_c{args.condition}_s{seed}_{init}"
            metrics_path = metrics_path_for(run_dir, suffix, args.horizon_ms, args.feature_layer)
            expected_model = "finetuned_mlp"
            key = (str(run_dir), str(seed), init)
            log_path = log_dir / f"{run_dir.name}_{suffix}.log"

            if not (args.resume and key in completed):
                if not metrics_complete(metrics_path, expected_model):
                    cmd = [
                        sys.executable,
                        "-u",
                        "decoders/predict_mcmaze_velocity_finetune.py",
                        "--run",
                        str(run_dir),
                        "--embedder-init",
                        init,
                        "--output-suffix",
                        suffix,
                        "--horizon-ms",
                        str(args.horizon_ms),
                        "--velocity-scale",
                        args.velocity_scale,
                        "--embed-batch-size",
                        str(args.embed_batch_size),
                        "--feature-layer",
                        args.feature_layer,
                        "--epochs",
                        str(args.epochs),
                        "--trial-batch-size",
                        str(args.trial_batch_size),
                        "--mlp-hidden-dim",
                        str(args.mlp_hidden_dim),
                        "--mlp-depth",
                        str(args.mlp_depth),
                        "--mlp-dropout",
                        str(args.mlp_dropout),
                        "--decoder-lr",
                        str(args.decoder_lr),
                        "--embedder-lr",
                        str(args.embedder_lr),
                        "--weight-decay",
                        str(args.weight_decay),
                        "--max-train-trials",
                        str(args.max_train_trials),
                        "--max-val-trials",
                        str(args.max_val_trials),
                        "--max-plot-points",
                        str(args.max_plot_points),
                        "--seed",
                        str(seed),
                    ]
                    run_logged(cmd, log_path, dry_run=args.dry_run)
                else:
                    print(f"Using existing metrics: {metrics_path}")

            if args.dry_run:
                continue
            row = metric_row(metrics_path, expected_model, init)
            summary_row = {
                **copied_source_fields(source_row),
                "seed": seed,
                "source_run_dir": str(run_dir),
                "embedder_init": init,
                "model": row.get("model", ""),
                "feature_layer": row.get("feature_layer", ""),
                "feature_dim": row.get("feature_dim", ""),
                "metrics_path": str(metrics_path),
                "log_path": str(log_path),
                **{metric: row.get(metric, "") for metric in METRIC_FIELDS},
            }
            if not (args.resume and key in completed):
                append_summary_rows(summary_csv, FIELDNAMES, [summary_row])

    if not args.dry_run:
        write_ci95_summary(summary_csv, ci_csv, METRIC_FIELDS)
        write_paired_summary(summary_csv, paired_csv, METRIC_FIELDS)
        print(f"Wrote results to {summary_csv}")
        print(f"Wrote 95% CI summary to {ci_csv}")
        print(f"Wrote paired t-test summary to {paired_csv}")


if __name__ == "__main__":
    main()
