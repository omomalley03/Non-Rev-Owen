"""Compare PhysioNetMI fine-tuning from pretrained vs random embedder init.

Pass a completed seed-sweep folder. The script reads ``results.csv``, selects
one condition group by 1-based index, and runs
``predict_physionet_condition_finetune.py`` for every seed run in that condition
with:

* ``--embedder-init pretrained``: start from the frozen checkpoint and initialize
  the fine-tune decoder from the frozen decoder baseline.
* ``--embedder-init random``: train the embedder and decoder end-to-end from
  random initialization.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from finetune_init_comparison_common import (
    append_summary_rows,
    copied_source_fields,
    existing_completed,
    metrics_complete,
    metric_row,
    print_condition_listing,
    run_logged,
    select_condition_rows,
    write_ci95_summary,
    write_paired_summary,
)


METRIC_FIELDS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "best_val_acc",
    "best_val_ce",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_f1",
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
    "flat_feature_dim",
    "decoder_type",
    "metrics_path",
    "log_path",
    *METRIC_FIELDS,
]


def metrics_path_for(run_dir: Path, suffix: str, init: str, feature_layer: str) -> Path:
    out_name = f"condition_prediction_finetune_{suffix}"
    if init == "random":
        out_name += "_random_init"
    if feature_layer == "hidden":
        out_name += "_hidden"
    return run_dir / "outputs" / out_name / "metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_folder", type=Path)
    parser.add_argument("--condition", type=int, default=1, help="1-based condition group from sweep results.csv.")
    parser.add_argument("--list-conditions", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--data", default=None, help="Override synthetic data .npy path from checkpoint config.")
    parser.add_argument("--decoder-split", choices=["checkpoint", "random", "subject_random"], default="checkpoint")
    parser.add_argument("--feature-layer", choices=["output", "hidden"], default="hidden")
    parser.add_argument("--decoder-type", choices=["mlp", "temporal_conv"], default="temporal_conv")
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--frozen-epochs", type=int, default=0)
    parser.add_argument("--trial-batch-size", type=int, default=64)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--mlp-dropout", type=float, default=0.2)
    parser.add_argument("--conv-hidden-dim", type=int, default=32)
    parser.add_argument("--conv-depth", type=int, default=2)
    parser.add_argument("--conv-kernel-size", type=int, default=31)
    parser.add_argument("--conv-dropout", type=float, default=0.6)
    parser.add_argument("--decoder-lr", type=float, default=1e-3)
    parser.add_argument("--embedder-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-plot-points", type=int, default=5000)
    parser.add_argument("--no-standardize", action="store_true")
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
            suffix = (
                f"initcmp_c{args.condition}_s{seed}_pretrained"
                if init == "pretrained"
                else f"initcmp_c{args.condition}_s{seed}_e2e"
            )
            metrics_path = metrics_path_for(run_dir, suffix, init, args.feature_layer)
            expected_model = f"finetuned_{args.decoder_type}"
            key = (str(run_dir), str(seed), init)
            log_path = log_dir / f"{run_dir.name}_{suffix}_{init}.log"

            if not (args.resume and key in completed):
                if not metrics_complete(metrics_path, expected_model):
                    cmd = [
                        sys.executable,
                        "-u",
                        "predict_physionet_condition_finetune.py",
                        "--run",
                        str(run_dir),
                        "--embedder-init",
                        init,
                        "--feature-layer",
                        args.feature_layer,
                        "--decoder-type",
                        args.decoder_type,
                        "--output-suffix",
                        suffix,
                        "--embed-batch-size",
                        str(args.embed_batch_size),
                        "--epochs",
                        str(args.epochs),
                        "--frozen-epochs",
                        str(args.frozen_epochs),
                        "--trial-batch-size",
                        str(args.trial_batch_size),
                        "--mlp-hidden-dim",
                        str(args.mlp_hidden_dim),
                        "--mlp-depth",
                        str(args.mlp_depth),
                        "--mlp-dropout",
                        str(args.mlp_dropout),
                        "--conv-hidden-dim",
                        str(args.conv_hidden_dim),
                        "--conv-depth",
                        str(args.conv_depth),
                        "--conv-kernel-size",
                        str(args.conv_kernel_size),
                        "--conv-dropout",
                        str(args.conv_dropout),
                        "--decoder-lr",
                        str(args.decoder_lr),
                        "--embedder-lr",
                        str(args.embedder_lr),
                        "--weight-decay",
                        str(args.weight_decay),
                        "--batch-size",
                        str(args.batch_size),
                        "--max-plot-points",
                        str(args.max_plot_points),
                        "--seed",
                        str(seed),
                    ]
                    if args.data is not None:
                        cmd.extend(["--data", args.data])
                    if args.decoder_split != "checkpoint":
                        cmd.extend(["--decoder-split", args.decoder_split])
                    if args.no_standardize:
                        cmd.append("--no-standardize")
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
                "flat_feature_dim": row.get("flat_feature_dim", ""),
                "decoder_type": row.get("decoder_type", ""),
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
