"""Run MC Maze train + velocity fine-tune sweeps over embedding dimension.

Each job sources ``mcmaze_config.sh`` as the baseline, overrides ``D`` and
``SEED``, runs ``main.py``, then runs ``predict_mcmaze_velocity_finetune.py`` on
the newly created run using that same seed and the fine-tune script defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mcmaze_train_finetune_common import (
    REPO_ROOT,
    TRIAL_SUMMARY_METRIC_FIELDS,
    append_rows,
    decoder_summary_metrics,
    default_finetune_metrics_path,
    embedding_summary_metrics,
    load_completed,
    read_finetune_metric_rows,
    run_main_then_finetune,
    source_mcmaze_config,
    write_ci95_summary,
    write_paired_ttest_summary,
)


DEFAULT_DIMS = (128, 256)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
OUT_DIR = REPO_ROOT / "mcmaze" / "dim_train_finetune_sweep_mean_plane_zeta"
LOG_DIR = OUT_DIR / "logs"
SUMMARY_CSV = OUT_DIR / "results.csv"
CI_CSV = OUT_DIR / "results_ci95.csv"
TTEST_CSV = OUT_DIR / "paired_ttests.csv"

FIELDNAMES = [
    "dimension",
    "seed",
    "lambda_start_frac",
    "lambda_block_cca",
    "run_dir",
    "saved_checkpoint_epoch",
    "checkpoint_selection",
    "checkpoint_lambda_scale",
    "checkpoint_regularization_lambda",
    "embedding_val_zeta",
    "mean_val_zeta",
    "embedding_whole_val_regularization_raw",
    "train_log",
    "finetune_log",
    "finetune_metrics",
    "frozen_decoder_feature_layer",
    "frozen_decoder_feature_dim",
    "frozen_decoder_rmse_x",
    "frozen_decoder_rmse_y",
    "frozen_decoder_rmse_mean",
    "frozen_decoder_r2_x",
    "frozen_decoder_r2_y",
    "frozen_decoder_r2_mean",
    "finetuned_decoder_feature_layer",
    "finetuned_decoder_feature_dim",
    "finetuned_decoder_rmse_x",
    "finetuned_decoder_rmse_y",
    "finetuned_decoder_rmse_mean",
    "finetuned_decoder_r2_x",
    "finetuned_decoder_r2_y",
    "finetuned_decoder_r2_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=list(DEFAULT_DIMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--results", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--ci-results", type=Path, default=CI_CSV)
    parser.add_argument("--paired-ttest-results", type=Path, default=TTEST_CSV)
    args = parser.parse_args()

    base_env = source_mcmaze_config()
    baseline_group = {
        "dimension": args.dims[0],
        "lambda_start_frac": base_env.get("LAMBDA_START_FRAC", ""),
        "lambda_block_cca": base_env.get("LAMBDA_BLOCK_CCA", ""),
    }

    if args.aggregate_only:
        write_ci95_summary(
            args.results,
            args.ci_results,
            group_fields=("dimension", "lambda_start_frac", "lambda_block_cca"),
            metric_fields=TRIAL_SUMMARY_METRIC_FIELDS,
            stratify_fields=(),
        )
        write_paired_ttest_summary(
            args.results,
            args.paired_ttest_results,
            group_fields=("dimension", "lambda_start_frac", "lambda_block_cca"),
            baseline_group=baseline_group,
            metric_fields=TRIAL_SUMMARY_METRIC_FIELDS,
            stratify_fields=(),
        )
        print(f"Wrote 95% CI summary to {args.ci_results}")
        print(f"Wrote paired t-test summary to {args.paired_ttest_results}")
        return

    completed = (
        load_completed(args.results, ("dimension", "seed"))
        if args.resume
        else set()
    )

    planned = [(dim, seed) for dim in args.dims for seed in args.seeds]
    print(f"Planned jobs: {len(planned)}")

    for idx, (dim, seed) in enumerate(planned, 1):
        key = (str(dim), str(seed))
        label = f"d{dim}_s{seed}"
        if key in completed:
            print(f"[{idx}/{len(planned)}] skip complete {label}")
            continue

        env = base_env.copy()
        env["D"] = str(dim)
        env["SEED"] = str(seed)
        env["PYTHONUNBUFFERED"] = "1"

        train_log = LOG_DIR / f"{label}_main.log"
        finetune_log = LOG_DIR / f"{label}_finetune.log"
        print(f"[{idx}/{len(planned)}] run {label}", flush=True)
        run_dir = run_main_then_finetune(
            env,
            train_log,
            finetune_log,
            seed=seed,
            dry_run=args.dry_run,
        )
        if args.dry_run or run_dir is None:
            continue

        metric_rows = read_finetune_metric_rows(run_dir)
        summary_row = {
            "dimension": dim,
            "seed": seed,
            "lambda_start_frac": env.get("LAMBDA_START_FRAC", ""),
            "lambda_block_cca": env.get("LAMBDA_BLOCK_CCA", ""),
            "run_dir": str(run_dir),
            "train_log": str(train_log),
            "finetune_log": str(finetune_log),
            "finetune_metrics": str(default_finetune_metrics_path(run_dir)),
            **embedding_summary_metrics(run_dir),
            **decoder_summary_metrics(metric_rows),
        }
        append_rows(args.results, FIELDNAMES, [summary_row])

    if not args.dry_run:
        write_ci95_summary(
            args.results,
            args.ci_results,
            group_fields=("dimension", "lambda_start_frac", "lambda_block_cca"),
            metric_fields=TRIAL_SUMMARY_METRIC_FIELDS,
            stratify_fields=(),
        )
        write_paired_ttest_summary(
            args.results,
            args.paired_ttest_results,
            group_fields=("dimension", "lambda_start_frac", "lambda_block_cca"),
            baseline_group=baseline_group,
            metric_fields=TRIAL_SUMMARY_METRIC_FIELDS,
            stratify_fields=(),
        )
        print(f"Wrote 95% CI summary to {args.ci_results}")
        print(f"Wrote paired t-test summary to {args.paired_ttest_results}")


if __name__ == "__main__":
    main()
