"""Run PhysioNetMI train + frozen condition-decoder sweeps over embedding dimension.

Each job sources ``physionetmi_config.sh`` as the baseline, overrides ``D`` and
``SEED``, runs ``main_synth.py``, then runs
``predict_physionet_condition_finetune.py`` on the newly created run using the
same seed.

Usage:
    source /home/omo26/Non-Rev-Owen/.venv/bin/activate
    python -u sweep_scripts/physionet/run_physionet_dim_train_finetune_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model.py").is_file())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
from pathlib import Path

from sweep_scripts.common.physionet_train_finetune_common import (
    CONDITION_METRIC_FIELDS,
    DEFAULT_CONFIG,
    REPO_ROOT,
    append_rows,
    decoder_summary_metrics,
    default_finetune_metrics_path,
    embedding_summary_metrics,
    load_completed,
    read_finetune_metric_rows,
    run_main_synth_then_finetune,
    source_physionet_config,
    write_ci95_summary,
    write_paired_ttest_summary,
)


DEFAULT_DIMS = (64, 128, 256, 512)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
OUT_DIR = REPO_ROOT / "physionetmi" / "dim_train_finetune_sweep_v2"
LOG_DIR = OUT_DIR / "logs"
SUMMARY_CSV = OUT_DIR / "results.csv"
CI_CSV = OUT_DIR / "results_ci95.csv"
TTEST_CSV = OUT_DIR / "paired_ttests.csv"

FIELDNAMES = [
    "dimension",
    "seed",
    "lambda_start_frac",
    "lambda_block_cca",
    "antisymmetric_planes",
    "run_dir",
    "saved_checkpoint_epoch",
    "checkpoint_selection",
    "checkpoint_metric",
    "checkpoint_score",
    "checkpoint_lambda_scale",
    "checkpoint_regularization_lambda",
    "embedding_val_zeta",
    "embedding_val_mean_plane_zeta",
    "embedding_val_s",
    "embedding_val_c_plus",
    "embedding_val_loss",
    "validation_zeta",
    "validation_loss",
    "validation_c_plus",
    "regularization_whole_batch_total_raw",
    "regularization_whole_batch_total_scaled",
    "train_log",
    "finetune_log",
    "finetune_metrics",
    "frozen_decoder_feature_layer",
    "frozen_decoder_feature_dim",
    "frozen_decoder_flat_feature_dim",
    "frozen_decoder_decoder_type",
    "frozen_decoder_accuracy",
    "frozen_decoder_balanced_accuracy",
    "frozen_decoder_macro_f1",
    "frozen_decoder_best_val_acc",
    "frozen_decoder_best_val_ce",
    "frozen_decoder_test_accuracy",
    "frozen_decoder_test_balanced_accuracy",
    "frozen_decoder_test_macro_f1",
]


def decoder_args(args: argparse.Namespace) -> list[str]:
    return [
        "--epochs",
        str(args.decoder_epochs),
        "--decoder-lr",
        str(args.decoder_lr),
        "--weight-decay",
        str(args.decoder_weight_decay),
        "--conv-hidden-dim",
        str(args.conv_hidden_dim),
        "--conv-depth",
        str(args.conv_depth),
        "--conv-kernel-size",
        str(args.conv_kernel_size),
        "--conv-dropout",
        str(args.conv_dropout),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=list(DEFAULT_DIMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--decoder-epochs", type=int, default=50)
    parser.add_argument("--decoder-lr", type=float, default=1e-3)
    parser.add_argument("--decoder-weight-decay", type=float, default=1e-3)
    parser.add_argument("--conv-hidden-dim", type=int, default=64)
    parser.add_argument("--conv-depth", type=int, default=2)
    parser.add_argument("--conv-kernel-size", type=int, default=31)
    parser.add_argument("--conv-dropout", type=float, default=0.4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--results", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--ci-results", type=Path, default=CI_CSV)
    parser.add_argument("--paired-ttest-results", type=Path, default=TTEST_CSV)
    args = parser.parse_args()

    base_env = source_physionet_config(args.config)
    baseline_group = {
        "dimension": args.dims[0],
        "lambda_start_frac": base_env.get("LAMBDA_START_FRAC", ""),
        "lambda_block_cca": base_env.get("LAMBDA_BLOCK_CCA", ""),
        "antisymmetric_planes": base_env.get("ANTISYMMETRIC_PLANES", ""),
    }

    if args.aggregate_only:
        write_ci95_summary(
            args.results,
            args.ci_results,
            group_fields=("dimension", "lambda_start_frac", "lambda_block_cca", "antisymmetric_planes"),
            metric_fields=CONDITION_METRIC_FIELDS,
        )
        write_paired_ttest_summary(
            args.results,
            args.paired_ttest_results,
            group_fields=("dimension", "lambda_start_frac", "lambda_block_cca", "antisymmetric_planes"),
            baseline_group=baseline_group,
            metric_fields=CONDITION_METRIC_FIELDS,
        )
        print(f"Wrote 95% CI summary to {args.ci_results}")
        print(f"Wrote paired t-test summary to {args.paired_ttest_results}")
        return

    completed = load_completed(args.results, ("dimension", "seed")) if args.resume else set()

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

        train_log = LOG_DIR / f"{label}_main_synth.log"
        finetune_log = LOG_DIR / f"{label}_condition_finetune.log"
        print(f"[{idx}/{len(planned)}] run {label}", flush=True)
        run_dir = run_main_synth_then_finetune(
            env,
            train_log,
            finetune_log,
            output_suffix=label,
            seed=seed,
            decoder_args=decoder_args(args),
            dry_run=args.dry_run,
        )
        if args.dry_run or run_dir is None:
            continue

        metrics_path = default_finetune_metrics_path(run_dir, label)
        metric_rows = read_finetune_metric_rows(metrics_path)
        summary_row = {
            "dimension": dim,
            "seed": seed,
            "lambda_start_frac": env.get("LAMBDA_START_FRAC", ""),
            "lambda_block_cca": env.get("LAMBDA_BLOCK_CCA", ""),
            "antisymmetric_planes": env.get("ANTISYMMETRIC_PLANES", ""),
            "run_dir": str(run_dir),
            "train_log": str(train_log),
            "finetune_log": str(finetune_log),
            "finetune_metrics": str(metrics_path),
            **embedding_summary_metrics(run_dir),
            **decoder_summary_metrics(metric_rows),
        }
        append_rows(args.results, FIELDNAMES, [summary_row])

    if not args.dry_run:
        write_ci95_summary(
            args.results,
            args.ci_results,
            group_fields=("dimension", "lambda_start_frac", "lambda_block_cca", "antisymmetric_planes"),
            metric_fields=CONDITION_METRIC_FIELDS,
        )
        write_paired_ttest_summary(
            args.results,
            args.paired_ttest_results,
            group_fields=("dimension", "lambda_start_frac", "lambda_block_cca", "antisymmetric_planes"),
            baseline_group=baseline_group,
            metric_fields=CONDITION_METRIC_FIELDS,
        )
        print(f"Wrote 95% CI summary to {args.ci_results}")
        print(f"Wrote paired t-test summary to {args.paired_ttest_results}")


if __name__ == "__main__":
    main()
