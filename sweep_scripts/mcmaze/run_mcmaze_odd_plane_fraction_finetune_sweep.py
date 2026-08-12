"""Run MC Maze train + velocity fine-tune sweeps over odd-plane percentage.

Each job sources ``mcmaze_config.sh`` as the baseline, overrides ``SEED`` and
``ANTISYMMETRIC_PLANES``, runs ``main.py``, then runs
``predict_mcmaze_velocity_finetune.py`` on the newly created run using that same
seed and the fine-tune script defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model.py").is_file())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
from pathlib import Path

from sweep_scripts.common.mcmaze_train_finetune_common import (
    REPO_ROOT,
    TRIAL_SUMMARY_METRIC_FIELDS,
    antisymmetric_planes_from_fraction,
    append_rows,
    decoder_summary_metrics,
    default_finetune_metrics_path,
    embedding_summary_metrics,
    fmt_value,
    load_completed,
    read_finetune_metric_rows,
    run_main_then_finetune,
    source_mcmaze_config,
    write_ci95_summary,
    write_paired_ttest_summary,
)


DEFAULT_ODD_PERCENTAGES = (0.0,)
DEFAULT_SEEDS = (4,)
OUT_DIR = REPO_ROOT / "mcmaze" / "odd_plane_fraction_finetune_sweep"
LOG_DIR = OUT_DIR / "logs"
SUMMARY_CSV = OUT_DIR / "results.csv"
CI_CSV = OUT_DIR / "results_ci95.csv"
TTEST_CSV = OUT_DIR / "paired_ttests.csv"

FIELDNAMES = [
    "odd_plane_fraction",
    "odd_plane_percent",
    "antisymmetric_planes",
    "symmetric_planes",
    "n_planes",
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


def parse_percentage(value: float) -> float:
    """Accept either 0..1 fractions or 0..100 percentages and return a fraction."""
    if value < 0.0:
        raise argparse.ArgumentTypeError("odd-plane percentage must be nonnegative")
    if value <= 1.0:
        return value
    if value <= 100.0:
        return value / 100.0
    raise argparse.ArgumentTypeError("odd-plane percentage must be in [0, 100]")


def fraction_label(fraction: float) -> str:
    return fmt_value(f"{fraction:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--percentages",
        type=parse_percentage,
        nargs="+",
        default=[parse_percentage(value) for value in DEFAULT_ODD_PERCENTAGES],
        help=(
            "Odd-plane percentages to sweep. Values in 0..1 are treated as fractions; "
            "values above 1 are treated as percentages. Defaults to 0 25 50 75 100."
        ),
    )
    parser.add_argument(
        "--dimension",
        "--dim",
        type=int,
        default=None,
        help="Embedding dimension D. Defaults to D from mcmaze_config.sh.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--results", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--ci-results", type=Path, default=CI_CSV)
    parser.add_argument("--paired-ttest-results", type=Path, default=TTEST_CSV)
    args = parser.parse_args()

    base_env = source_mcmaze_config()
    dim = int(args.dimension if args.dimension is not None else base_env.get("D", "128"))
    if dim % 2 != 0:
        raise ValueError(f"mixed_parity requires an even embedding dimension, got D={dim}")
    baseline_fraction = args.percentages[0]
    baseline_antisymmetric_planes = antisymmetric_planes_from_fraction(dim, baseline_fraction)
    baseline_n_planes = dim // 2
    baseline_group = {
        "odd_plane_fraction": baseline_fraction,
        "antisymmetric_planes": baseline_antisymmetric_planes,
        "symmetric_planes": baseline_n_planes - baseline_antisymmetric_planes,
        "dimension": dim,
        "lambda_start_frac": base_env.get("LAMBDA_START_FRAC", ""),
        "lambda_block_cca": base_env.get("LAMBDA_BLOCK_CCA", ""),
    }

    if args.aggregate_only:
        write_ci95_summary(
            args.results,
            args.ci_results,
            group_fields=(
                "odd_plane_fraction",
                "antisymmetric_planes",
                "symmetric_planes",
                "dimension",
                "lambda_start_frac",
                "lambda_block_cca",
            ),
            metric_fields=TRIAL_SUMMARY_METRIC_FIELDS,
            stratify_fields=(),
        )
        write_paired_ttest_summary(
            args.results,
            args.paired_ttest_results,
            group_fields=(
                "odd_plane_fraction",
                "antisymmetric_planes",
                "symmetric_planes",
                "dimension",
                "lambda_start_frac",
                "lambda_block_cca",
            ),
            baseline_group=baseline_group,
            metric_fields=TRIAL_SUMMARY_METRIC_FIELDS,
            stratify_fields=(),
        )
        print(f"Wrote 95% CI summary to {args.ci_results}")
        print(f"Wrote paired t-test summary to {args.paired_ttest_results}")
        return

    completed = (
        load_completed(
            args.results,
            ("odd_plane_fraction", "antisymmetric_planes", "dimension", "seed"),
        )
        if args.resume
        else set()
    )

    planned = []
    for fraction in args.percentages:
        antisymmetric_planes = antisymmetric_planes_from_fraction(dim, fraction)
        n_planes = dim // 2
        symmetric_planes = n_planes - antisymmetric_planes
        planned.extend(
            (fraction, antisymmetric_planes, symmetric_planes, n_planes, seed)
            for seed in args.seeds
        )
    print(f"Planned jobs: {len(planned)}")

    for idx, (fraction, antisymmetric_planes, symmetric_planes, n_planes, seed) in enumerate(planned, 1):
        key = (str(fraction), str(antisymmetric_planes), str(dim), str(seed))
        label = (
            f"odd{fraction_label(fraction)}_"
            f"ap{antisymmetric_planes}_d{dim}_s{seed}"
        )
        if key in completed:
            print(f"[{idx}/{len(planned)}] skip complete {label}")
            continue

        env = base_env.copy()
        env["D"] = str(dim)
        env["SEED"] = str(seed)
        env["ANTISYMMETRIC_PLANES"] = str(antisymmetric_planes)
        env["PYTHONUNBUFFERED"] = "1"

        train_log = LOG_DIR / f"{label}_main.log"
        finetune_log = LOG_DIR / f"{label}_finetune.log"
        print(
            f"[{idx}/{len(planned)}] run {label} "
            f"({100.0 * fraction:.1f}% odd planes)",
            flush=True,
        )
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
            "odd_plane_fraction": fraction,
            "odd_plane_percent": 100.0 * fraction,
            "antisymmetric_planes": antisymmetric_planes,
            "symmetric_planes": symmetric_planes,
            "n_planes": n_planes,
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
            group_fields=(
                "odd_plane_fraction",
                "antisymmetric_planes",
                "symmetric_planes",
                "dimension",
                "lambda_start_frac",
                "lambda_block_cca",
            ),
            metric_fields=TRIAL_SUMMARY_METRIC_FIELDS,
            stratify_fields=(),
        )
        write_paired_ttest_summary(
            args.results,
            args.paired_ttest_results,
            group_fields=(
                "odd_plane_fraction",
                "antisymmetric_planes",
                "symmetric_planes",
                "dimension",
                "lambda_start_frac",
                "lambda_block_cca",
            ),
            baseline_group=baseline_group,
            metric_fields=TRIAL_SUMMARY_METRIC_FIELDS,
            stratify_fields=(),
        )
        print(f"Wrote 95% CI summary to {args.ci_results}")
        print(f"Wrote paired t-test summary to {args.paired_ttest_results}")


if __name__ == "__main__":
    main()
