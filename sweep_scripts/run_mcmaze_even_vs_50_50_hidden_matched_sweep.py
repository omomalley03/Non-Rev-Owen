"""Run MC Maze all-even vs 50/50 even/odd sweeps with matched decoder inputs.

The script sources ``mcmaze_config.sh`` as the baseline and changes only:

* ``SEED`` for each seed,
* ``ANTISYMMETRIC_PLANES`` for all-even vs 50/50 even/odd,
* ``HIDDEN_DIM`` for all-even, doubled so the hidden-feature decoder input
  width matches the 50/50 mixed-parity model.

With the default hidden decoder, a 50/50 mixed-parity model exposes one hidden
feature stack from the even branch and one from the odd branch. The all-even
model has only the even branch, so doubling its hidden layer size keeps the
decoder feature dimension comparable.
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


DEFAULT_SEEDS = (0, 1, 2, 3, 4)
OUT_DIR = REPO_ROOT / "mcmaze" / "even_vs_50_50_parity_hidden_matched_sweep"
LOG_DIR = OUT_DIR / "logs"
SUMMARY_CSV = OUT_DIR / "results.csv"
CI_CSV = OUT_DIR / "results_ci95.csv"
TTEST_CSV = OUT_DIR / "paired_ttests.csv"

FIELDNAMES = [
    "condition",
    "odd_plane_fraction",
    "antisymmetric_planes",
    "symmetric_planes",
    "n_planes",
    "dimension",
    "seed",
    "hidden_dim",
    "base_hidden_dim",
    "expected_hidden_decoder_feature_dim",
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

GROUP_FIELDS = (
    "condition",
    "odd_plane_fraction",
    "antisymmetric_planes",
    "symmetric_planes",
    "dimension",
    "hidden_dim",
    "lambda_start_frac",
    "lambda_block_cca",
)
COMPLETED_KEY_FIELDS = ("condition", "dimension", "hidden_dim", "antisymmetric_planes", "seed")


def experiment_conditions(dim: int, base_hidden_dim: int) -> list[dict[str, object]]:
    if dim % 2 != 0:
        raise ValueError(f"mixed_parity requires an even embedding dimension, got D={dim}")
    n_planes = dim // 2
    if n_planes < 2:
        raise ValueError(f"Need at least two 2D planes for a 50/50 split, got D={dim}")

    half_odd = n_planes // 2
    return [
        {
            "condition": "all_even",
            "odd_plane_fraction": 0.0,
            "antisymmetric_planes": 0,
            "symmetric_planes": n_planes,
            "hidden_dim": 2 * base_hidden_dim,
            "expected_hidden_decoder_feature_dim": 2 * base_hidden_dim,
        },
        {
            "condition": "even_odd_50_50",
            "odd_plane_fraction": half_odd / n_planes,
            "antisymmetric_planes": half_odd,
            "symmetric_planes": n_planes - half_odd,
            "hidden_dim": base_hidden_dim,
            "expected_hidden_decoder_feature_dim": 2 * base_hidden_dim,
        },
    ]


def label_for(condition: dict[str, object], seed: int) -> str:
    return (
        f"{condition['condition']}_"
        f"d{condition['dimension']}_"
        f"h{condition['hidden_dim']}_"
        f"ap{condition['antisymmetric_planes']}_"
        f"s{seed}"
    )


def baseline_group(base_env: dict[str, str], dim: int, base_hidden_dim: int) -> dict[str, object]:
    all_even = experiment_conditions(dim, base_hidden_dim)[0]
    return {
        "condition": all_even["condition"],
        "odd_plane_fraction": all_even["odd_plane_fraction"],
        "antisymmetric_planes": all_even["antisymmetric_planes"],
        "symmetric_planes": all_even["symmetric_planes"],
        "dimension": dim,
        "hidden_dim": all_even["hidden_dim"],
        "lambda_start_frac": base_env.get("LAMBDA_START_FRAC", ""),
        "lambda_block_cca": base_env.get("LAMBDA_BLOCK_CCA", ""),
    }


def write_aggregates(args, base_env: dict[str, str], dim: int, base_hidden_dim: int) -> None:
    write_ci95_summary(
        args.results,
        args.ci_results,
        group_fields=GROUP_FIELDS,
        metric_fields=TRIAL_SUMMARY_METRIC_FIELDS,
        stratify_fields=(),
    )
    write_paired_ttest_summary(
        args.results,
        args.paired_ttest_results,
        group_fields=GROUP_FIELDS,
        baseline_group=baseline_group(base_env, dim, base_hidden_dim),
        metric_fields=TRIAL_SUMMARY_METRIC_FIELDS,
        stratify_fields=(),
    )
    print(f"Wrote 95% CI summary to {args.ci_results}")
    print(f"Wrote paired t-test summary to {args.paired_ttest_results}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--dimension",
        "--dim",
        type=int,
        default=None,
        help="Embedding dimension D. Defaults to D from mcmaze_config.sh.",
    )
    parser.add_argument(
        "--base-hidden-dim",
        type=int,
        default=None,
        help="Base HIDDEN_DIM. Defaults to HIDDEN_DIM from mcmaze_config.sh.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--results", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--ci-results", type=Path, default=CI_CSV)
    parser.add_argument("--paired-ttest-results", type=Path, default=TTEST_CSV)
    args = parser.parse_args()

    base_env = source_mcmaze_config()
    dim = int(args.dimension if args.dimension is not None else base_env.get("D", "128"))
    base_hidden_dim = int(
        args.base_hidden_dim
        if args.base_hidden_dim is not None
        else base_env.get("HIDDEN_DIM", "256")
    )
    conditions = experiment_conditions(dim, base_hidden_dim)

    if args.aggregate_only:
        write_aggregates(args, base_env, dim, base_hidden_dim)
        return

    completed = (
        load_completed(args.results, COMPLETED_KEY_FIELDS)
        if args.resume
        else set()
    )

    planned = [
        ({**condition, "dimension": dim}, seed)
        for condition in conditions
        for seed in args.seeds
    ]
    print(f"Planned jobs: {len(planned)}")
    print(
        "Conditions: "
        + ", ".join(
            f"{c['condition']} ap={c['antisymmetric_planes']} hidden={c['hidden_dim']}"
            for c in conditions
        )
    )

    for idx, (condition, seed) in enumerate(planned, 1):
        key = tuple(
            str(
                {
                    **condition,
                    "seed": seed,
                }[field]
            )
            for field in COMPLETED_KEY_FIELDS
        )
        label = label_for(condition, seed)
        if key in completed:
            print(f"[{idx}/{len(planned)}] skip complete {label}")
            continue

        env = base_env.copy()
        env["D"] = str(dim)
        env["SEED"] = str(seed)
        env["ANTISYMMETRIC_PLANES"] = str(condition["antisymmetric_planes"])
        env["HIDDEN_DIM"] = str(condition["hidden_dim"])
        env["PYTHONUNBUFFERED"] = "1"

        train_log = LOG_DIR / f"{label}_main.log"
        finetune_log = LOG_DIR / f"{label}_finetune.log"
        print(
            f"[{idx}/{len(planned)}] run {label} "
            f"(condition={condition['condition']}, hidden_dim={condition['hidden_dim']})",
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
            **condition,
            "n_planes": dim // 2,
            "seed": seed,
            "base_hidden_dim": base_hidden_dim,
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
        write_aggregates(args, base_env, dim, base_hidden_dim)


if __name__ == "__main__":
    main()
