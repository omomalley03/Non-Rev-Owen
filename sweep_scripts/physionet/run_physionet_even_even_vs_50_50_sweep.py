"""Run PhysioNetMI even/even vs 50/50 even/odd two-branch seed sweeps.

The script sources ``physionetmi_config.sh`` as the baseline and changes only:

* ``SEED`` for each seed,
* ``TEMPORAL_FRONTEND`` for even/even vs mixed even/odd,
* ``ANTISYMMETRIC_PLANES`` for the mixed even/odd condition.

Both conditions use the same two-branch architecture. Each branch has hidden
layers of size ``HIDDEN_DIM`` from ``physionetmi_config.sh`` and emits half the
total embedding planes. In the even/even condition, the lower branch is
symmetric instead of antisymmetric; the two branch outputs are concatenated and
trained with the normal loss and cross-plane regularisation.
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
from run_physionet_dim_train_finetune_sweep import FIELDNAMES as BASE_FIELDNAMES, decoder_args


DEFAULT_SEEDS = (0, 1, 2, 3, 4)
OUT_DIR = REPO_ROOT / "physionetmi" / "even_even_vs_50_50_parity_sweep"
LOG_DIR = OUT_DIR / "logs"
SUMMARY_CSV = OUT_DIR / "results.csv"
CI_CSV = OUT_DIR / "results_ci95.csv"
TTEST_CSV = OUT_DIR / "paired_ttests.csv"

CONDITION_FIELDS = [
    "condition",
    "temporal_frontend",
    "odd_plane_fraction",
    "antisymmetric_planes",
    "symmetric_planes",
    "first_symmetric_planes",
    "second_symmetric_planes",
    "n_planes",
    "dimension",
    "seed",
    "hidden_dim",
    "base_hidden_dim",
    "expected_hidden_decoder_feature_dim",
]
FIELDNAMES = [
    *CONDITION_FIELDS,
    *[
        field
        for field in BASE_FIELDNAMES
        if field not in {"dimension", "seed", "antisymmetric_planes"}
    ],
]

GROUP_FIELDS = (
    "condition",
    "temporal_frontend",
    "odd_plane_fraction",
    "antisymmetric_planes",
    "symmetric_planes",
    "first_symmetric_planes",
    "second_symmetric_planes",
    "dimension",
    "hidden_dim",
    "lambda_start_frac",
    "lambda_block_cca",
)
COMPLETED_KEY_FIELDS = (
    "condition",
    "temporal_frontend",
    "dimension",
    "hidden_dim",
    "antisymmetric_planes",
    "seed",
)


def experiment_conditions(dim: int, base_hidden_dim: int) -> list[dict[str, object]]:
    if dim % 2 != 0:
        raise ValueError(f"mixed_parity requires an even embedding dimension, got D={dim}")
    n_planes = dim // 2
    if n_planes < 2:
        raise ValueError(f"Need at least two 2D planes for a 50/50 split, got D={dim}")

    half_odd = n_planes // 2
    first_symmetric_planes = n_planes // 2
    second_symmetric_planes = n_planes - first_symmetric_planes
    return [
        {
            "condition": "dual_symmetric",
            "temporal_frontend": "dual_symmetric",
            "odd_plane_fraction": 0.0,
            "antisymmetric_planes": 0,
            "symmetric_planes": n_planes,
            "first_symmetric_planes": first_symmetric_planes,
            "second_symmetric_planes": second_symmetric_planes,
            "hidden_dim": base_hidden_dim,
            "expected_hidden_decoder_feature_dim": 2 * base_hidden_dim,
        },
        {
            "condition": "even_odd_50_50",
            "temporal_frontend": "mixed_parity",
            "odd_plane_fraction": half_odd / n_planes,
            "antisymmetric_planes": half_odd,
            "symmetric_planes": n_planes - half_odd,
            "first_symmetric_planes": n_planes - half_odd,
            "second_symmetric_planes": 0,
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
    dual_symmetric = experiment_conditions(dim, base_hidden_dim)[0]
    return {
        "condition": dual_symmetric["condition"],
        "temporal_frontend": dual_symmetric["temporal_frontend"],
        "odd_plane_fraction": dual_symmetric["odd_plane_fraction"],
        "antisymmetric_planes": dual_symmetric["antisymmetric_planes"],
        "symmetric_planes": dual_symmetric["symmetric_planes"],
        "first_symmetric_planes": dual_symmetric["first_symmetric_planes"],
        "second_symmetric_planes": dual_symmetric["second_symmetric_planes"],
        "dimension": dim,
        "hidden_dim": dual_symmetric["hidden_dim"],
        "lambda_start_frac": base_env.get("LAMBDA_START_FRAC", ""),
        "lambda_block_cca": base_env.get("LAMBDA_BLOCK_CCA", ""),
    }


def write_aggregates(args, base_env: dict[str, str], dim: int, base_hidden_dim: int) -> None:
    write_ci95_summary(
        args.results,
        args.ci_results,
        group_fields=GROUP_FIELDS,
        metric_fields=CONDITION_METRIC_FIELDS,
    )
    write_paired_ttest_summary(
        args.results,
        args.paired_ttest_results,
        group_fields=GROUP_FIELDS,
        baseline_group=baseline_group(base_env, dim, base_hidden_dim),
        metric_fields=CONDITION_METRIC_FIELDS,
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
        help="Embedding dimension D. Defaults to D from physionetmi_config.sh.",
    )
    parser.add_argument(
        "--base-hidden-dim",
        type=int,
        default=None,
        help="Base HIDDEN_DIM. Defaults to HIDDEN_DIM from physionetmi_config.sh.",
    )
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
            f"{c['condition']} frontend={c['temporal_frontend']} "
            f"ap={c['antisymmetric_planes']} hidden={c['hidden_dim']}"
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
        env["TEMPORAL_FRONTEND"] = str(condition["temporal_frontend"])
        env["ANTISYMMETRIC_PLANES"] = str(condition["antisymmetric_planes"])
        env["HIDDEN_DIM"] = str(condition["hidden_dim"])
        env["PYTHONUNBUFFERED"] = "1"

        train_log = LOG_DIR / f"{label}_main_synth.log"
        finetune_log = LOG_DIR / f"{label}_condition_finetune.log"
        print(
            f"[{idx}/{len(planned)}] run {label} "
            f"(condition={condition['condition']}, hidden_dim={condition['hidden_dim']})",
            flush=True,
        )
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
            **condition,
            "n_planes": dim // 2,
            "seed": seed,
            "base_hidden_dim": base_hidden_dim,
            "lambda_start_frac": env.get("LAMBDA_START_FRAC", ""),
            "lambda_block_cca": env.get("LAMBDA_BLOCK_CCA", ""),
            "run_dir": str(run_dir),
            "train_log": str(train_log),
            "finetune_log": str(finetune_log),
            "finetune_metrics": str(metrics_path),
            **embedding_summary_metrics(run_dir),
            **decoder_summary_metrics(metric_rows),
        }
        append_rows(args.results, FIELDNAMES, [summary_row])

    if not args.dry_run:
        write_aggregates(args, base_env, dim, base_hidden_dim)


if __name__ == "__main__":
    main()
