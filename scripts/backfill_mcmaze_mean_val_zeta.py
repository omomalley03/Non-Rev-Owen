"""Backfill mean_val_zeta into MC Maze sweep CSVs.

mean_val_zeta is the unweighted mean native-plane zeta computed once over the
whole validation split for each run's checkpoints/best.pt.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from traceback import format_exception_only


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = REPO_ROOT / "sweep_scripts"
for path in (REPO_ROOT, SWEEP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sweep_scripts.common.mcmaze_train_finetune_common import (  # noqa: E402
    TRIAL_SUMMARY_METRIC_FIELDS,
    compute_whole_validation_mean_plane_zeta,
    write_ci95_summary,
)


CI_GROUPS = {
    "sweep_results/hyperparam_sweep_paper_logs/mcmaze/cca_sweep_constant_lambda": (
        "lambda_start_frac",
        "lambda_block_cca",
        "dimension",
    ),
    "mcmaze/cca_warmup_finetune_sweep": (
        "lambda_start_frac",
        "lambda_block_cca",
        "dimension",
    ),
    "mcmaze/zeta_plane_mean": (
        "lambda_start_frac",
        "lambda_block_cca",
        "dimension",
    ),
    "mcmaze/dim_train_finetune_sweep": (
        "dimension",
        "lambda_start_frac",
        "lambda_block_cca",
    ),
    "sweep_results/hyperparam_sweep_paper_logs/mcmaze/dim_train_finetune_sweep_mean_plane_zeta": (
        "dimension",
        "lambda_start_frac",
        "lambda_block_cca",
    ),
    "mcmaze/odd_plane_fraction_finetune_sweep": (
        "odd_plane_fraction",
        "antisymmetric_planes",
        "symmetric_planes",
        "dimension",
        "lambda_start_frac",
        "lambda_block_cca",
    ),
}

RESULT_CSVS = (
    *tuple(Path(key) / "results.csv" for key in CI_GROUPS),
    Path("mcmaze/embedding_dim_seed_sweep/results.csv"),
    Path("mcmaze/embedding_dim_vs_rmse_experiment/nonlinear_multiscale_zeta/results.csv"),
)


def insert_after(fieldnames: list[str], new_field: str, anchors: tuple[str, ...]) -> list[str]:
    if new_field in fieldnames:
        return fieldnames
    for anchor in anchors:
        if anchor in fieldnames:
            index = fieldnames.index(anchor) + 1
            return [*fieldnames[:index], new_field, *fieldnames[index:]]
    return [*fieldnames, new_field]


def row_run_dir(row: dict[str, str]) -> Path | None:
    for key in ("run_dir", "run"):
        value = row.get(key)
        if value:
            return Path(value)
    return None


def backfill_results_csv(path: Path, force: bool = False) -> int:
    if not path.is_file():
        return 0

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        return 0

    fieldnames = insert_after(
        fieldnames,
        "mean_val_zeta",
        ("embedding_val_zeta", "best_val_zeta"),
    )

    changed = 0
    cache: dict[Path, object] = {}
    for row in rows:
        if row.get("mean_val_zeta") and not force:
            continue
        run_dir = row_run_dir(row)
        if run_dir is None:
            row["mean_val_zeta"] = ""
            continue
        if run_dir not in cache:
            try:
                cache[run_dir] = compute_whole_validation_mean_plane_zeta(run_dir)
            except Exception as exc:  # Historical checkpoints may not match current model code.
                message = "".join(format_exception_only(type(exc), exc)).strip()
                print(f"{path}: skipped {run_dir}: {message}", file=sys.stderr)
                cache[run_dir] = ""
        value = cache[run_dir]
        row["mean_val_zeta"] = "" if value == "" else f"{float(value):.10g}"
        changed += 1

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return changed


def regenerate_ci95(results_csv: Path) -> None:
    rel_parent = results_csv.parent.relative_to(REPO_ROOT).as_posix()
    group_fields = CI_GROUPS.get(rel_parent)
    if group_fields is None:
        return
    write_ci95_summary(
        source_csv=results_csv,
        ci_csv=results_csv.parent / "results_ci95.csv",
        group_fields=group_fields,
        metric_fields=TRIAL_SUMMARY_METRIC_FIELDS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute mean_val_zeta even when the column is already populated.",
    )
    args = parser.parse_args()

    for rel_path in RESULT_CSVS:
        path = REPO_ROOT / rel_path
        changed = backfill_results_csv(path, force=args.force)
        if path.is_file():
            regenerate_ci95(path)
            print(f"{rel_path}: updated {changed} rows")


if __name__ == "__main__":
    main()
