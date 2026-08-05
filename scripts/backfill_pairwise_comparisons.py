#!/usr/bin/env python3
"""Write paired-seed comparison CSVs for all existing sweep results.

This scans PhysioNetMI and MC Maze sweep folders for ``results.csv`` files and
creates ``pairwise_comparisons.csv`` next to each file when the rows can be
paired by ``seed``.
"""

from __future__ import annotations

from pathlib import Path

from pairwise_compare_results import (
    DEFAULT_METRIC_CANDIDATES,
    build_comparisons,
    config_key,
    infer_config_columns,
    present_columns,
    read_rows,
    write_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_GLOBS = (
    "physionetmi/**/results.csv",
    "mcmaze/**/results.csv",
)

CONFIG_COLUMN_OVERRIDES = {
    Path("physionetmi/odd_plane_fraction_finetune_sweep/results.csv"): ["odd_plane_fraction"],
    Path("physionetmi/odd_plane_fraction_finetune_sweep_v2/results.csv"): ["odd_plane_fraction"],
    Path("mcmaze/odd_plane_fraction_finetune_sweep/results.csv"): ["odd_plane_fraction"],
}


def discover_results() -> list[Path]:
    paths = []
    for pattern in RESULTS_GLOBS:
        paths.extend(REPO_ROOT.glob(pattern))
    return sorted(set(paths))


def empty_summary(path: Path) -> None:
    write_summary(path, [])


def main() -> None:
    processed = 0
    skipped = 0
    for results_csv in discover_results():
        rel_path = results_csv.relative_to(REPO_ROOT)
        rows = read_rows(results_csv)
        out_path = results_csv.with_name("pairwise_comparisons.csv")

        if not rows:
            empty_summary(out_path)
            print(f"skip empty: {rel_path}")
            skipped += 1
            continue

        fieldnames = rows[0].keys()
        if "seed" not in fieldnames:
            empty_summary(out_path)
            print(f"skip no seed: {rel_path}")
            skipped += 1
            continue

        metrics = present_columns(rows, DEFAULT_METRIC_CANDIDATES)
        if not metrics:
            empty_summary(out_path)
            print(f"skip no known metrics: {rel_path}")
            skipped += 1
            continue

        config_cols = CONFIG_COLUMN_OVERRIDES.get(
            rel_path,
            infer_config_columns(rows, metrics, ["seed"]),
        )
        if not config_cols:
            empty_summary(out_path)
            print(f"skip one config/no inferred config cols: {rel_path}")
            skipped += 1
            continue

        configs = {config_key(row, config_cols) for row in rows}
        if len(configs) < 2:
            empty_summary(out_path)
            print(f"skip one config: {rel_path}")
            skipped += 1
            continue

        try:
            comparisons = build_comparisons(rows, config_cols, ["seed"], metrics)
        except ValueError as exc:
            empty_summary(out_path)
            print(f"skip duplicate pair keys: {rel_path}: {exc}")
            skipped += 1
            continue

        write_summary(out_path, comparisons)
        print(
            f"wrote {out_path.relative_to(REPO_ROOT)} "
            f"({len(rows)} rows, {len(configs)} configs, {len(comparisons)} comparisons)"
        )
        processed += 1

    print(f"Processed {processed}; skipped {skipped}.")


if __name__ == "__main__":
    main()
