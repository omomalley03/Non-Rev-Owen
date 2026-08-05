#!/usr/bin/env python3
"""Paired comparisons between sweep configurations using a results.csv file.

Pairs are formed by matching rows with the same seed, then comparing every
configuration pair on the requested metrics.

Examples:
    python scripts/pairwise_compare_results.py physionetmi/cca_warmup_finetune_sweep_v2/results.csv
    python scripts/pairwise_compare_results.py physionetmi/odd_plane_fraction_finetune_sweep_v2/results.csv --config-cols odd_plane_fraction
    python scripts/pairwise_compare_results.py mcmaze/cca_warmup_finetune_sweep/results.csv --metrics embedding_val_zeta mean_val_zeta frozen_decoder_r2_mean
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_METRIC_CANDIDATES = (
    "best_val_zeta",
    "mean_val_zeta",
    "best_val_s",
    "best_val_c_plus",
    "final_val_zeta",
    "embedding_val_zeta",
    "embedding_val_mean_plane_zeta",
    "embedding_val_s",
    "embedding_val_c_plus",
    "embedding_whole_val_regularization_raw",
    "validation_zeta",
    "validation_c_plus",
    "regularization_whole_batch_total_raw",
    "frozen_decoder_accuracy",
    "frozen_decoder_balanced_accuracy",
    "frozen_decoder_test_accuracy",
    "frozen_decoder_test_balanced_accuracy",
    "frozen_decoder_r2_mean",
    "frozen_decoder_rmse_mean",
    "finetuned_decoder_r2_mean",
    "finetuned_decoder_rmse_mean",
)

DEFAULT_CONFIG_CANDIDATES = (
    "lambda_start_frac",
    "lambda_block_cca",
    "odd_plane_fraction",
    "antisymmetric_planes",
    "symmetric_planes",
    "n_planes",
    "dimension",
    "embedding_dim",
    "checkpoint_selection",
    "checkpoint_metric",
    "frozen_decoder_feature_layer",
    "frozen_decoder_decoder_type",
)

BOOKKEEPING_COLUMNS = {
    "seed",
    "run_dir",
    "train_log",
    "finetune_log",
    "finetune_metrics",
    "saved_checkpoint_epoch",
    "checkpoint_score",
    "checkpoint_lambda_scale",
    "checkpoint_regularization_lambda",
    "frozen_decoder_feature_dim",
    "frozen_decoder_flat_feature_dim",
    "finetuned_decoder_feature_dim",
}

T_CRITICAL_95 = {
    1: 12.706204736432095,
    2: 4.302652729749464,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.570581835636314,
    6: 2.4469118487916806,
    7: 2.3646242510102993,
    8: 2.3060041350333704,
    9: 2.2621571627409915,
    10: 2.2281388519649385,
    11: 2.200985160091638,
    12: 2.1788128296672303,
    13: 2.1603686564610127,
    14: 2.1447866879169273,
    15: 2.131449545559323,
    16: 2.119905299221011,
    17: 2.10981557783318,
    18: 2.10092204024096,
    19: 2.093024054408263,
    20: 2.085963447265837,
    21: 2.079613844727662,
    22: 2.073873067904015,
    23: 2.068657610419041,
    24: 2.063898561628021,
    25: 2.059538552753294,
    26: 2.055529438642871,
    27: 2.051830516480283,
    28: 2.048407141795244,
    29: 2.045229642132703,
    30: 2.042272456301238,
}


def parse_float(value: object) -> Optional[float]:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def format_value(value: object) -> str:
    text = str(value)
    value_float = parse_float(text)
    if value_float is None:
        return text
    if value_float.is_integer():
        return str(int(value_float))
    return f"{value_float:.10g}"


def format_number(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.10g}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def ci95_half_width(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    critical = T_CRITICAL_95.get(len(values) - 1, 1.959963984540054)
    return critical * stdev(values) / math.sqrt(len(values))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def present_columns(rows: list[dict[str, str]], candidates: Iterable[str]) -> list[str]:
    if not rows:
        return []
    fieldnames = set(rows[0])
    return [column for column in candidates if column in fieldnames]


def infer_config_columns(rows: list[dict[str, str]], metrics: list[str], pair_cols: list[str]) -> list[str]:
    candidates = present_columns(rows, DEFAULT_CONFIG_CANDIDATES)
    excluded = set(metrics) | set(pair_cols) | BOOKKEEPING_COLUMNS
    inferred = []
    for column in candidates:
        if column in excluded:
            continue
        values = {row.get(column, "") for row in rows}
        if len(values) > 1:
            inferred.append(column)
    if inferred:
        return inferred

    fallback = []
    for column in rows[0]:
        if column in excluded:
            continue
        values = {row.get(column, "") for row in rows}
        if 1 < len(values) <= max(2, len(rows) // 2):
            fallback.append(column)
    return fallback


def config_key(row: dict[str, str], config_cols: list[str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in config_cols)


def config_label(key: tuple[str, ...], config_cols: list[str]) -> str:
    return ",".join(
        f"{column}={format_value(value)}"
        for column, value in zip(config_cols, key)
    )


def pair_key(row: dict[str, str], pair_cols: list[str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in pair_cols)


def collect_by_config_and_pair(
    rows: list[dict[str, str]],
    config_cols: list[str],
    pair_cols: list[str],
) -> dict[tuple[str, ...], dict[tuple[str, ...], dict[str, str]]]:
    grouped: dict[tuple[str, ...], dict[tuple[str, ...], dict[str, str]]] = defaultdict(dict)
    for row in rows:
        c_key = config_key(row, config_cols)
        p_key = pair_key(row, pair_cols)
        # Appended sweep CSVs can contain reruns for a config/seed. Keep the
        # later row, matching resume/backfill behavior elsewhere in the repo.
        grouped[c_key][p_key] = row
    return grouped


def compare_metric(
    rows_a: dict[tuple[str, ...], dict[str, str]],
    rows_b: dict[tuple[str, ...], dict[str, str]],
    metric: str,
) -> Optional[dict[str, object]]:
    common_pairs = sorted(set(rows_a) & set(rows_b))
    values_a = []
    values_b = []
    diffs = []
    paired_ids = []
    for p_key in common_pairs:
        value_a = parse_float(rows_a[p_key].get(metric))
        value_b = parse_float(rows_b[p_key].get(metric))
        if value_a is None or value_b is None:
            continue
        values_a.append(value_a)
        values_b.append(value_b)
        diffs.append(value_b - value_a)
        paired_ids.append("|".join(p_key))

    if not diffs:
        return None

    diff_mean = mean(diffs)
    diff_sd = stdev(diffs)
    diff_ci95 = ci95_half_width(diffs)
    if len(diffs) >= 2 and diff_sd > 0.0:
        paired_t = diff_mean / (diff_sd / math.sqrt(len(diffs)))
    else:
        paired_t = float("nan")

    return {
        "n": len(diffs),
        "paired_keys": ";".join(paired_ids),
        "mean_a": mean(values_a),
        "mean_b": mean(values_b),
        "delta_mean": diff_mean,
        "delta_ci95_half_width": diff_ci95,
        "delta_ci95_low": diff_mean - diff_ci95 if math.isfinite(diff_ci95) else float("nan"),
        "delta_ci95_high": diff_mean + diff_ci95 if math.isfinite(diff_ci95) else float("nan"),
        "paired_t": paired_t,
        "wins_b": sum(diff > 0.0 for diff in diffs),
        "wins_a": sum(diff < 0.0 for diff in diffs),
        "ties": sum(diff == 0.0 for diff in diffs),
    }


def build_comparisons(
    rows: list[dict[str, str]],
    config_cols: list[str],
    pair_cols: list[str],
    metrics: list[str],
) -> list[dict[str, object]]:
    grouped = collect_by_config_and_pair(rows, config_cols, pair_cols)
    comparisons = []
    for config_a, config_b in itertools.combinations(sorted(grouped), 2):
        for metric in metrics:
            summary = compare_metric(grouped[config_a], grouped[config_b], metric)
            if summary is None:
                continue
            comparisons.append(
                {
                    "config_a": config_label(config_a, config_cols),
                    "config_b": config_label(config_b, config_cols),
                    "metric": metric,
                    **summary,
                }
            )
    return comparisons


def write_summary(path: Path, comparisons: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_a",
        "config_b",
        "metric",
        "n",
        "mean_a",
        "mean_b",
        "delta_mean",
        "delta_ci95_low",
        "delta_ci95_high",
        "delta_ci95_half_width",
        "paired_t",
        "wins_b",
        "wins_a",
        "ties",
        "paired_keys",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparisons:
            writer.writerow(
                {
                    key: format_number(value) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def print_preview(comparisons: list[dict[str, object]], limit: int) -> None:
    if not comparisons:
        print("No paired comparisons were available.")
        return
    for row in comparisons[:limit]:
        print(
            f"{row['metric']}: {row['config_b']} - {row['config_a']} "
            f"n={row['n']} delta={format_number(row['delta_mean'])} "
            f"ci95=[{format_number(row['delta_ci95_low'])}, {format_number(row['delta_ci95_high'])}] "
            f"wins_b={row['wins_b']} wins_a={row['wins_a']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV. Default: <results_csv_dir>/pairwise_comparisons.csv",
    )
    parser.add_argument(
        "--config-cols",
        nargs="+",
        default=None,
        help="Columns defining a configuration. Default: inferred varying sweep columns.",
    )
    parser.add_argument(
        "--pair-cols",
        nargs="+",
        default=["seed"],
        help="Columns used for pairing rows. Default: seed.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metric columns to compare. Default: common zeta/signal/decoder metrics present in the file.",
    )
    parser.add_argument("--preview", type=int, default=20)
    args = parser.parse_args()

    rows = read_rows(args.results_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.results_csv}")

    missing_pair_cols = [column for column in args.pair_cols if column not in rows[0]]
    if missing_pair_cols:
        raise SystemExit(f"Missing pair column(s): {', '.join(missing_pair_cols)}")

    metrics = args.metrics or present_columns(rows, DEFAULT_METRIC_CANDIDATES)
    missing_metrics = [column for column in metrics if column not in rows[0]]
    if missing_metrics:
        raise SystemExit(f"Missing metric column(s): {', '.join(missing_metrics)}")

    config_cols = args.config_cols or infer_config_columns(rows, metrics, args.pair_cols)
    if not config_cols:
        raise SystemExit(
            "Could not infer configuration columns. Pass --config-cols explicitly."
        )
    missing_config_cols = [column for column in config_cols if column not in rows[0]]
    if missing_config_cols:
        raise SystemExit(f"Missing config column(s): {', '.join(missing_config_cols)}")

    out_path = args.out or args.results_csv.with_name("pairwise_comparisons.csv")
    comparisons = build_comparisons(rows, config_cols, args.pair_cols, metrics)
    write_summary(out_path, comparisons)

    configs = {
        config_key(row, config_cols)
        for row in rows
    }
    print(f"Read {len(rows)} rows from {args.results_csv}")
    print(f"Config columns: {', '.join(config_cols)} ({len(configs)} configs)")
    print(f"Pair columns: {', '.join(args.pair_cols)}")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Wrote {len(comparisons)} paired metric comparisons to {out_path}")
    print_preview(comparisons, args.preview)


if __name__ == "__main__":
    main()
