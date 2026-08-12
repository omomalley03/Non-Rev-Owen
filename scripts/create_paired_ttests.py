#!/usr/bin/env python3
"""Create baseline-vs-config paired t-tests from a sweep results folder.

The input can be either a sweep directory containing ``results.csv`` or an
explicit results CSV. Config indices are 1-based and follow first appearance in
``results.csv``.

Examples:
    python scripts/create_paired_ttests.py sweep_results/hyperparam_sweep_paper_logs/mcmaze/cca_sweep_constant_lambda
    python scripts/create_paired_ttests.py sweep_results/hyperparam_sweep_paper_logs/mcmaze/cca_sweep_constant_lambda --baseline-index 2
    python scripts/create_paired_ttests.py sweep_results/hyperparam_sweep_paper_logs/physionetmi/dim_train_finetune_sweep_v2 --list-configs
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional

from pairwise_compare_results import (
    DEFAULT_METRIC_CANDIDATES,
    ci95_half_width,
    config_key,
    config_label,
    collect_by_config_and_pair,
    infer_config_columns,
    mean,
    parse_float,
    present_columns,
    read_rows,
    stdev,
)


def resolve_results_csv(path: Path) -> Path:
    if path.is_dir():
        return path / "results.csv"
    return path


def format_number(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, float):
        return str(value)
    if math.isnan(value):
        return ""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.10g}"


def seed_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def two_sided_t_p_value(t_stat: float, df: int) -> float:
    if df < 1 or not math.isfinite(t_stat):
        return 0.0 if math.isinf(t_stat) else float("nan")
    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(t_stat), df))
    except Exception:
        return math.erfc(abs(t_stat) / math.sqrt(2.0))


def paired_t_summary(deltas: list[float]) -> dict[str, float | int]:
    n = len(deltas)
    if n == 0:
        return {
            "n": 0,
            "mean_delta": float("nan"),
            "delta_ci95_half_width": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
        }
    mean_delta = mean(deltas)
    half_width = ci95_half_width(deltas)
    if n < 2:
        t_stat = float("nan")
        p_value = float("nan")
    else:
        sd = stdev(deltas)
        if sd == 0.0:
            t_stat = 0.0 if mean_delta == 0.0 else math.copysign(float("inf"), mean_delta)
        else:
            t_stat = mean_delta / (sd / math.sqrt(n))
        p_value = two_sided_t_p_value(t_stat, n - 1)
    return {
        "n": n,
        "mean_delta": mean_delta,
        "delta_ci95_half_width": half_width,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def ordered_config_keys(
    rows: list[dict[str, str]],
    config_cols: list[str],
) -> list[tuple[str, ...]]:
    configs = []
    seen = set()
    for row in rows:
        key = config_key(row, config_cols)
        if key in seen:
            continue
        seen.add(key)
        configs.append(key)
    return configs


def config_meta(key: tuple[str, ...], config_cols: list[str]) -> dict[str, str]:
    return {
        column: value
        for column, value in zip(config_cols, key)
    }


def paired_metric_row(
    baseline_rows: dict[tuple[str, ...], dict[str, str]],
    experimental_rows: dict[tuple[str, ...], dict[str, str]],
    metric: str,
    pair_cols: list[str],
) -> Optional[dict[str, object]]:
    common_pairs = sorted(set(baseline_rows) & set(experimental_rows), key=lambda key: tuple(seed_sort_key(part) for part in key))
    baseline_values = []
    experimental_values = []
    deltas = []
    paired_keys = []
    for pair in common_pairs:
        baseline_value = parse_float(baseline_rows[pair].get(metric))
        experimental_value = parse_float(experimental_rows[pair].get(metric))
        if baseline_value is None or experimental_value is None:
            continue
        baseline_values.append(baseline_value)
        experimental_values.append(experimental_value)
        deltas.append(experimental_value - baseline_value)
        paired_keys.append("|".join(pair))

    if not deltas:
        return None

    stats = paired_t_summary(deltas)
    mean_delta = stats["mean_delta"]
    half_width = stats["delta_ci95_half_width"]
    return {
        "metric": metric,
        "n": stats["n"],
        "paired_keys": ";".join(paired_keys),
        "pair_columns": "|".join(pair_cols),
        "baseline_mean": mean(baseline_values),
        "experimental_mean": mean(experimental_values),
        "mean_delta": mean_delta,
        "delta_ci95_low": mean_delta - half_width if math.isfinite(half_width) else float("nan"),
        "delta_ci95_high": mean_delta + half_width if math.isfinite(half_width) else float("nan"),
        "delta_ci95_half_width": half_width,
        "t_stat": stats["t_stat"],
        "p_value": stats["p_value"],
        "wins_experimental": sum(delta > 0.0 for delta in deltas),
        "wins_baseline": sum(delta < 0.0 for delta in deltas),
        "ties": sum(delta == 0.0 for delta in deltas),
    }


def build_paired_ttests(
    rows: list[dict[str, str]],
    config_cols: list[str],
    pair_cols: list[str],
    metrics: list[str],
    baseline_index: int,
) -> tuple[list[dict[str, object]], list[tuple[str, ...]], tuple[str, ...]]:
    configs = ordered_config_keys(rows, config_cols)
    if not configs:
        raise ValueError("No configurations found.")
    if baseline_index < 1 or baseline_index > len(configs):
        raise ValueError(
            f"--baseline-index must be in 1..{len(configs)}, got {baseline_index}"
        )

    baseline_config = configs[baseline_index - 1]
    grouped = collect_by_config_and_pair(rows, config_cols, pair_cols)
    baseline_rows = grouped[baseline_config]
    comparisons = []
    for config_index, experimental_config in enumerate(configs, 1):
        if experimental_config == baseline_config:
            continue
        experimental_rows = grouped[experimental_config]
        for metric in metrics:
            summary = paired_metric_row(baseline_rows, experimental_rows, metric, pair_cols)
            if summary is None:
                continue
            comparisons.append(
                {
                    "baseline_index": baseline_index,
                    "config_index": config_index,
                    "baseline_config": config_label(baseline_config, config_cols),
                    "experimental_config": config_label(experimental_config, config_cols),
                    **{
                        f"baseline_{column}": value
                        for column, value in config_meta(baseline_config, config_cols).items()
                    },
                    **config_meta(experimental_config, config_cols),
                    **summary,
                }
            )
    return comparisons, configs, baseline_config


def write_summary(
    path: Path,
    rows: list[dict[str, object]],
    config_cols: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "baseline_index",
        "config_index",
        "baseline_config",
        "experimental_config",
        *[f"baseline_{column}" for column in config_cols],
        *config_cols,
        "metric",
        "n",
        "pair_columns",
        "paired_keys",
        "baseline_mean",
        "experimental_mean",
        "mean_delta",
        "delta_ci95_low",
        "delta_ci95_high",
        "delta_ci95_half_width",
        "t_stat",
        "p_value",
        "wins_experimental",
        "wins_baseline",
        "ties",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_number(row.get(key, "")) for key in fieldnames})


def print_configs(
    configs: list[tuple[str, ...]],
    config_cols: list[str],
    baseline_config: tuple[str, ...],
) -> None:
    for index, key in enumerate(configs, 1):
        marker = " baseline" if key == baseline_config else ""
        print(f"{index}: {config_label(key, config_cols)}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder_or_results_csv", type=Path)
    parser.add_argument(
        "--baseline-index",
        "--baseline-run-index",
        type=int,
        default=1,
        help="1-based config index to use as the baseline. Default: 1.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV. Default: <sweep_folder>/paired_ttests.csv",
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
        help="Columns used to pair rows. Default: seed.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metric columns to test. Default: known zeta/signal/decoder metrics present in results.csv.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Print 1-based config indices before writing the paired t-test CSV.",
    )
    args = parser.parse_args()

    results_csv = resolve_results_csv(args.folder_or_results_csv)
    if not results_csv.is_file():
        raise SystemExit(f"No results.csv found at {results_csv}")

    rows = read_rows(results_csv)
    if not rows:
        raise SystemExit(f"No rows found in {results_csv}")

    missing_pair_cols = [column for column in args.pair_cols if column not in rows[0]]
    if missing_pair_cols:
        raise SystemExit(f"Missing pair column(s): {', '.join(missing_pair_cols)}")

    metrics = args.metrics or present_columns(rows, DEFAULT_METRIC_CANDIDATES)
    missing_metrics = [column for column in metrics if column not in rows[0]]
    if missing_metrics:
        raise SystemExit(f"Missing metric column(s): {', '.join(missing_metrics)}")
    if not metrics:
        raise SystemExit("No metric columns found. Pass --metrics explicitly.")

    config_cols = args.config_cols or infer_config_columns(rows, metrics, args.pair_cols)
    if not config_cols:
        raise SystemExit(
            "Could not infer configuration columns. Pass --config-cols explicitly."
        )
    missing_config_cols = [column for column in config_cols if column not in rows[0]]
    if missing_config_cols:
        raise SystemExit(f"Missing config column(s): {', '.join(missing_config_cols)}")

    comparisons, configs, baseline_config = build_paired_ttests(
        rows,
        config_cols,
        args.pair_cols,
        metrics,
        args.baseline_index,
    )
    out_path = args.out or results_csv.with_name("paired_ttests.csv")
    write_summary(out_path, comparisons, config_cols)

    print(f"Read {len(rows)} rows from {results_csv}")
    print(f"Config columns: {', '.join(config_cols)} ({len(configs)} configs)")
    print(f"Pair columns: {', '.join(args.pair_cols)}")
    print(f"Metrics: {', '.join(metrics)}")
    if args.list_configs:
        print_configs(configs, config_cols, baseline_config)
    else:
        print(
            f"Baseline {args.baseline_index}: "
            f"{config_label(baseline_config, config_cols)}"
        )
    print(f"Wrote {len(comparisons)} paired t-test rows to {out_path}")


if __name__ == "__main__":
    main()
