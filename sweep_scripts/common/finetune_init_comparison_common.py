from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model.py").is_file())

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
    16: 2.1199052992210112,
    17: 2.1098155778331806,
    18: 2.10092204024096,
    19: 2.093024054408263,
    20: 2.0859634472658364,
    21: 2.079613844727662,
    22: 2.0738730679040147,
    23: 2.0686576104190406,
    24: 2.0638985616280205,
    25: 2.059538552753294,
    26: 2.055529438642871,
    27: 2.0518305164802833,
    28: 2.048407141795244,
    29: 2.045229642132703,
    30: 2.042272456301238,
}

RUN_SPEC_FIELDS = (
    "condition",
    "temporal_frontend",
    "odd_plane_fraction",
    "antisymmetric_planes",
    "symmetric_planes",
    "first_symmetric_planes",
    "second_symmetric_planes",
    "n_planes",
    "dimension",
    "hidden_dim",
    "lambda_start_frac",
    "lambda_block_cca",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: object) -> float | None:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt_stat(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return ""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.10g}"


def ci95_half_width(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    standard_error = math.sqrt(variance) / math.sqrt(n)
    return T_CRITICAL_95.get(n - 1, 1.959963984540054) * standard_error


def two_sided_t_p_value(t_stat: float, df: int) -> float:
    if df < 1 or not math.isfinite(t_stat):
        return 0.0 if math.isinf(t_stat) else float("nan")
    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(t_stat), df))
    except Exception:
        return math.erfc(abs(t_stat) / math.sqrt(2.0))


def paired_t_test(deltas: list[float]) -> dict[str, float | int]:
    n = len(deltas)
    if n == 0:
        return {
            "n": 0,
            "mean_delta": float("nan"),
            "delta_ci95_half_width": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
        }
    mean_delta = sum(deltas) / n
    half_width = ci95_half_width(deltas)
    if n < 2:
        t_stat = float("nan")
        p_value = float("nan")
    else:
        variance = sum((value - mean_delta) ** 2 for value in deltas) / (n - 1)
        standard_error = math.sqrt(variance) / math.sqrt(n)
        if standard_error == 0.0:
            t_stat = 0.0 if mean_delta == 0.0 else math.copysign(float("inf"), mean_delta)
        else:
            t_stat = mean_delta / standard_error
        p_value = two_sided_t_p_value(t_stat, n - 1)
    return {
        "n": n,
        "mean_delta": mean_delta,
        "delta_ci95_half_width": half_width,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def row_group_key(row: dict[str, str]) -> tuple[str, ...]:
    if "condition" in row:
        return (row.get("condition", ""),)
    skip = {
        "seed",
        "run_dir",
        "train_log",
        "finetune_log",
        "finetune_metrics",
        "saved_checkpoint_epoch",
        "checkpoint_selection",
        "checkpoint_metric",
        "checkpoint_score",
        "checkpoint_lambda_scale",
        "checkpoint_regularization_lambda",
    }
    metric_prefixes = ("embedding_", "validation_", "regularization_", "frozen_decoder_", "finetuned_decoder_")
    return tuple(
        f"{key}={row.get(key, '')}"
        for key in row
        if key not in skip and not key.startswith(metric_prefixes)
    )


def condition_groups(rows: Iterable[dict[str, str]]) -> list[tuple[tuple[str, ...], list[dict[str, str]]]]:
    groups: list[tuple[tuple[str, ...], list[dict[str, str]]]] = []
    index_by_key: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = row_group_key(row)
        if key not in index_by_key:
            index_by_key[key] = len(groups)
            groups.append((key, []))
        groups[index_by_key[key]][1].append(row)
    return groups


def select_condition_rows(results_csv: Path, condition_index: int) -> list[dict[str, str]]:
    rows = read_rows(results_csv)
    groups = condition_groups(rows)
    if condition_index < 1 or condition_index > len(groups):
        raise ValueError(f"condition index {condition_index} out of range 1-{len(groups)}")
    return groups[condition_index - 1][1]


def print_condition_listing(results_csv: Path) -> None:
    groups = condition_groups(read_rows(results_csv))
    for idx, (_key, rows) in enumerate(groups, 1):
        first = rows[0]
        label = first.get("condition") or ", ".join(_key)
        seeds = ",".join(row.get("seed", "") for row in rows)
        print(f"{idx}: {label}  n={len(rows)}  seeds={seeds}")


def run_logged(cmd: list[str], log_path: Path, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(part) for part in cmd)
    if dry_run:
        print(f"Would run: {printable}")
        print(f"  log: {log_path}")
        return

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_nonrev")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end="")
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"command failed with exit code {rc}: {printable}")


def metrics_complete(metrics_path: Path, expected_model: str) -> bool:
    if not metrics_path.is_file():
        return False
    return any(row.get("model") == expected_model for row in read_rows(metrics_path))


def append_summary_rows(summary_csv: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.is_file()
    with summary_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def existing_completed(summary_csv: Path) -> set[tuple[str, str, str]]:
    if not summary_csv.is_file():
        return set()
    completed = set()
    for row in read_rows(summary_csv):
        metrics_path = Path(row.get("metrics_path", ""))
        expected_model = row.get("model", "")
        if metrics_complete(metrics_path, expected_model):
            completed.add((row.get("source_run_dir", ""), row.get("seed", ""), row.get("embedder_init", "")))
    return completed


def metric_row(metrics_path: Path, model_name: str, embedder_init: str) -> dict[str, str]:
    rows = read_rows(metrics_path)
    for row in rows:
        if row.get("model") == model_name and row.get("embedder_init") == embedder_init:
            return row
    for row in rows:
        if row.get("model") == model_name:
            return row
    raise ValueError(f"missing {model_name!r} row in {metrics_path}")


def write_ci95_summary(
    summary_csv: Path,
    ci_csv: Path,
    metric_fields: tuple[str, ...],
    group_fields: tuple[str, ...] = ("source_condition", "model", "embedder_init", "feature_layer"),
) -> None:
    rows = read_rows(summary_csv) if summary_csv.is_file() else []
    groups: dict[tuple[str, ...], dict[str, list[float]]] = {}
    meta: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        groups.setdefault(key, {metric: [] for metric in metric_fields})
        meta.setdefault(key, {field: row.get(field, "") for field in group_fields})
        for metric in metric_fields:
            value = parse_float(row.get(metric))
            if value is not None:
                groups[key][metric].append(value)

    ci_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*group_fields, "metric", "n", "mean", "ci95_low", "ci95_high", "ci95_half_width"]
    out_rows = []
    for key in sorted(groups):
        for metric in metric_fields:
            values = groups[key][metric]
            if not values:
                continue
            mean = sum(values) / len(values)
            half_width = ci95_half_width(values)
            out_rows.append(
                {
                    **meta[key],
                    "metric": metric,
                    "n": len(values),
                    "mean": fmt_stat(mean),
                    "ci95_low": "" if math.isnan(half_width) else fmt_stat(mean - half_width),
                    "ci95_high": "" if math.isnan(half_width) else fmt_stat(mean + half_width),
                    "ci95_half_width": fmt_stat(half_width),
                }
            )
    with ci_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def write_paired_summary(
    summary_csv: Path,
    paired_csv: Path,
    metric_fields: tuple[str, ...],
    pretrained_init: str = "pretrained",
    random_init: str = "random",
) -> None:
    rows = read_rows(summary_csv) if summary_csv.is_file() else []
    by_seed_init: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("model", "").startswith("finetuned_"):
            by_seed_init[(row.get("seed", ""), row.get("embedder_init", ""))] = row

    seeds = sorted(
        {
            seed
            for seed, init in by_seed_init
            if init == pretrained_init and (seed, random_init) in by_seed_init
        },
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)),
    )
    paired_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_condition",
        "model",
        "metric",
        "n",
        "paired_seeds",
        "pretrained_mean",
        "random_mean",
        "mean_delta_pretrained_minus_random",
        "delta_ci95_low",
        "delta_ci95_high",
        "delta_ci95_half_width",
        "t_stat",
        "p_value",
    ]
    out_rows = []
    source_condition = ""
    model = ""
    for metric in metric_fields:
        pretrained_values = []
        random_values = []
        deltas = []
        used_seeds = []
        for seed in seeds:
            pre = parse_float(by_seed_init[(seed, pretrained_init)].get(metric))
            rnd = parse_float(by_seed_init[(seed, random_init)].get(metric))
            if pre is None or rnd is None:
                continue
            pretrained_values.append(pre)
            random_values.append(rnd)
            deltas.append(pre - rnd)
            used_seeds.append(seed)
            source_condition = by_seed_init[(seed, pretrained_init)].get("source_condition", source_condition)
            model = by_seed_init[(seed, pretrained_init)].get("model", model)
        if not deltas:
            continue
        stats = paired_t_test(deltas)
        mean_delta = float(stats["mean_delta"])
        half_width = float(stats["delta_ci95_half_width"])
        out_rows.append(
            {
                "source_condition": source_condition,
                "model": model,
                "metric": metric,
                "n": stats["n"],
                "paired_seeds": ";".join(used_seeds),
                "pretrained_mean": fmt_stat(sum(pretrained_values) / len(pretrained_values)),
                "random_mean": fmt_stat(sum(random_values) / len(random_values)),
                "mean_delta_pretrained_minus_random": fmt_stat(mean_delta),
                "delta_ci95_low": "" if math.isnan(half_width) else fmt_stat(mean_delta - half_width),
                "delta_ci95_high": "" if math.isnan(half_width) else fmt_stat(mean_delta + half_width),
                "delta_ci95_half_width": fmt_stat(half_width),
                "t_stat": fmt_stat(float(stats["t_stat"])),
                "p_value": fmt_stat(float(stats["p_value"])),
            }
        )
    with paired_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def copied_source_fields(source_row: dict[str, str]) -> dict[str, str]:
    return {f"source_{field}": source_row.get(field, "") for field in RUN_SPEC_FIELDS if field in source_row}
