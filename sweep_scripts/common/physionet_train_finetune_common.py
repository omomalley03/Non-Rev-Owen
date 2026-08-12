from __future__ import annotations

import csv
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model.py").is_file())
RUNS_DIR = REPO_ROOT / "physionetmi" / "synth_runs"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "physionetmi_config.sh"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONDITION_METRIC_FIELDS = (
    "embedding_val_zeta",
    "embedding_val_mean_plane_zeta",
    "checkpoint_score",
    "validation_zeta",
    "validation_loss",
    "regularization_whole_batch_total_raw",
    "checkpoint_regularization_lambda",
    "frozen_decoder_accuracy",
    "frozen_decoder_balanced_accuracy",
    "frozen_decoder_macro_f1",
    "frozen_decoder_best_val_acc",
    "frozen_decoder_best_val_ce",
    "frozen_decoder_test_accuracy",
    "frozen_decoder_test_balanced_accuracy",
    "frozen_decoder_test_macro_f1",
    "finetuned_decoder_accuracy",
    "finetuned_decoder_balanced_accuracy",
    "finetuned_decoder_macro_f1",
    "finetuned_decoder_best_val_acc",
    "finetuned_decoder_best_val_ce",
    "finetuned_decoder_test_accuracy",
    "finetuned_decoder_test_balanced_accuracy",
    "finetuned_decoder_test_macro_f1",
)

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


def fmt_value(value: object) -> str:
    text = str(value)
    return text.replace(".", "p").replace("-", "m").replace("+", "")


def source_physionet_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, str]:
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"missing PhysioNet config: {config_path}")
    cmd = f". {shlex.quote(str(config_path))} >/dev/null 2>&1; env"
    proc = subprocess.run(
        ["sh", "-lc", cmd],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    env = os.environ.copy()
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key] = value
    env["DATASET_NAME"] = "physionetmi"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_nonrev")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def run_logged(cmd: list[str], env: dict[str, str], log_path: Path, dry_run: bool = False) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(part) for part in cmd)
    if dry_run:
        print(f"Would run: {printable}")
        print(f"  log: {log_path}")
        return ""

    output = []
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
            output.append(line)
            log.write(line)
            log.flush()
            print(line, end="")
        rc = proc.wait()
    text = "".join(output)
    if rc != 0:
        raise RuntimeError(f"command failed with exit code {rc}: {printable}")
    return text


def parse_run_dir(train_output: str) -> Path:
    match = re.search(r"^Run directory:\s*(.+)$", train_output, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("could not find 'Run directory:' in main_synth.py output")
    return Path(match.group(1).strip())


def finetune_output_dir(run_dir: Path, output_suffix: str, feature_layer: str = "hidden") -> Path:
    out_name = f"condition_prediction_finetune_{output_suffix}"
    if feature_layer == "hidden":
        out_name += "_hidden"
    return run_dir / "outputs" / out_name


def default_finetune_metrics_path(
    run_dir: Path,
    output_suffix: str,
    feature_layer: str = "hidden",
) -> Path:
    return finetune_output_dir(run_dir, output_suffix, feature_layer) / "metrics.csv"


def finetune_complete(metrics_path: Path, require_finetuned: bool = False) -> bool:
    if not metrics_path.is_file():
        return False
    with metrics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    models = {row.get("model") for row in rows}
    required = {"frozen_temporal_conv"}
    if require_finetuned:
        required.add("finetuned_temporal_conv")
    return required.issubset(models)


def load_completed(
    summary_csv: Path,
    key_fields: tuple[str, ...],
    require_finetuned: bool = False,
) -> set[tuple[str, ...]]:
    if not summary_csv.is_file():
        return set()
    completed = set()
    with summary_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            metrics_path = Path(row.get("finetune_metrics", ""))
            if not finetune_complete(metrics_path, require_finetuned=require_finetuned):
                continue
            try:
                completed.add(tuple(row[field] for field in key_fields))
            except KeyError:
                continue
    return completed


def append_rows(summary_csv: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.is_file()
    if exists:
        with summary_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = list(reader.fieldnames or [])
            if existing_fieldnames and existing_fieldnames != fieldnames:
                merged_fieldnames = [
                    *existing_fieldnames,
                    *[field for field in fieldnames if field not in existing_fieldnames],
                ]
                existing_rows = list(reader)
                with summary_csv.open("w", newline="") as out:
                    writer = csv.DictWriter(out, fieldnames=merged_fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(existing_rows)
                fieldnames = merged_fieldnames
    with summary_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_key_value_file(path: Path) -> dict[str, str]:
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_float(value: object) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def antisymmetric_planes_from_fraction(dim: int, fraction: float) -> int:
    if dim % 2 != 0:
        raise ValueError(f"mixed_parity requires an even embedding dimension, got D={dim}")
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"odd-plane fraction must be in [0, 1], got {fraction}")
    n_planes = dim // 2
    planes = int(math.floor(n_planes * fraction + 0.5))
    if fraction > 0.0:
        planes = max(1, planes)
    return min(n_planes, planes)


def ci95_half_width(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    standard_error = math.sqrt(variance) / math.sqrt(n)
    t_critical = T_CRITICAL_95.get(n - 1, 1.959963984540054)
    return t_critical * standard_error


def _two_sided_t_p_value(t_stat: float, df: int) -> float:
    if df < 1 or not math.isfinite(t_stat):
        return 0.0 if math.isinf(t_stat) else float("nan")
    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(t_stat), df))
    except Exception:
        # Fallback for environments without SciPy. Sweeps run in an env with SciPy,
        # but this keeps aggregation usable for quick smoke tests.
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
        p_value = _two_sided_t_p_value(t_stat, n - 1)
    return {
        "n": n,
        "mean_delta": mean_delta,
        "delta_ci95_half_width": half_width,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def _fmt_stat(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return ""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.10g}"


def _seed_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def write_paired_ttest_summary(
    source_csv: Path,
    ttest_csv: Path,
    group_fields: tuple[str, ...],
    baseline_group: dict[str, object],
    metric_fields: tuple[str, ...] = CONDITION_METRIC_FIELDS,
    seed_field: str = "seed",
    stratify_fields: tuple[str, ...] = (),
) -> None:
    """Write paired t-tests of each config's seed deltas against the baseline."""
    ttest_csv.parent.mkdir(parents=True, exist_ok=True)
    baseline_key_by_field = {
        field: str(baseline_group[field]) for field in group_fields
    }
    base_fieldnames = [f"baseline_{field}" for field in group_fields]
    empty_fieldnames = [
        *base_fieldnames,
        *group_fields,
        *stratify_fields,
        "metric",
        "n",
        "paired_seeds",
        "baseline_mean",
        "experimental_mean",
        "mean_delta",
        "delta_ci95_low",
        "delta_ci95_high",
        "delta_ci95_half_width",
        "t_stat",
        "p_value",
    ]
    if not source_csv.is_file():
        with ttest_csv.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=empty_fieldnames).writeheader()
        return

    grouped: dict[tuple[str, ...], dict[str, dict[str, float]]] = {}
    group_meta: dict[tuple[str, ...], dict[str, str]] = {}
    with source_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        present_stratify_fields = tuple(
            field for field in stratify_fields if field in (reader.fieldnames or [])
        )
        key_fields = (*group_fields, *present_stratify_fields)
        fieldnames = [
            *base_fieldnames,
            *group_fields,
            *present_stratify_fields,
            "metric",
            "n",
            "paired_seeds",
            "baseline_mean",
            "experimental_mean",
            "mean_delta",
            "delta_ci95_low",
            "delta_ci95_high",
            "delta_ci95_half_width",
            "t_stat",
            "p_value",
        ]
        for row in reader:
            seed = row.get(seed_field, "")
            if seed == "":
                continue
            key = tuple(row.get(field, "") for field in key_fields)
            grouped.setdefault(key, {metric: {} for metric in metric_fields})
            group_meta.setdefault(key, {field: row.get(field, "") for field in key_fields})
            for metric in metric_fields:
                value = parse_float(row.get(metric))
                if value is not None and math.isfinite(value):
                    grouped[key][metric][seed] = value

    baseline_prefix = tuple(baseline_key_by_field[field] for field in group_fields)
    baseline_meta = {f"baseline_{field}": baseline_key_by_field[field] for field in group_fields}
    rows = []
    for key in sorted(grouped):
        if key[: len(group_fields)] == baseline_prefix:
            continue
        meta = group_meta[key]
        baseline_key = (*baseline_prefix, *key[len(group_fields):])
        baseline_metrics = grouped.get(baseline_key)
        if baseline_metrics is None:
            continue
        for metric in metric_fields:
            baseline_by_seed = baseline_metrics.get(metric, {})
            experimental_by_seed = grouped[key].get(metric, {})
            paired_seeds = sorted(
                set(baseline_by_seed) & set(experimental_by_seed),
                key=_seed_sort_key,
            )
            deltas = [
                experimental_by_seed[seed] - baseline_by_seed[seed]
                for seed in paired_seeds
            ]
            if not deltas:
                continue
            stats = paired_t_test(deltas)
            mean_delta = stats["mean_delta"]
            half_width = stats["delta_ci95_half_width"]
            rows.append(
                {
                    **baseline_meta,
                    **meta,
                    "metric": metric,
                    "n": stats["n"],
                    "paired_seeds": ";".join(paired_seeds),
                    "baseline_mean": _fmt_stat(sum(baseline_by_seed[seed] for seed in paired_seeds) / len(paired_seeds)),
                    "experimental_mean": _fmt_stat(sum(experimental_by_seed[seed] for seed in paired_seeds) / len(paired_seeds)),
                    "mean_delta": _fmt_stat(mean_delta),
                    "delta_ci95_low": "" if math.isnan(half_width) else _fmt_stat(mean_delta - half_width),
                    "delta_ci95_high": "" if math.isnan(half_width) else _fmt_stat(mean_delta + half_width),
                    "delta_ci95_half_width": _fmt_stat(half_width),
                    "t_stat": _fmt_stat(stats["t_stat"]),
                    "p_value": _fmt_stat(stats["p_value"]),
                }
            )

    with ttest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ci95_summary(
    source_csv: Path,
    ci_csv: Path,
    group_fields: tuple[str, ...],
    metric_fields: tuple[str, ...] = CONDITION_METRIC_FIELDS,
    stratify_fields: tuple[str, ...] = (),
) -> None:
    ci_csv.parent.mkdir(parents=True, exist_ok=True)
    if not source_csv.is_file():
        fieldnames = [
            *group_fields,
            *stratify_fields,
            "metric",
            "n",
            "mean",
            "ci95_low",
            "ci95_high",
            "ci95_half_width",
        ]
        with ci_csv.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return

    grouped: dict[tuple[str, ...], dict[str, list[float]]] = {}
    group_meta: dict[tuple[str, ...], dict[str, str]] = {}
    with source_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        present_stratify_fields = tuple(
            field for field in stratify_fields if field in (reader.fieldnames or [])
        )
        fieldnames = [
            *group_fields,
            *present_stratify_fields,
            "metric",
            "n",
            "mean",
            "ci95_low",
            "ci95_high",
            "ci95_half_width",
        ]
        for row in reader:
            key_fields = (*group_fields, *present_stratify_fields)
            key = tuple(row.get(field, "") for field in key_fields)
            grouped.setdefault(key, {metric: [] for metric in metric_fields})
            group_meta.setdefault(key, {field: row.get(field, "") for field in key_fields})
            for metric in metric_fields:
                value = parse_float(row.get(metric))
                if value is not None and math.isfinite(value):
                    grouped[key][metric].append(value)

    rows = []
    for key in sorted(grouped):
        meta = group_meta[key]
        for metric in metric_fields:
            values = grouped[key][metric]
            if not values:
                continue
            n = len(values)
            mean = sum(values) / n
            half_width = ci95_half_width(values)
            rows.append(
                {
                    **meta,
                    "metric": metric,
                    "n": n,
                    "mean": f"{mean:.10g}",
                    "ci95_low": "" if math.isnan(half_width) else f"{mean - half_width:.10g}",
                    "ci95_high": "" if math.isnan(half_width) else f"{mean + half_width:.10g}",
                    "ci95_half_width": "" if math.isnan(half_width) else f"{half_width:.10g}",
                }
            )

    with ci_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_finetune_metric_rows(metrics_path: Path) -> list[dict[str, str]]:
    if not metrics_path.is_file():
        return []
    with metrics_path.open(newline="") as f:
        return list(csv.DictReader(f))


def embedding_summary_metrics(run_dir: Path) -> dict[str, object]:
    import torch

    about = read_key_value_file(run_dir / "about.txt")
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lambda_scale = parse_float(about.get("lambda_scale"))
    lambda_block_cca = parse_float(about.get("lambda_block_cca"))
    checkpoint_regularization_lambda = (
        lambda_scale * lambda_block_cca
        if lambda_scale is not None and lambda_block_cca is not None
        else None
    )
    return {
        "saved_checkpoint_epoch": ckpt.get("epoch", about.get("best_epoch", "")),
        "checkpoint_selection": ckpt.get("checkpoint_selection", ""),
        "checkpoint_metric": ckpt.get("val_checkpoint_metric", ""),
        "checkpoint_score": ckpt.get("val_checkpoint_score", ""),
        "embedding_val_zeta": ckpt.get("val_zeta", ""),
        "embedding_val_mean_plane_zeta": ckpt.get("val_mean_plane_zeta", ""),
        "embedding_val_s": ckpt.get("val_s", ""),
        "embedding_val_c_plus": ckpt.get("val_c_plus", ""),
        "embedding_val_loss": ckpt.get("val_loss", ""),
        "validation_zeta": about.get("validation_zeta", ""),
        "validation_loss": about.get("validation_loss", ""),
        "validation_c_plus": about.get("validation_c_plus", ""),
        "regularization_whole_batch_total_raw": about.get("regularization_whole_batch_total_raw", ""),
        "regularization_whole_batch_total_scaled": about.get("regularization_whole_batch_total_scaled", ""),
        "checkpoint_lambda_scale": about.get("lambda_scale", ""),
        "checkpoint_regularization_lambda": (
            "" if checkpoint_regularization_lambda is None else checkpoint_regularization_lambda
        ),
    }


def decoder_summary_metrics(metric_rows: list[dict[str, str]]) -> dict[str, str]:
    summary: dict[str, str] = {}
    prefixes = {
        "frozen_temporal_conv": "frozen_decoder",
        "finetuned_temporal_conv": "finetuned_decoder",
    }
    copied = (
        "embedder_init",
        "feature_layer",
        "feature_dim",
        "flat_feature_dim",
        "decoder_type",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "best_val_acc",
        "best_val_ce",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
    )
    for row in metric_rows:
        prefix = prefixes.get(row.get("model", ""))
        if prefix is None:
            continue
        for metric in copied:
            summary[f"{prefix}_{metric}"] = row.get(metric, "")
    return summary


def run_main_synth_then_finetune(
    env: dict[str, str],
    train_log: Path,
    finetune_log: Path,
    output_suffix: str,
    seed: int,
    decoder_args: list[str] | None = None,
    dry_run: bool = False,
    frozen_only: bool = True,
) -> Optional[Path]:
    train_output = run_logged(
        [sys.executable, "-u", "main_synth.py"],
        env,
        train_log,
        dry_run=dry_run,
    )
    run_dir = RUNS_DIR / "<new main_synth.py run>" if dry_run else parse_run_dir(train_output)
    cmd = [
        sys.executable,
        "-u",
        "decoders/predict_physionet_condition_finetune.py",
        "--run",
        str(run_dir),
        "--seed",
        str(seed),
        "--feature-layer",
        "hidden",
        "--decoder-type",
        "temporal_conv",
        "--output-suffix",
        output_suffix,
    ]
    if frozen_only:
        cmd.append("--frozen-only")
    if decoder_args:
        cmd.extend(decoder_args)
    run_logged(cmd, env, finetune_log, dry_run=dry_run)
    return None if dry_run else run_dir
