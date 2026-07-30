from __future__ import annotations

import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "mcmaze" / "runs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
METRIC_FIELDS = (
    "mse_x",
    "mse_y",
    "rmse_x",
    "rmse_y",
    "rmse_mean",
    "r2_x",
    "r2_y",
    "r2_mean",
    "corr_x",
    "corr_y",
    "best_val_mse_z",
)
TRIAL_SUMMARY_METRIC_FIELDS = (
    "embedding_val_zeta",
    "embedding_whole_val_regularization_raw",
    "checkpoint_regularization_lambda",
    "frozen_decoder_rmse_x",
    "frozen_decoder_rmse_y",
    "frozen_decoder_rmse_mean",
    "frozen_decoder_r2_x",
    "frozen_decoder_r2_y",
    "frozen_decoder_r2_mean",
    "finetuned_decoder_rmse_x",
    "finetuned_decoder_rmse_y",
    "finetuned_decoder_rmse_mean",
    "finetuned_decoder_r2_x",
    "finetuned_decoder_r2_y",
    "finetuned_decoder_r2_mean",
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


def source_mcmaze_config() -> dict[str, str]:
    cmd = ". ./mcmaze_config.sh >/dev/null 2>&1; env"
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
    env["DATASET_NAME"] = "mcmaze"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_nonrev")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def run_logged(cmd: list[str], env: dict[str, str], log_path: Path, dry_run: bool = False) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(cmd)
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
        raise RuntimeError("could not find 'Run directory:' in main.py output")
    return Path(match.group(1).strip())


def default_finetune_metrics_path(run_dir: Path) -> Path:
    return run_dir / "outputs" / "velocity_prediction_finetune_100ms_hidden" / "metrics.csv"


def finetune_complete(run_dir: Path) -> bool:
    metrics_path = default_finetune_metrics_path(run_dir)
    if not metrics_path.is_file():
        return False
    with metrics_path.open(newline="") as f:
        return any(row.get("model") == "finetuned_mlp" for row in csv.DictReader(f))


def load_completed(summary_csv: Path, key_fields: tuple[str, ...]) -> set[tuple[str, ...]]:
    if not summary_csv.is_file():
        return set()
    completed = set()
    with summary_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            run_dir = Path(row.get("run_dir", ""))
            if not (run_dir.is_dir() and finetune_complete(run_dir)):
                continue
            try:
                completed.add(tuple(row[field] for field in key_fields))
            except KeyError:
                continue
    return completed


def append_rows(summary_csv: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.is_file()
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
    """Convert an odd-plane fraction into an integer number of 2D odd planes."""
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


def write_ci95_summary(
    source_csv: Path,
    ci_csv: Path,
    group_fields: tuple[str, ...],
    metric_fields: tuple[str, ...] = METRIC_FIELDS,
    stratify_fields: tuple[str, ...] = ("model", "embedder_init", "feature_layer"),
) -> None:
    """Write per-condition 95% CIs across seed rows from a sweep results CSV."""
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


def read_finetune_metric_rows(run_dir: Path) -> list[dict[str, str]]:
    metrics_path = default_finetune_metrics_path(run_dir)
    if not metrics_path.is_file():
        return []
    with metrics_path.open(newline="") as f:
        return list(csv.DictReader(f))


def embedding_summary_metrics(run_dir: Path) -> dict[str, object]:
    import torch

    about = read_key_value_file(run_dir / "about.txt")
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", about.get("best_epoch", ""))
    val_zeta = ckpt.get("val_zeta", about.get("validation_zeta", ""))
    lambda_scale = parse_float(about.get("lambda_scale"))
    lambda_block_cca = parse_float(about.get("lambda_block_cca"))
    checkpoint_regularization_lambda = (
        lambda_scale * lambda_block_cca
        if lambda_scale is not None and lambda_block_cca is not None
        else None
    )
    raw_whole_val_reg: object = about.get("regularization_whole_batch_total_raw", "")
    if raw_whole_val_reg == "" and checkpoint_regularization_lambda:
        scaled_whole_val_reg = parse_float(
            about.get("regularization_whole_batch_total_scaled")
        )
        if scaled_whole_val_reg is not None:
            raw_whole_val_reg = scaled_whole_val_reg / checkpoint_regularization_lambda
    return {
        "saved_checkpoint_epoch": epoch,
        "checkpoint_selection": ckpt.get("checkpoint_selection", ""),
        "embedding_val_zeta": val_zeta,
        "embedding_whole_val_regularization_raw": raw_whole_val_reg,
        "checkpoint_lambda_scale": about.get("lambda_scale", ""),
        "checkpoint_regularization_lambda": (
            "" if checkpoint_regularization_lambda is None else checkpoint_regularization_lambda
        ),
    }


def decoder_summary_metrics(metric_rows: list[dict[str, str]]) -> dict[str, str]:
    summary: dict[str, str] = {}
    prefixes = {
        "frozen_mlp": "frozen_decoder",
        "finetuned_mlp": "finetuned_decoder",
    }
    for row in metric_rows:
        prefix = prefixes.get(row.get("model", ""))
        if prefix is None:
            continue
        summary[f"{prefix}_feature_layer"] = row.get("feature_layer", "")
        summary[f"{prefix}_feature_dim"] = row.get("feature_dim", "")
        for metric in ("rmse_x", "rmse_y", "rmse_mean", "r2_x", "r2_y", "r2_mean"):
            summary[f"{prefix}_{metric}"] = row.get(metric, "")
    return summary


def run_main_then_finetune(
    env: dict[str, str],
    train_log: Path,
    finetune_log: Path,
    seed: int,
    dry_run: bool = False,
) -> Optional[Path]:
    train_output = run_logged(
        [sys.executable, "-u", "main.py"],
        env,
        train_log,
        dry_run=dry_run,
    )
    if dry_run:
        run_dir = RUNS_DIR / "<new main.py run>"
        run_logged(
            [
                sys.executable,
                "-u",
                "predict_mcmaze_velocity_finetune.py",
                "--run",
                str(run_dir),
                "--seed",
                str(seed),
            ],
            env,
            finetune_log,
            dry_run=True,
        )
        return None

    run_dir = parse_run_dir(train_output)
    run_logged(
        [
            sys.executable,
            "-u",
            "predict_mcmaze_velocity_finetune.py",
            "--run",
            str(run_dir),
            "--seed",
            str(seed),
        ],
        env,
        finetune_log,
    )
    return run_dir
