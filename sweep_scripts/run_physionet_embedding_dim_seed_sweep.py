"""Run PhysioNetMI embedding models for dimensions 32..512 and seeds 0..4.

The script sources the PhysioNet config, then overrides only ``D`` and ``SEED``
for each training run. Runs are created by ``main_synth.py`` under the normal
PhysioNet synthetic-run directory configured by ``DATASET_NAME``.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "physionetmi_config.sh"
DEFAULT_DIMS = (16, 32, 64, 128, 256, 512)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
OUT_DIR = ROOT / "physionetmi" / "embedding_dim_seed_sweep"
LOG_DIR = OUT_DIR / "logs"
SUMMARY_CSV = OUT_DIR / "results.csv"


FIELDNAMES = [
    "dimension",
    "seed",
    "run_dir",
    "best_epoch",
    "checkpoint_selection",
    "best_val_zeta",
    "best_val_s",
    "best_val_c_plus",
    "best_val_loss",
    "validation_batches",
    "regularization_train_log_equiv",
    "regularization_whole_batch_log_equiv",
    "regularization_total_scaled",
    "regularization_whole_batch_total_scaled",
    "train_log",
]


def source_physionet_config(config_path: Path) -> dict[str, str]:
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"missing PhysioNet config: {config_path}")
    cmd = f". {shlex.quote(str(config_path))} >/dev/null 2>&1; env"
    proc = subprocess.run(
        ["sh", "-lc", cmd],
        cwd=ROOT,
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
    return env


def run_logged(cmd: list[str], env: dict[str, str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output = []
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
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
        raise RuntimeError(f"command failed with exit code {rc}: {' '.join(cmd)}")
    return text


def parse_run_dir(train_output: str) -> Path:
    match = re.search(r"^Run directory:\s*(.+)$", train_output, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("could not find 'Run directory:' in training output")
    return Path(match.group(1).strip())


def parse_about_metrics(run_dir: Path) -> dict[str, str]:
    about_path = run_dir / "about.txt"
    if not about_path.is_file():
        return {}
    metrics = {}
    wanted = {
        "validation_batches",
        "regularization_train_log_equiv",
        "regularization_whole_batch_log_equiv",
        "regularization_total_scaled",
        "regularization_whole_batch_total_scaled",
    }
    for line in about_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in wanted:
            metrics[key] = value.strip()
    return metrics


def checkpoint_metrics(run_dir: Path) -> dict[str, object]:
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing best checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "best_epoch": ckpt.get("epoch"),
        "checkpoint_selection": ckpt.get("checkpoint_selection"),
        "best_val_zeta": ckpt.get("val_zeta"),
        "best_val_s": ckpt.get("val_s"),
        "best_val_c_plus": ckpt.get("val_c_plus"),
        "best_val_loss": ckpt.get("val_loss"),
    }


def load_completed(path: Path) -> set[tuple[int, int]]:
    if not path.is_file():
        return set()
    completed = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (int(row["dimension"]), int(row["seed"]))
            except (KeyError, ValueError):
                continue
            run_dir = Path(row.get("run_dir", ""))
            if run_dir.is_dir() and (run_dir / "checkpoints" / "best.pt").is_file():
                completed.add(key)
    return completed


def append_summary(row: dict[str, object]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = SUMMARY_CSV.is_file()
    with SUMMARY_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(dim: int, seed: int, base_env: dict[str, str]) -> None:
    env = base_env.copy()
    env["D"] = str(dim)
    env["SEED"] = str(seed)
    env["PYTHONUNBUFFERED"] = "1"

    log_path = LOG_DIR / f"d{dim}_seed{seed}.log"
    print(f"\n=== PhysioNetMI embedding: D={dim}, SEED={seed} ===")
    train_output = run_logged([sys.executable, "-u", "main_synth.py"], env, log_path)
    run_dir = parse_run_dir(train_output)

    row = {
        "dimension": dim,
        "seed": seed,
        "run_dir": str(run_dir),
        "train_log": str(log_path),
    }
    row.update(checkpoint_metrics(run_dir))
    row.update(parse_about_metrics(run_dir))
    append_summary(row)

    print(
        "Recorded "
        f"D={dim} seed={seed}: "
        f"best_val_zeta={row.get('best_val_zeta')} "
        f"whole_batch_reg={row.get('regularization_whole_batch_log_equiv')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=list(DEFAULT_DIMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="PhysioNet config shell file to source before applying D/SEED overrides.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip dimension/seed rows already present in results.csv.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without launching training.")
    args = parser.parse_args()

    base_env = source_physionet_config(args.config)
    base_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_nonrev")
    Path(base_env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    completed = load_completed(SUMMARY_CSV) if args.resume else set()
    planned = [(dim, seed) for dim in args.dims for seed in args.seeds]
    for dim, seed in planned:
        if (dim, seed) in completed:
            print(f"Skipping completed D={dim}, SEED={seed}")
            continue
        if args.dry_run:
            print(f"Would run D={dim}, SEED={seed}")
            continue
        run_one(dim, seed, base_env)


if __name__ == "__main__":
    main()
