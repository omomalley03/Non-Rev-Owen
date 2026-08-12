"""Run MC Maze embedding models for dimensions 32..512 and seeds 0..4.

The script sources ``mcmaze_config.sh`` once, then overrides only ``D`` and
``SEED`` for each training run. Runs are created by ``main.py`` under
``mcmaze/runs`` using the normal run naming.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model.py").is_file())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sweep_scripts.common.mcmaze_train_finetune_common import compute_whole_validation_mean_plane_zeta

DEFAULT_DIMS = (128,)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
OUT_DIR = ROOT / "mcmaze" / "embedding_dim_seed_sweep_dataset_split"
LOG_DIR = OUT_DIR / "logs"
SUMMARY_CSV = OUT_DIR / "results.csv"


FIELDNAMES = [
    "dimension",
    "seed",
    "run_dir",
    "best_epoch",
    "saved_checkpoint_epoch",
    "checkpoint_selection",
    "best_val_zeta",
    "mean_val_zeta",
    "best_val_s",
    "best_val_c_plus",
    "best_val_loss",
    "validation_batches",
    "regularization_train_log_equiv",
    "regularization_whole_batch_log_equiv",
    "regularization_total_scaled",
    "regularization_whole_batch_total_raw",
    "regularization_whole_batch_total_scaled",
    "train_log",
]


def source_mcmaze_config() -> dict[str, str]:
    cmd = ". ./configs/mcmaze_config.sh >/dev/null 2>&1; env"
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
        "regularization_whole_batch_total_raw",
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


def parse_about_value(run_dir: Path, wanted_key: str) -> Optional[str]:
    about_path = run_dir / "about.txt"
    if not about_path.is_file():
        return None
    for line in about_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == wanted_key:
            return value.strip()
    return None


def checkpoint_metrics(run_dir: Path) -> dict[str, object]:
    import torch

    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing best checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch")
    return {
        "best_epoch": epoch,
        "saved_checkpoint_epoch": epoch,
        "checkpoint_selection": ckpt.get("checkpoint_selection"),
        "best_val_zeta": ckpt.get("val_zeta"),
        "mean_val_zeta": compute_whole_validation_mean_plane_zeta(run_dir),
        "best_val_s": ckpt.get("val_s"),
        "best_val_c_plus": ckpt.get("val_c_plus"),
        "best_val_loss": ckpt.get("val_loss"),
    }


def antisymmetric_planes_from_fraction(dim: int, fraction: float) -> int:
    if dim % 2 != 0:
        raise ValueError(f"mixed_parity requires an even embedding dimension, got D={dim}")
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"antisymmetric plane fraction must be in [0, 1], got {fraction}")
    n_planes = dim // 2
    planes = int(math.floor(n_planes * fraction + 0.5))
    if fraction > 0.0:
        planes = max(1, planes)
    return min(n_planes, planes)


def load_completed(path: Path, antisymmetric_plane_fraction: Optional[float] = None) -> set[tuple[int, int]]:
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
            if not (run_dir.is_dir() and (run_dir / "checkpoints" / "best.pt").is_file()):
                continue
            if antisymmetric_plane_fraction is not None:
                expected = antisymmetric_planes_from_fraction(key[0], antisymmetric_plane_fraction)
                recorded = parse_about_value(run_dir, "antisymmetric_planes")
                try:
                    if recorded is None or int(recorded) != expected:
                        continue
                except ValueError:
                    continue
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


def run_one(
    dim: int,
    seed: int,
    base_env: dict[str, str],
    antisymmetric_plane_fraction: Optional[float] = None,
) -> None:
    env = base_env.copy()
    env["D"] = str(dim)
    env["SEED"] = str(seed)
    if antisymmetric_plane_fraction is not None:
        env["ANTISYMMETRIC_PLANES"] = str(
            antisymmetric_planes_from_fraction(dim, antisymmetric_plane_fraction)
        )
    env["PYTHONUNBUFFERED"] = "1"

    log_path = LOG_DIR / f"d{dim}_seed{seed}.log"
    split_suffix = (
        f", ANTISYMMETRIC_PLANES={env['ANTISYMMETRIC_PLANES']}"
        if antisymmetric_plane_fraction is not None
        else ""
    )
    print(f"\n=== MC Maze embedding: D={dim}, SEED={seed}{split_suffix} ===")
    train_output = run_logged([sys.executable, "-u", "main.py"], env, log_path)
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
        "--antisymmetric-plane-fraction",
        type=float,
        default=None,
        help=(
            "Optional fixed fraction of 2D planes assigned to odd outputs for each D. "
            "For example, 0.25 gives round((D/2)*0.25) odd planes and writes the "
            "resulting integer ANTISYMMETRIC_PLANES into each run environment."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Skip dimension/seed rows already present in results.csv.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without launching training.")
    args = parser.parse_args()

    env_fraction = os.environ.get("ANTISYMMETRIC_PLANE_FRACTION")
    if args.antisymmetric_plane_fraction is None and env_fraction not in (None, ""):
        args.antisymmetric_plane_fraction = float(env_fraction)
    if args.antisymmetric_plane_fraction is not None:
        if args.antisymmetric_plane_fraction < 0.0 or args.antisymmetric_plane_fraction > 1.0:
            parser.error("--antisymmetric-plane-fraction must be in [0, 1]")

    base_env = source_mcmaze_config()
    base_env["DATASET_NAME"] = "mcmaze"
    base_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_nonrev")
    Path(base_env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    completed = (
        load_completed(SUMMARY_CSV, args.antisymmetric_plane_fraction)
        if args.resume
        else set()
    )
    planned = [(dim, seed) for dim in args.dims for seed in args.seeds]
    for dim, seed in planned:
        if (dim, seed) in completed:
            print(f"Skipping completed D={dim}, SEED={seed}")
            continue
        if args.dry_run:
            suffix = ""
            if args.antisymmetric_plane_fraction is not None:
                suffix = (
                    " "
                    f"ANTISYMMETRIC_PLANES={antisymmetric_planes_from_fraction(dim, args.antisymmetric_plane_fraction)}"
                )
            print(f"Would run D={dim}, SEED={seed}{suffix}")
            continue
        run_one(dim, seed, base_env, args.antisymmetric_plane_fraction)


if __name__ == "__main__":
    main()
