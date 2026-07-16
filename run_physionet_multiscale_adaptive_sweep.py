"""Adaptive short sweep for PhysioNetMI multiscale symmetric temporal filters.

This runner screens up to 20 multiscale-symmetric front-end settings for
30 epochs each.  It intentionally avoids the expensive diagnostic plots from
main_synth.py and records a compact CSV/JSON summary instead.

Usage:
    micromamba run -n nonrev python -u run_physionet_multiscale_adaptive_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SWEEP_DIR = ROOT / "physionetmi" / "multiscale_symmetric_sweep_23subj"
RESULTS_CSV = SWEEP_DIR / "results.csv"
RESULTS_JSON = SWEEP_DIR / "results.json"


BASE_ENV = {
    "RUNS_BASE": str(ROOT),
    "DATASET_NAME": "physionetmi",
    "CACHE_DIR": str(ROOT / "cache"),
    "MPLCONFIGDIR": "/tmp/matplotlib_nonrev",
    "PHYSIONETMI_NPY": str(ROOT / "cache" / "physionetmi_train_val.npy"),
    "PHYSIONETMI_LABELS_NPY": str(ROOT / "cache" / "physionetmi_train_val_labels.npy"),
    "PHYSIONETMI_SUBJECTS_NPY": str(ROOT / "cache" / "physionetmi_train_val_subjects.npy"),
    "SYNTH_DATA_LAYOUT": "knt",
    "SYNTH_NORMALIZE": "zscore",
    "SYNTH_PREPROCESS": "none",
    "SYNTH_NOISE_STD": "0",
    "SYNTH_SPLIT": "subject_random",
    "SYNTH_SUBJECT_COUNT": "23",
    "SYNTH_SUBJECT_IDS": "",
    "SYNTH_HOLDOUT_SUBJECT_COUNT": "0",
    "SYNTH_HOLDOUT_SUBJECT_IDS": "",
    "SYNTH_VIZ_MAX_TRIALS": "1",
    "SYNTH_VIZ_MAX_TIMEPOINTS": "1000",
    "SYNTH_VIZ_PARTICIPANT_MODE": "top_zeta",
    "SYNTH_VIZ_PARTICIPANT_COUNT": "8",
    "VAL_SPLIT": "0.2",
    "D": "128",
    "HIDDEN_DIM": "256",
    "DROPOUT": "0.2",
    "TEMPORAL_FRONTEND": "multiscale_symmetric",
    "EPOCHS": "30",
    "BATCH_SIZE": "128",
    "WEIGHT_DECAY": "1e-4",
    "LAMBDA_XP": "0.0",
    "LAMBDA_BT": "0.0",
    "LAMBDA_PLANE_BT": "0.0",
    "LAMBDA_BLOCK_CCA": "1.0",
    "LAMBDA_START_FRAC": "1.0",
    "VAL_CHECKPOINT_METRIC": "zeta",
    "VAL_ZETA_CHECKPOINTS": "0.1,0.2,0.3,0.4,0.5",
    "S_OBJECTIVE": "mean",
    "S_SOFTMIN_TAU": "0.05",
    "T_0": "10",
    "T_MULT": "2",
    "QUIET_TRAIN": "1",
}


START_CANDIDATES = [
    {"temporal_filters": 4, "kernels": "7,15,31,61", "conv_layers": 1, "depth": 2, "lr": "1e-3"},
    {"temporal_filters": 8, "kernels": "7,15,31,61", "conv_layers": 1, "depth": 2, "lr": "1e-3"},
    {"temporal_filters": 4, "kernels": "7,15,31,61", "conv_layers": 2, "depth": 2, "lr": "1e-3"},
    {"temporal_filters": 8, "kernels": "7,15,31,61", "conv_layers": 2, "depth": 2, "lr": "1e-3"},
    {"temporal_filters": 4, "kernels": "3,7,15,31,63", "conv_layers": 1, "depth": 2, "lr": "1e-3"},
    {"temporal_filters": 8, "kernels": "15,31,63,127", "conv_layers": 1, "depth": 2, "lr": "1e-3"},
]


FOLLOWUP_POOL = [
    {"temporal_filters": 2, "kernels": "7,15,31,61", "conv_layers": 1, "depth": 2, "lr": "1e-3"},
    {"temporal_filters": 16, "kernels": "7,15,31,61", "conv_layers": 1, "depth": 2, "lr": "1e-3"},
    {"temporal_filters": 8, "kernels": "3,7,15,31,63", "conv_layers": 1, "depth": 2, "lr": "1e-3"},
    {"temporal_filters": 8, "kernels": "7,15,31", "conv_layers": 1, "depth": 2, "lr": "1e-3"},
    {"temporal_filters": 8, "kernels": "15,31,61", "conv_layers": 1, "depth": 2, "lr": "1e-3"},
    {"temporal_filters": 8, "kernels": "31,63,127", "conv_layers": 1, "depth": 2, "lr": "1e-3"},
    {"temporal_filters": 4, "kernels": "15,31,63,127", "conv_layers": 2, "depth": 2, "lr": "1e-3"},
    {"temporal_filters": 8, "kernels": "15,31,63,127", "conv_layers": 2, "depth": 2, "lr": "1e-3"},
    {"temporal_filters": 4, "kernels": "5,11,23,47,95", "conv_layers": 1, "depth": 2, "lr": "1e-3"},
    {"temporal_filters": 8, "kernels": "5,11,23,47,95", "conv_layers": 1, "depth": 2, "lr": "1e-3"},
    {"temporal_filters": 4, "kernels": "7,15,31,61", "conv_layers": 1, "depth": 1, "lr": "1e-3"},
    {"temporal_filters": 8, "kernels": "7,15,31,61", "conv_layers": 1, "depth": 1, "lr": "1e-3"},
    {"temporal_filters": 4, "kernels": "7,15,31,61", "conv_layers": 1, "depth": 2, "lr": "3e-4"},
    {"temporal_filters": 8, "kernels": "7,15,31,61", "conv_layers": 1, "depth": 2, "lr": "3e-4"},
    {"temporal_filters": 8, "kernels": "3,7,15,31,63", "conv_layers": 2, "depth": 2, "lr": "3e-4"},
    {"temporal_filters": 16, "kernels": "15,31,63,127", "conv_layers": 1, "depth": 2, "lr": "3e-4"},
]


def candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(candidate["temporal_filters"]),
        str(candidate["kernels"]),
        int(candidate["conv_layers"]),
        int(candidate["depth"]),
        str(candidate["lr"]),
    )


def candidate_valid(candidate: dict[str, Any]) -> bool:
    kernels = [part.strip() for part in str(candidate["kernels"]).split(",") if part.strip()]
    if int(candidate["temporal_filters"]) < len(kernels):
        return False
    if int(candidate["conv_layers"]) not in {1, 2}:
        return False
    return all(int(k) > 0 and int(k) % 2 == 1 for k in kernels)


def set_seed(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one(attempt: int, candidate_json: str) -> dict[str, Any]:
    # Import after the subprocess environment is set; Config reads env defaults at import time.
    import numpy as np
    import torch

    from config import Config
    from main_synth import set_seed as set_train_seed
    from main_synth import train_val_split_synth
    from model import MLP
    from synth_data import load_synthetic_subjects, load_synthetic_windows
    from train import train

    candidate = json.loads(candidate_json)
    set_train_seed(int(os.environ.get("SEED", "1")))

    cfg = Config()
    run_tag = (
        f"{attempt:02d}_tf{cfg.temporal_filters}_k{cfg.residual_kernels.replace(',', '-')}"
        f"_cl{cfg.multiscale_symmetric_conv_layers}_dep{cfg.depth}_lr{cfg.lr:g}"
    )
    run_dir = SWEEP_DIR / "runs" / run_tag
    cfg.ckpt_dir = str(run_dir / "checkpoints")
    cfg.out_dir = str(run_dir / "outputs")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_about(str(run_dir))

    windows = load_synthetic_windows(cfg)
    subjects = load_synthetic_subjects(cfg)
    train_ds, val_ds, trainval_subjects, eligible_trials, holdout_subjects, holdout_trials = train_val_split_synth(
        windows,
        cfg.val_split,
        cfg.seed,
        cfg.synth_split,
        subjects=subjects,
        subject_count=getattr(cfg, "synth_subject_count", 0),
        subject_ids=getattr(cfg, "synth_subject_ids", ""),
        holdout_subject_count=getattr(cfg, "synth_holdout_subject_count", 0),
        holdout_subject_ids=getattr(cfg, "synth_holdout_subject_ids", ""),
    )

    model = MLP(
        in_channels=windows.shape[1],
        d=cfg.d,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        dropout=cfg.dropout,
        temporal_filters=cfg.temporal_filters,
        temporal_kernel_size=cfg.temporal_kernel_size,
        temporal_frontend=cfg.temporal_frontend,
        residual_kernels=cfg.residual_kernels,
        multiscale_symmetric_conv_layers=cfg.multiscale_symmetric_conv_layers,
    )
    history = train(model, train_ds, val_ds, cfg)

    best_epoch = int(history["best_val_epoch"] or 0)
    best_idx = max(best_epoch - 1, 0)
    result = {
        "attempt": attempt,
        "run_dir": str(run_dir),
        "temporal_filters": cfg.temporal_filters,
        "kernels": cfg.residual_kernels,
        "conv_layers": cfg.multiscale_symmetric_conv_layers,
        "depth": cfg.depth,
        "hidden_dim": cfg.hidden_dim,
        "d": cfg.d,
        "lr": cfg.lr,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lambda_block_cca": cfg.lambda_block_cca,
        "best_val_zeta": float(history["best_val_zeta"]),
        "best_val_epoch": best_epoch,
        "best_val_s": float(history["val_s"][best_idx]) if history["val_s"] else float("nan"),
        "best_val_c_plus": float(history["val_c_plus"][best_idx]) if history["val_c_plus"] else float("nan"),
        "final_val_zeta": float(history["val_zeta"][-1]),
        "final_val_s": float(history["val_s"][-1]),
        "final_val_c_plus": float(history["val_c_plus"][-1]),
        "elapsed_s": float(history["elapsed_s"]),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "eligible_trials": int(eligible_trials),
        "holdout_trials": int(holdout_trials),
        "candidate": candidate,
    }
    with open(run_dir / "sweep_result.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    with open(run_dir / "candidate.json", "w") as f:
        json.dump(candidate, f, indent=2, sort_keys=True)
    return result


def load_results() -> list[dict[str, Any]]:
    if not RESULTS_JSON.is_file():
        return []
    with open(RESULTS_JSON) as f:
        return json.load(f)


def save_results(results: list[dict[str, Any]]) -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    fields = [
        "rank",
        "attempt",
        "best_val_zeta",
        "best_val_s",
        "best_val_c_plus",
        "final_val_zeta",
        "temporal_filters",
        "kernels",
        "conv_layers",
        "depth",
        "lr",
        "best_val_epoch",
        "elapsed_s",
        "run_dir",
    ]
    ranked = sorted(results, key=lambda r: (r["best_val_zeta"], r["best_val_s"]), reverse=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            out = {field: row.get(field, "") for field in fields}
            out["rank"] = rank
            writer.writerow(out)


def propose_next(results: list[dict[str, Any]], max_attempts: int) -> dict[str, Any] | None:
    tried = {candidate_key(r["candidate"]) for r in results}
    queue: list[dict[str, Any]] = []
    queue.extend(START_CANDIDATES)

    if results:
        best = max(results, key=lambda r: (r["best_val_zeta"], r["best_val_s"]))
        b = best["candidate"]
        kernels = b["kernels"]
        layers = int(b["conv_layers"])
        depth = int(b["depth"])
        # Local refinements around the current best before falling back to the wider pool.
        for tf in sorted({max(1, int(b["temporal_filters"]) // 2), int(b["temporal_filters"]), int(b["temporal_filters"]) * 2}):
            if tf <= 16:
                queue.append({**b, "temporal_filters": tf})
        for lr in ("1e-4", "3e-4", "1e-3"):
            queue.append({**b, "lr": lr})
        for alt_kernels in (kernels, "7,15,31,61", "15,31,63,127", "3,7,15,31,63", "5,11,23,47,95"):
            queue.append({**b, "kernels": alt_kernels})
        queue.append({**b, "conv_layers": 1 if layers == 2 else 2})
        queue.append({**b, "depth": 1 if depth == 2 else 2})

    queue.extend(FOLLOWUP_POOL)
    for candidate in queue:
        if candidate_valid(candidate) and candidate_key(candidate) not in tried:
            return candidate
    return None


def controller(max_attempts: int) -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEP_DIR / "logs").mkdir(exist_ok=True)

    results = load_results()
    save_results(results)

    while len(results) < max_attempts:
        candidate = propose_next(results, max_attempts)
        if candidate is None:
            print("No untried candidates remain.")
            break
        attempt = len(results) + 1
        env = os.environ.copy()
        env.update(BASE_ENV)
        env.update(
            {
                "SYNTH_DATA_PATH": env.get("PHYSIONETMI_NPY", BASE_ENV["PHYSIONETMI_NPY"]),
                "SYNTH_LABELS_PATH": env.get("PHYSIONETMI_LABELS_NPY", BASE_ENV["PHYSIONETMI_LABELS_NPY"]),
                "SYNTH_SUBJECTS_PATH": env.get("PHYSIONETMI_SUBJECTS_NPY", BASE_ENV["PHYSIONETMI_SUBJECTS_NPY"]),
                "TEMPORAL_FILTERS": str(candidate["temporal_filters"]),
                "RESIDUAL_KERNELS": str(candidate["kernels"]),
                "MULTISCALE_SYMMETRIC_CONV_LAYERS": str(candidate["conv_layers"]),
                "DEPTH": str(candidate["depth"]),
                "LR": str(candidate["lr"]),
                "SEED": "1",
            }
        )
        log_path = SWEEP_DIR / "logs" / f"attempt_{attempt:02d}.log"
        print(
            f"\n=== attempt {attempt}/{max_attempts}: "
            f"tf={candidate['temporal_filters']} kernels={candidate['kernels']} "
            f"layers={candidate['conv_layers']} depth={candidate['depth']} lr={candidate['lr']} ==="
        )
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--run-one",
            str(attempt),
            "--candidate-json",
            json.dumps(candidate, sort_keys=True),
        ]
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            print(f"Attempt {attempt} failed; see {log_path}")
            raise subprocess.CalledProcessError(proc.returncode, cmd)

        run_tag = (
            f"{attempt:02d}_tf{candidate['temporal_filters']}_k{str(candidate['kernels']).replace(',', '-')}"
            f"_cl{candidate['conv_layers']}_dep{candidate['depth']}_lr{float(candidate['lr']):g}"
        )
        result_path = SWEEP_DIR / "runs" / run_tag / "sweep_result.json"
        with open(result_path) as f:
            result = json.load(f)
        results.append(result)
        save_results(results)
        ranked = sorted(results, key=lambda r: (r["best_val_zeta"], r["best_val_s"]), reverse=True)
        best = ranked[0]
        print(
            f"attempt {attempt} best_zeta={result['best_val_zeta']:.4f} "
            f"S={result['best_val_s']:.4f} C+={result['best_val_c_plus']:.4f}; "
            f"leader attempt {best['attempt']} zeta={best['best_val_zeta']:.4f}"
        )

    print(f"\nWrote {RESULTS_CSV}")
    ranked = sorted(results, key=lambda r: (r["best_val_zeta"], r["best_val_s"]), reverse=True)
    for rank, row in enumerate(ranked[:10], 1):
        print(
            f"{rank:2d}. attempt {row['attempt']:02d} "
            f"zeta={row['best_val_zeta']:.4f} S={row['best_val_s']:.4f} C+={row['best_val_c_plus']:.4f} "
            f"tf={row['temporal_filters']} kernels={row['kernels']} "
            f"layers={row['conv_layers']} depth={row['depth']} lr={row['lr']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--run-one", type=int, default=None)
    parser.add_argument("--candidate-json", default="")
    args = parser.parse_args()

    if args.run_one is not None:
        result = run_one(args.run_one, args.candidate_json)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    controller(max_attempts=min(max(args.max_attempts, 1), 20))


if __name__ == "__main__":
    main()
