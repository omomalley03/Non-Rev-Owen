"""PhysioNetMI condition-accuracy sweep with frozen embeddings.

Each attempt:
  1. trains a PhysioNetMI embedder for 50 epochs,
  2. freezes the embedder output trajectories,
  3. trains an MLP condition classifier for 50 epochs,
  4. ranks by validation condition-classification accuracy.

Only multiscale-symmetric candidates where TEMPORAL_FILTERS evenly divides the
number of kernel sizes are allowed, so every branch receives the same number of
learned filters.

Usage:
    micromamba run -n nonrev python -u run_physionet_condition_accuracy_sweep.py
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
SWEEP_DIR = ROOT / "physionetmi" / "condition_accuracy_sweep_6subj"
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
    "SYNTH_DATA_PATH": str(ROOT / "cache" / "physionetmi_train_val.npy"),
    "SYNTH_LABELS_PATH": str(ROOT / "cache" / "physionetmi_train_val_labels.npy"),
    "SYNTH_SUBJECTS_PATH": str(ROOT / "cache" / "physionetmi_train_val_subjects.npy"),
    "SYNTH_DATA_LAYOUT": "knt",
    "SYNTH_NORMALIZE": "zscore",
    "SYNTH_PREPROCESS": "none",
    "SYNTH_NOISE_STD": "0",
    "SYNTH_MAX_TRIALS": "10000",
    "SYNTH_SPLIT": "subject_random",
    "SYNTH_SUBJECT_COUNT": "6",
    "SYNTH_SUBJECT_IDS": "",
    "SYNTH_HOLDOUT_SUBJECT_COUNT": "1",
    "SYNTH_HOLDOUT_SUBJECT_IDS": "",
    "VAL_SPLIT": "0.2",
    "TEMPORAL_FRONTEND": "multiscale_symmetric",
    "EPOCHS": "50",
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
    # Strong zeta-frontier settings from the prior temporal-filter sweep, adjusted
    # to equal filters per branch where needed.
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "3,7,15,31", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "7,15,31,61", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 8, "kernels": "7,15,31,61", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 15, "kernels": "3,7,15,31,63", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 10, "kernels": "3,7,15,31,63", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "15,31,63,127", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
]


FOLLOWUP_POOL = [
    {"d": 64, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "3,7,15,31", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 256, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "3,7,15,31", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 1, "embed_dropout": 0.0, "temporal_filters": 16, "kernels": "3,7,15,31", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "3,7,15,31", "conv_layers": 1, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 8, "kernels": "3,7,15,31", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 20, "kernels": "3,7,15,31,63", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "5,11,23,47", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "31,63,95,127", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "3,7,15,31", "conv_layers": 2, "embed_lr": "3e-4", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "3,7,15,31", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 2.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    # Decoder hyperparameter probes around likely good embedders.
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "3,7,15,31", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 512, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "3,7,15,31", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 128, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "3,7,15,31", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 3, "mlp_dropout": 0.2, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "3,7,15,31", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 1, "mlp_dropout": 0.0, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "3,7,15,31", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.0, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "3,7,15,31", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.4, "mlp_lr": "1e-3"},
    {"d": 128, "hidden_dim": 256, "embed_depth": 2, "embed_dropout": 0.2, "temporal_filters": 16, "kernels": "3,7,15,31", "conv_layers": 2, "embed_lr": "1e-3", "lambda_block_cca": 1.0, "mlp_hidden_dim": 256, "mlp_depth": 2, "mlp_dropout": 0.2, "mlp_lr": "3e-4"},
]


def kernel_count(candidate: dict[str, Any]) -> int:
    return len([part for part in str(candidate["kernels"]).split(",") if part.strip()])


def candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(candidate["d"]),
        int(candidate["hidden_dim"]),
        int(candidate["embed_depth"]),
        float(candidate["embed_dropout"]),
        int(candidate["temporal_filters"]),
        str(candidate["kernels"]),
        int(candidate["conv_layers"]),
        str(candidate["embed_lr"]),
        float(candidate["lambda_block_cca"]),
        int(candidate["mlp_hidden_dim"]),
        int(candidate["mlp_depth"]),
        float(candidate["mlp_dropout"]),
        str(candidate["mlp_lr"]),
    )


def candidate_valid(candidate: dict[str, Any]) -> bool:
    kernels = [int(part.strip()) for part in str(candidate["kernels"]).split(",") if part.strip()]
    tf = int(candidate["temporal_filters"])
    return (
        len(kernels) > 0
        and tf >= len(kernels)
        and tf % len(kernels) == 0
        and int(candidate["conv_layers"]) in {1, 2}
        and int(candidate["d"]) % 2 == 0
        and all(k > 0 and k % 2 == 1 for k in kernels)
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_source_indices(ds):
    from torch.utils.data import Subset
    import numpy as np

    if isinstance(ds, Subset):
        parent = _dataset_source_indices(ds.dataset)
        return parent[np.asarray(ds.indices, dtype=int)]
    return np.arange(len(ds), dtype=int)


def run_one(attempt: int, candidate_json: str) -> dict[str, Any]:
    import numpy as np
    import torch

    from config import Config
    from loss import _batch_rms_normalize, non_reversibility_components
    from model import MLP
    from predict_physionet_condition import (
        classification_metrics,
        compute_features,
        predict_mlp_classifier,
        train_mlp_classifier,
        trajectory_features,
        transform_trajectory_features,
    )
    from synth_data import load_synthetic_labels, load_synthetic_subjects, load_synthetic_windows
    from train import train
    from visualize_synth import train_val_split_synth

    candidate = json.loads(candidate_json)
    set_seed(int(os.environ.get("SEED", "1")))
    cfg = Config()

    run_tag = (
        f"{attempt:02d}_acc_d{cfg.d}_tf{cfg.temporal_filters}"
        f"_k{cfg.residual_kernels.replace(',', '-')}_cl{cfg.multiscale_symmetric_conv_layers}"
        f"_ed{cfg.depth}_lr{cfg.lr:g}_mh{candidate['mlp_hidden_dim']}"
        f"_md{candidate['mlp_depth']}_mlr{float(candidate['mlp_lr']):g}"
    )
    run_dir = SWEEP_DIR / "runs" / run_tag
    cfg.ckpt_dir = str(run_dir / "checkpoints")
    cfg.out_dir = str(run_dir / "outputs")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_about(str(run_dir))

    windows = load_synthetic_windows(cfg)
    labels = load_synthetic_labels(cfg)
    subjects = load_synthetic_subjects(cfg)
    if labels is None:
        raise ValueError("PhysioNet condition labels are required for this sweep.")

    train_ds, val_ds, holdout_ds, trainval_subjects, holdout_subjects = train_val_split_synth(
        windows,
        cfg.val_split,
        cfg.seed,
        cfg.synth_split,
        subjects=subjects,
        subject_count=getattr(cfg, "synth_subject_count", 0),
        subject_ids=getattr(cfg, "synth_subject_ids", ""),
        holdout_subject_count=getattr(cfg, "synth_holdout_subject_count", 0),
        holdout_subject_ids=getattr(cfg, "synth_holdout_subject_ids", ""),
        return_holdout=True,
    )
    train_idx = _dataset_source_indices(train_ds)
    val_idx = _dataset_source_indices(val_ds)
    holdout_idx = _dataset_source_indices(holdout_ds) if holdout_ds is not None else None

    y_train_raw = labels[train_idx]
    y_val_raw = labels[val_idx]
    y_test_raw = labels[holdout_idx] if holdout_idx is not None else None
    class_arrays = [y_train_raw, y_val_raw]
    if y_test_raw is not None:
        class_arrays.append(y_test_raw)
    classes = np.array(sorted(np.unique(np.concatenate(class_arrays))), dtype=np.int64)
    class_to_idx = {label: i for i, label in enumerate(classes)}
    y_train = np.array([class_to_idx[int(label)] for label in y_train_raw], dtype=np.int64)
    y_val = np.array([class_to_idx[int(label)] for label in y_val_raw], dtype=np.int64)
    y_test = (
        np.array([class_to_idx[int(label)] for label in y_test_raw], dtype=np.int64)
        if y_test_raw is not None
        else None
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
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "epoch": cfg.epochs,
            "checkpoint_selection": "final_epoch_for_condition_sweep",
        },
        run_dir / "checkpoints" / "final.pt",
    )

    best_epoch = int(history["best_val_epoch"] or 0)
    best_idx = max(best_epoch - 1, 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F_train = compute_features(model, train_ds, cfg.batch_size, device, "output")
    F_val = compute_features(model, val_ds, cfg.batch_size, device, "output")
    F_test = compute_features(model, holdout_ds, cfg.batch_size, device, "output") if holdout_ds is not None else None
    X_train, X_val, emb_mean, feature_mean, feature_std = trajectory_features(
        F_train,
        F_val,
        standardize=True,
    )
    X_test = (
        transform_trajectory_features(F_test, emb_mean, feature_mean, feature_std)
        if F_test is not None
        else None
    )

    pred_val, mlp, mlp_info = train_mlp_classifier(
        X_train,
        y_train,
        X_val,
        y_val,
        n_classes=len(classes),
        hidden_dim=int(candidate["mlp_hidden_dim"]),
        depth=int(candidate["mlp_depth"]),
        dropout=float(candidate["mlp_dropout"]),
        epochs=50,
        batch_size=128,
        lr=float(candidate["mlp_lr"]),
        weight_decay=1e-4,
        seed=int(os.environ.get("SEED", "1")),
        device=device,
    )
    val_metrics = classification_metrics(y_val, pred_val, prefix="val_")
    test_metrics = {}
    if X_test is not None and y_test is not None:
        pred_test = predict_mlp_classifier(mlp, X_test, device)
        test_metrics = classification_metrics(y_test, pred_test, prefix="test_")

    with torch.no_grad():
        Fv = torch.from_numpy(F_val)
        Fv = Fv - Fv.mean(dim=cfg.F_mean_axis, keepdim=True)
        val_c_minus, val_c_plus, val_zeta = non_reversibility_components(_batch_rms_normalize(Fv), "mean")

    decoder_dir = run_dir / "outputs" / "condition_mlp"
    decoder_dir.mkdir(parents=True, exist_ok=True)
    with open(decoder_dir / "metrics.csv", "w", newline="") as f:
        fields = ["model", "val_accuracy", "val_balanced_accuracy", "val_macro_f1", "test_accuracy", "test_balanced_accuracy", "test_macro_f1", "best_val_acc", "best_val_ce"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        row = {
            "model": "mlp",
            **val_metrics,
            **test_metrics,
            "best_val_acc": float(mlp_info["best_val_acc"]),
            "best_val_ce": float(mlp_info["best_val_ce"]),
        }
        writer.writerow(row)

    result = {
        "attempt": attempt,
        "run_dir": str(run_dir),
        "d": cfg.d,
        "hidden_dim": cfg.hidden_dim,
        "embed_depth": cfg.depth,
        "embed_dropout": cfg.dropout,
        "temporal_filters": cfg.temporal_filters,
        "kernels": cfg.residual_kernels,
        "kernel_count": kernel_count(candidate),
        "filters_per_kernel": cfg.temporal_filters // kernel_count(candidate),
        "conv_layers": cfg.multiscale_symmetric_conv_layers,
        "embed_lr": cfg.lr,
        "lambda_block_cca": cfg.lambda_block_cca,
        "mlp_hidden_dim": int(candidate["mlp_hidden_dim"]),
        "mlp_depth": int(candidate["mlp_depth"]),
        "mlp_dropout": float(candidate["mlp_dropout"]),
        "mlp_lr": float(candidate["mlp_lr"]),
        "embed_epochs": cfg.epochs,
        "mlp_epochs": 50,
        "best_embed_val_zeta": float(history["best_val_zeta"]),
        "best_embed_val_epoch": best_epoch,
        "best_embed_val_s": float(history["val_s"][best_idx]) if history["val_s"] else float("nan"),
        "best_embed_val_c_plus": float(history["val_c_plus"][best_idx]) if history["val_c_plus"] else float("nan"),
        "final_embed_val_zeta": float(history["val_zeta"][-1]),
        "frozen_val_zeta": float(val_zeta.item()),
        "frozen_val_s": float(val_c_minus.item()),
        "frozen_val_c_plus": float(val_c_plus.item()),
        "val_accuracy": float(val_metrics["val_accuracy"]),
        "val_balanced_accuracy": float(val_metrics["val_balanced_accuracy"]),
        "val_macro_f1": float(val_metrics["val_macro_f1"]),
        "test_accuracy": float(test_metrics.get("test_accuracy", float("nan"))),
        "test_balanced_accuracy": float(test_metrics.get("test_balanced_accuracy", float("nan"))),
        "test_macro_f1": float(test_metrics.get("test_macro_f1", float("nan"))),
        "mlp_best_val_acc": float(mlp_info["best_val_acc"]),
        "mlp_best_val_ce": float(mlp_info["best_val_ce"]),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(holdout_ds) if holdout_ds is not None else 0,
        "trainval_subjects": np.asarray(trainval_subjects).tolist() if trainval_subjects is not None else [],
        "holdout_subjects": np.asarray(holdout_subjects).tolist() if holdout_subjects is not None else [],
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


def result_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row.get("val_accuracy", 0.0)),
        float(row.get("val_balanced_accuracy", 0.0)),
        float(row.get("test_accuracy", 0.0)) if row.get("test_accuracy") == row.get("test_accuracy") else 0.0,
    )


def save_results(results: list[dict[str, Any]]) -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    fields = [
        "rank",
        "attempt",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "d",
        "temporal_filters",
        "filters_per_kernel",
        "kernels",
        "conv_layers",
        "embed_depth",
        "embed_lr",
        "lambda_block_cca",
        "mlp_hidden_dim",
        "mlp_depth",
        "mlp_dropout",
        "mlp_lr",
        "best_embed_val_zeta",
        "frozen_val_zeta",
        "run_dir",
    ]
    ranked = sorted(results, key=result_sort_key, reverse=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            out = {field: row.get(field, "") for field in fields}
            out["rank"] = rank
            writer.writerow(out)


def propose_next(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    tried = {candidate_key(r["candidate"]) for r in results}
    queue: list[dict[str, Any]] = []
    queue.extend(START_CANDIDATES)

    if results:
        best = max(results, key=result_sort_key)
        b = best["candidate"]
        # Accuracy-targeted local moves around the current best: first decoder,
        # then embedder capacity/context.
        for mlp_hidden in (128, 256, 512):
            queue.append({**b, "mlp_hidden_dim": mlp_hidden})
        for mlp_depth in (1, 2, 3):
            queue.append({**b, "mlp_depth": mlp_depth, "mlp_dropout": 0.0 if mlp_depth == 1 else b["mlp_dropout"]})
        for mlp_dropout in (0.0, 0.2, 0.4):
            queue.append({**b, "mlp_dropout": mlp_dropout})
        for mlp_lr in ("3e-4", "1e-3"):
            queue.append({**b, "mlp_lr": mlp_lr})
        for d in (64, 128, 256):
            queue.append({**b, "d": d})
        for tf in (8, 16, 20):
            queue.append({**b, "temporal_filters": tf})
        for kernels in ("3,7,15,31", "7,15,31,61", "15,31,63,127", "3,7,15,31,63", "5,11,23,47,95"):
            queue.append({**b, "kernels": kernels})
        for conv_layers in (1, 2):
            queue.append({**b, "conv_layers": conv_layers})
        for embed_lr in ("3e-4", "1e-3"):
            queue.append({**b, "embed_lr": embed_lr})
        for lambda_block_cca in (0.0, 1.0, 2.0):
            queue.append({**b, "lambda_block_cca": lambda_block_cca})

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
        candidate = propose_next(results)
        if candidate is None:
            print("No untried valid candidates remain.")
            break
        attempt = len(results) + 1
        env = os.environ.copy()
        env.update(BASE_ENV)
        env.update(
            {
                "D": str(candidate["d"]),
                "HIDDEN_DIM": str(candidate["hidden_dim"]),
                "DEPTH": str(candidate["embed_depth"]),
                "DROPOUT": str(candidate["embed_dropout"]),
                "TEMPORAL_FILTERS": str(candidate["temporal_filters"]),
                "RESIDUAL_KERNELS": str(candidate["kernels"]),
                "MULTISCALE_SYMMETRIC_CONV_LAYERS": str(candidate["conv_layers"]),
                "LR": str(candidate["embed_lr"]),
                "LAMBDA_BLOCK_CCA": str(candidate["lambda_block_cca"]),
                "SEED": "1",
            }
        )
        log_path = SWEEP_DIR / "logs" / f"attempt_{attempt:02d}.log"
        print(
            f"\n=== attempt {attempt}/{max_attempts}: "
            f"d={candidate['d']} tf={candidate['temporal_filters']} "
            f"kernels={candidate['kernels']} layers={candidate['conv_layers']} "
            f"embed_lr={candidate['embed_lr']} mlp=({candidate['mlp_hidden_dim']},"
            f"depth{candidate['mlp_depth']},drop{candidate['mlp_dropout']},lr{candidate['mlp_lr']}) ==="
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

        # The run tag is produced inside run_one from resolved Config values.
        result_files = sorted((SWEEP_DIR / "runs").glob(f"{attempt:02d}_*/sweep_result.json"))
        if not result_files:
            raise FileNotFoundError(f"No sweep_result.json found for attempt {attempt}")
        with open(result_files[-1]) as f:
            result = json.load(f)
        results.append(result)
        save_results(results)
        leader = max(results, key=result_sort_key)
        print(
            f"attempt {attempt} val_acc={result['val_accuracy']:.4f} "
            f"bal={result['val_balanced_accuracy']:.4f} test={result['test_accuracy']:.4f}; "
            f"leader attempt {leader['attempt']} val_acc={leader['val_accuracy']:.4f}"
        )

    print(f"\nWrote {RESULTS_CSV}")
    ranked = sorted(results, key=result_sort_key, reverse=True)
    for rank, row in enumerate(ranked[:10], 1):
        print(
            f"{rank:2d}. attempt {row['attempt']:02d} "
            f"val_acc={row['val_accuracy']:.4f} bal={row['val_balanced_accuracy']:.4f} "
            f"test={row['test_accuracy']:.4f} d={row['d']} tf={row['temporal_filters']} "
            f"kernels={row['kernels']} layers={row['conv_layers']} "
            f"mlp_h={row['mlp_hidden_dim']} mlp_d={row['mlp_depth']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--run-one", type=int, default=None)
    parser.add_argument("--candidate-json", default="")
    args = parser.parse_args()

    if args.run_one is not None:
        result = run_one(args.run_one, args.candidate_json)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    controller(max_attempts=min(max(args.max_attempts, 1), 30))


if __name__ == "__main__":
    main()
