"""All-participant PhysioNetMI frozen temporal-conv decoder sweep.

Each attempt trains or reuses a PhysioNetMI embedder, freezes the best-zeta
checkpoint, extracts hidden embedding trajectories, and trains a temporal
Conv1d condition decoder. The decoder setup matches the optimized reference
path under ``condition_prediction_finetune_temporal_conv_hidden`` while keeping
the embedder frozen.

Usage:
    micromamba run -n nonrev python -u run_physionet_full_frozen_conv_sweep.py --max-attempts 20
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
SWEEP_DIR = ROOT / "physionetmi" / "full_frozen_temporal_conv_sweep_89subj"
RESULTS_CSV = SWEEP_DIR / "results.csv"
RESULTS_JSON = SWEEP_DIR / "results.json"

REFERENCE_RUN = (
    ROOT
    / "physionetmi"
    / "synth_runs"
    / "20260714_112036_d128_h256_dep2_bs128_ep150_lr1e-04_lxp0.0_lbt0.0_lcca2.0_sig10.0_s1"
)

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
    "SYNTH_SUBJECT_COUNT": "89",
    "SYNTH_SUBJECT_IDS": "",
    "SYNTH_HOLDOUT_SUBJECT_COUNT": "5",
    "SYNTH_HOLDOUT_SUBJECT_IDS": "",
    "VAL_SPLIT": "0.2",
    "TEMPORAL_FRONTEND": "multiscale_symmetric",
    "BATCH_SIZE": "128",
    "WEIGHT_DECAY": "1e-4",
    "LAMBDA_XP": "0.0",
    "LAMBDA_BT": "0.0",
    "LAMBDA_PLANE_BT": "0.0",
    "LAMBDA_BLOCK_CCA": "2.0",
    "LAMBDA_START_FRAC": "1.0",
    "VAL_CHECKPOINT_METRIC": "zeta",
    "VAL_ZETA_CHECKPOINTS": "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    "S_OBJECTIVE": "mean",
    "S_SOFTMIN_TAU": "0.05",
    "T_0": "10",
    "T_MULT": "2",
    "QUIET_TRAIN": "1",
    "SEED": "1",
}


BASE_DECODER = {
    "feature_layer": "hidden",
    "decoder_hidden_dim": 128,
    "decoder_depth": 2,
    "decoder_kernel_size": 31,
    "decoder_dropout": 0.4,
    "decoder_lr": "1e-3",
    "decoder_weight_decay": "1e-3",
    "decoder_epochs": 50,
    "decoder_batch_size": 128,
    "decoder_seed": 0,
}

DEFAULT_EMBED_EPOCHS = 70
DECODER_EARLY_STOP_EPOCH = 25
DECODER_LEADER_MARGIN = 0.08
DECODER_MIN_EARLY_STOP_ACC = 0.35


def candidate(
    *,
    d: int = 128,
    hidden_dim: int = 256,
    embed_depth: int = 2,
    embed_dropout: float = 0.2,
    temporal_filters: int = 4,
    kernels: str = "7,15,31,61",
    conv_layers: int = 1,
    embed_lr: str = "1e-4",
    lambda_block_cca: float = 2.0,
    embed_epochs: int = DEFAULT_EMBED_EPOCHS,
    source_run_dir: str = "",
    decoder_hidden_dim: int = 128,
    decoder_depth: int = 2,
    decoder_kernel_size: int = 31,
    decoder_dropout: float = 0.4,
    decoder_lr: str = "1e-3",
    decoder_weight_decay: str = "1e-3",
) -> dict[str, Any]:
    c = {
        "d": d,
        "hidden_dim": hidden_dim,
        "embed_depth": embed_depth,
        "embed_dropout": embed_dropout,
        "temporal_filters": temporal_filters,
        "kernels": kernels,
        "conv_layers": conv_layers,
        "embed_lr": embed_lr,
        "lambda_block_cca": lambda_block_cca,
        "embed_epochs": embed_epochs,
        "source_run_dir": source_run_dir,
    }
    c.update(BASE_DECODER)
    c.update(
        {
            "decoder_hidden_dim": decoder_hidden_dim,
            "decoder_depth": decoder_depth,
            "decoder_kernel_size": decoder_kernel_size,
            "decoder_dropout": decoder_dropout,
            "decoder_lr": decoder_lr,
            "decoder_weight_decay": decoder_weight_decay,
        }
    )
    return c


CANDIDATES = [
    candidate(source_run_dir=str(REFERENCE_RUN)),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=256),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=128, decoder_depth=3),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=128, decoder_depth=2, decoder_kernel_size=15),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=128, decoder_depth=2, decoder_kernel_size=63),
    candidate(temporal_filters=8, kernels="7,15,31,61", conv_layers=1, decoder_hidden_dim=256, decoder_depth=2),
    candidate(temporal_filters=16, kernels="7,15,31,61", conv_layers=2, decoder_hidden_dim=256, decoder_depth=2),
    candidate(temporal_filters=20, kernels="15,31,63,127", conv_layers=2, lambda_block_cca=1.0, embed_lr="1e-3", decoder_hidden_dim=512, decoder_depth=3),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=128, decoder_depth=2, decoder_kernel_size=15, decoder_dropout=0.2),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=128, decoder_depth=2, decoder_kernel_size=15, decoder_dropout=0.3),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=128, decoder_depth=2, decoder_kernel_size=15, decoder_dropout=0.5),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=128, decoder_depth=2, decoder_kernel_size=15, decoder_lr="3e-4"),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=128, decoder_depth=2, decoder_kernel_size=15, decoder_weight_decay="1e-4"),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=256, decoder_depth=2, decoder_kernel_size=15),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=64, decoder_depth=2, decoder_kernel_size=15),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=128, decoder_depth=1, decoder_kernel_size=15),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=128, decoder_depth=2, decoder_kernel_size=7),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=128, decoder_depth=2, decoder_kernel_size=21),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=128, decoder_depth=2, decoder_kernel_size=31, decoder_dropout=0.3),
    candidate(source_run_dir=str(REFERENCE_RUN), decoder_hidden_dim=128, decoder_depth=2, decoder_kernel_size=15, decoder_lr="3e-3"),
]


FIELDNAMES = [
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
    "embed_epochs",
    "embed_lr",
    "lambda_block_cca",
    "source_run_dir",
    "feature_layer",
    "decoder_hidden_dim",
    "decoder_depth",
    "decoder_kernel_size",
    "decoder_dropout",
    "decoder_lr",
    "decoder_weight_decay",
    "decoder_epochs",
    "decoder_epochs_ran",
    "best_embed_val_zeta",
    "best_embed_val_epoch",
    "frozen_val_zeta",
    "run_dir",
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
        int(candidate["embed_epochs"]),
        str(candidate.get("source_run_dir", "")),
        str(candidate["feature_layer"]),
        int(candidate["decoder_hidden_dim"]),
        int(candidate["decoder_depth"]),
        int(candidate["decoder_kernel_size"]),
        float(candidate["decoder_dropout"]),
        str(candidate["decoder_lr"]),
        str(candidate["decoder_weight_decay"]),
        int(candidate["decoder_epochs"]),
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
        and int(candidate["decoder_depth"]) >= 1
        and int(candidate["decoder_hidden_dim"]) >= 1
        and int(candidate["decoder_kernel_size"]) > 0
        and int(candidate["decoder_kernel_size"]) % 2 == 1
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


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _run_tag(attempt: int, cfg, candidate: dict[str, Any]) -> str:
    source = "ref" if candidate.get("source_run_dir") else "train"
    return (
        f"{attempt:02d}_{source}_d{cfg.d}_tf{cfg.temporal_filters}"
        f"_k{str(cfg.residual_kernels).replace(',', '-')}_cl{cfg.multiscale_symmetric_conv_layers}"
        f"_ed{cfg.depth}_ep{cfg.epochs}_elr{float(cfg.lr):g}_lcca{float(cfg.lambda_block_cca):g}"
        f"_ch{candidate['decoder_hidden_dim']}_cd{candidate['decoder_depth']}"
        f"_ck{candidate['decoder_kernel_size']}_cdrop{float(candidate['decoder_dropout']):g}"
    )


def run_one(attempt: int, candidate_json: str) -> dict[str, Any]:
    import numpy as np
    import torch

    from config import Config
    from loss import _batch_rms_normalize, non_reversibility_components
    from model import MLP
    from predict_physionet_condition import (
        build_model_from_checkpoint,
        classification_metrics,
        compute_features,
        predict_temporal_conv_classifier,
        sequence_features,
        train_temporal_conv_classifier,
        transform_sequence_features,
    )
    from synth_data import load_synthetic_labels, load_synthetic_subjects, load_synthetic_windows
    from train import train
    from visualize_synth import train_val_split_synth

    candidate = json.loads(candidate_json)
    set_seed(int(os.environ.get("SEED", "1")))
    source_run_dir = str(candidate.get("source_run_dir") or "").strip()

    if source_run_dir:
        source_dir = Path(source_run_dir)
        ckpt_path = source_dir / "checkpoints" / "best.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"missing source checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        run_tag = _run_tag(attempt, cfg, candidate)
        run_dir = SWEEP_DIR / "runs" / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save_about(str(run_dir))
        model = build_model_from_checkpoint(
            cfg,
            ckpt["model_state_dict"],
            load_synthetic_windows(cfg).shape[1],
            init="pretrained",
        )
        history = None
        best_embed_val_zeta = _safe_float(ckpt.get("val_zeta"))
        best_embed_val_epoch = int(ckpt.get("epoch") or 0)
    else:
        cfg = Config()
        run_tag = _run_tag(attempt, cfg, candidate)
        run_dir = SWEEP_DIR / "runs" / run_tag
        cfg.ckpt_dir = str(run_dir / "checkpoints")
        cfg.out_dir = str(run_dir / "outputs")
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save_about(str(run_dir))

        windows_for_shape = load_synthetic_windows(cfg)
        subjects_for_split = load_synthetic_subjects(cfg)
        train_ds_embed, val_ds_embed, _, _, _ = train_val_split_synth(
            windows_for_shape,
            cfg.val_split,
            cfg.seed,
            cfg.synth_split,
            subjects=subjects_for_split,
            subject_count=getattr(cfg, "synth_subject_count", 0),
            subject_ids=getattr(cfg, "synth_subject_ids", ""),
            holdout_subject_count=getattr(cfg, "synth_holdout_subject_count", 0),
            holdout_subject_ids=getattr(cfg, "synth_holdout_subject_ids", ""),
            return_holdout=True,
        )
        model = MLP(
            in_channels=windows_for_shape.shape[1],
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
        history = train(model, train_ds_embed, val_ds_embed, cfg)
        ckpt_path = Path(cfg.ckpt_dir) / "best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model_from_checkpoint(
            cfg,
            ckpt["model_state_dict"],
            windows_for_shape.shape[1],
            init="pretrained",
        )
        best_embed_val_zeta = _safe_float(history.get("best_val_zeta") if history else ckpt.get("val_zeta"))
        best_embed_val_epoch = int(history.get("best_val_epoch") if history else ckpt.get("epoch") or 0)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_layer = str(candidate["feature_layer"])
    F_train = compute_features(model, train_ds, int(cfg.batch_size), device, feature_layer)
    F_val = compute_features(model, val_ds, int(cfg.batch_size), device, feature_layer)
    F_test = (
        compute_features(model, holdout_ds, int(cfg.batch_size), device, feature_layer)
        if holdout_ds is not None
        else None
    )
    F_train_seq, F_val_seq, emb_mean, seq_feature_mean, seq_feature_std = sequence_features(
        F_train,
        F_val,
        standardize=True,
    )
    F_test_seq = (
        transform_sequence_features(F_test, emb_mean, seq_feature_mean, seq_feature_std)
        if F_test is not None
        else None
    )

    pred_val, decoder, decoder_info = train_temporal_conv_classifier(
        F_train_seq,
        y_train,
        F_val_seq,
        y_val,
        n_classes=len(classes),
        hidden_dim=int(candidate["decoder_hidden_dim"]),
        depth=int(candidate["decoder_depth"]),
        kernel_size=int(candidate["decoder_kernel_size"]),
        dropout=float(candidate["decoder_dropout"]),
        epochs=int(candidate["decoder_epochs"]),
        batch_size=int(candidate["decoder_batch_size"]),
        lr=float(candidate["decoder_lr"]),
        weight_decay=float(candidate["decoder_weight_decay"]),
        seed=int(candidate["decoder_seed"]),
        device=device,
        early_stop_epoch=os.environ.get("DECODER_EARLY_STOP_EPOCH"),
        early_stop_min_acc=os.environ.get("DECODER_EARLY_STOP_MIN_ACC"),
    )
    val_metrics = classification_metrics(y_val, pred_val, prefix="val_")
    test_metrics = {}
    if F_test_seq is not None and y_test is not None:
        pred_test = predict_temporal_conv_classifier(decoder, F_test_seq, device)
        test_metrics = classification_metrics(y_test, pred_test, prefix="test_")

    with torch.no_grad():
        Fv = torch.from_numpy(F_val)
        Fv = Fv - Fv.mean(dim=cfg.F_mean_axis, keepdim=True)
        val_c_minus, val_c_plus, val_zeta = non_reversibility_components(_batch_rms_normalize(Fv), "mean")

    decoder_dir = run_dir / "outputs" / "condition_frozen_temporal_conv_hidden"
    decoder_dir.mkdir(parents=True, exist_ok=True)
    with open(decoder_dir / "metrics.csv", "w", newline="") as f:
        fields = [
            "model",
            "val_accuracy",
            "val_balanced_accuracy",
            "val_macro_f1",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
            "best_val_acc",
            "best_val_ce",
            "epochs_ran",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "model": "frozen_temporal_conv_hidden",
                **val_metrics,
                **test_metrics,
                "best_val_acc": float(decoder_info["best_val_acc"]),
                "best_val_ce": float(decoder_info["best_val_ce"]),
                "epochs_ran": int(decoder_info.get("epochs_ran", candidate["decoder_epochs"])),
            }
        )
    torch.save(
        {
            "temporal_conv_state_dict": decoder.cpu().state_dict(),
            "candidate": candidate,
            "source_run_dir": source_run_dir,
            "checkpoint_path": str(ckpt_path),
            "classes": classes,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "holdout_indices": holdout_idx,
            "trainval_subjects": np.asarray(trainval_subjects),
            "holdout_subjects": np.asarray(holdout_subjects),
            "embedding_mean": emb_mean,
            "feature_mean": seq_feature_mean,
            "feature_std": seq_feature_std,
        },
        decoder_dir / "frozen_temporal_conv_decoder.pt",
    )

    result = {
        "attempt": attempt,
        "run_dir": str(run_dir),
        "source_run_dir": source_run_dir,
        "d": int(cfg.d),
        "hidden_dim": int(cfg.hidden_dim),
        "embed_depth": int(cfg.depth),
        "embed_dropout": float(cfg.dropout),
        "temporal_filters": int(cfg.temporal_filters),
        "kernels": str(cfg.residual_kernels),
        "kernel_count": kernel_count(candidate),
        "filters_per_kernel": int(cfg.temporal_filters) // kernel_count(candidate),
        "conv_layers": int(cfg.multiscale_symmetric_conv_layers),
        "embed_lr": float(cfg.lr),
        "lambda_block_cca": float(cfg.lambda_block_cca),
        "embed_epochs": int(cfg.epochs),
        "feature_layer": feature_layer,
        "decoder_hidden_dim": int(candidate["decoder_hidden_dim"]),
        "decoder_depth": int(candidate["decoder_depth"]),
        "decoder_kernel_size": int(candidate["decoder_kernel_size"]),
        "decoder_dropout": float(candidate["decoder_dropout"]),
        "decoder_lr": float(candidate["decoder_lr"]),
        "decoder_weight_decay": float(candidate["decoder_weight_decay"]),
        "decoder_epochs": int(candidate["decoder_epochs"]),
        "best_embed_val_zeta": best_embed_val_zeta,
        "best_embed_val_epoch": best_embed_val_epoch,
        "frozen_val_zeta": float(val_zeta.item()),
        "frozen_val_s": float(val_c_minus.item()),
        "frozen_val_c_plus": float(val_c_plus.item()),
        "val_accuracy": float(val_metrics["val_accuracy"]),
        "val_balanced_accuracy": float(val_metrics["val_balanced_accuracy"]),
        "val_macro_f1": float(val_metrics["val_macro_f1"]),
        "test_accuracy": float(test_metrics.get("test_accuracy", float("nan"))),
        "test_balanced_accuracy": float(test_metrics.get("test_balanced_accuracy", float("nan"))),
        "test_macro_f1": float(test_metrics.get("test_macro_f1", float("nan"))),
        "decoder_best_val_acc": float(decoder_info["best_val_acc"]),
        "decoder_best_val_ce": float(decoder_info["best_val_ce"]),
        "decoder_epochs_ran": int(decoder_info.get("epochs_ran", candidate["decoder_epochs"])),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(holdout_ds) if holdout_ds is not None else 0,
        "trainval_subjects": np.asarray(trainval_subjects).tolist(),
        "holdout_subjects": np.asarray(holdout_subjects).tolist(),
    }
    with open(run_dir / "sweep_result.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


def result_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _safe_float(row.get("val_accuracy"), -1.0),
        _safe_float(row.get("val_balanced_accuracy"), -1.0),
        _safe_float(row.get("test_accuracy"), -1.0),
    )


def write_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_sorted = sorted(rows, key=result_sort_key, reverse=True)
    ranked = []
    for i, row in enumerate(rows_sorted, start=1):
        ranked.append({"rank": i, **row})
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in ranked:
            writer.writerow(row)
    with open(RESULTS_JSON, "w") as f:
        json.dump(ranked, f, indent=2, sort_keys=True)
    return ranked


def load_existing_results() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((SWEEP_DIR / "runs").glob("*/sweep_result.json")):
        try:
            with open(path) as f:
                rows.append(json.load(f))
        except Exception as exc:
            print(f"warning: could not read {path}: {exc}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--run-one", type=int, default=0)
    parser.add_argument("--candidate-json", default="")
    args = parser.parse_args()

    if args.run_one:
        result = run_one(args.run_one, args.candidate_json)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return

    invalid = [(i, c) for i, c in enumerate(CANDIDATES, start=1) if not candidate_valid(c)]
    if invalid:
        raise SystemExit(f"invalid candidates: {invalid}")

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_existing_results()
    completed = {int(row["attempt"]) for row in existing}
    rows = list(existing)
    write_results(rows)

    for attempt, c in enumerate(CANDIDATES[: args.max_attempts], start=1):
        if attempt in completed:
            print(f"=== attempt {attempt}/{args.max_attempts} already complete ===", flush=True)
            continue

        env = os.environ.copy()
        env.update(BASE_ENV)
        env.update(
            {
                "D": str(c["d"]),
                "HIDDEN_DIM": str(c["hidden_dim"]),
                "DEPTH": str(c["embed_depth"]),
                "DROPOUT": str(c["embed_dropout"]),
                "TEMPORAL_FILTERS": str(c["temporal_filters"]),
                "RESIDUAL_KERNELS": str(c["kernels"]),
                "MULTISCALE_SYMMETRIC_CONV_LAYERS": str(c["conv_layers"]),
                "LR": str(c["embed_lr"]),
                "LAMBDA_BLOCK_CCA": str(c["lambda_block_cca"]),
                "EPOCHS": str(c["embed_epochs"]),
            }
        )
        if rows:
            leader_acc = max(_safe_float(row.get("val_accuracy"), -1.0) for row in rows)
            early_stop_min_acc = max(DECODER_MIN_EARLY_STOP_ACC, leader_acc - DECODER_LEADER_MARGIN)
            env["DECODER_EARLY_STOP_EPOCH"] = str(DECODER_EARLY_STOP_EPOCH)
            env["DECODER_EARLY_STOP_MIN_ACC"] = str(early_stop_min_acc)
        else:
            early_stop_min_acc = None
        candidate_json = json.dumps(c, sort_keys=True)
        log_dir = SWEEP_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"attempt_{attempt:02d}.log"
        print(
            f"\n=== attempt {attempt}/{args.max_attempts}: "
            f"d={c['d']} tf={c['temporal_filters']} kernels={c['kernels']} "
            f"layers={c['conv_layers']} embed_epochs={c['embed_epochs']} embed_lr={c['embed_lr']} "
            f"conv=({c['decoder_hidden_dim']},depth{c['decoder_depth']},"
            f"k{c['decoder_kernel_size']},drop{c['decoder_dropout']},lr{c['decoder_lr']})"
            + (
                f" early_stop@{DECODER_EARLY_STOP_EPOCH}<val_acc {early_stop_min_acc:.4f}"
                if early_stop_min_acc is not None
                else ""
            )
            + " ===",
            flush=True,
        )
        cmd = [
            sys.executable,
            "-u",
            __file__,
            "--run-one",
            str(attempt),
            "--candidate-json",
            candidate_json,
        ]
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log_f.write(line)
                log_f.flush()
            proc.wait()
        if proc.returncode != 0:
            print(f"attempt {attempt} failed; see {log_path}", flush=True)
            raise SystemExit(proc.returncode)

        result_path = next((SWEEP_DIR / "runs").glob(f"{attempt:02d}_*/sweep_result.json"))
        with open(result_path) as f:
            row = json.load(f)
        rows = [r for r in rows if int(r["attempt"]) != attempt] + [row]
        ranked = write_results(rows)
        leader = ranked[0]
        print(
            f"attempt {attempt} val_acc={row['val_accuracy']:.4f} "
            f"bal={row['val_balanced_accuracy']:.4f} test={row['test_accuracy']:.4f}; "
            f"leader attempt {leader['attempt']} val_acc={leader['val_accuracy']:.4f}",
            flush=True,
        )

    ranked = write_results(rows)
    print(f"\nWrote {RESULTS_CSV}", flush=True)
    for row in ranked[:10]:
        print(
            f"{row['rank']:>2}. attempt {row['attempt']} "
            f"val_acc={row['val_accuracy']:.4f} bal={row['val_balanced_accuracy']:.4f} "
            f"test={row['test_accuracy']:.4f} d={row['d']} tf={row['temporal_filters']} "
            f"kernels={row['kernels']} layers={row['conv_layers']} "
            f"conv_h={row['decoder_hidden_dim']} conv_d={row['decoder_depth']} "
            f"conv_k={row['decoder_kernel_size']} drop={row['decoder_dropout']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
