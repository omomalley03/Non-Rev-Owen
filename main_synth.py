import json
import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset, random_split

from config import Config
from paths import SYNTH_RUNS_DIR
from model import MLP
from train import train
from visualize_synth import make_diagnostic_plots_synth
from best_metrics import append_best_model_metrics
from synth_data import apply_train_zscore, load_synthetic_labels, load_synthetic_subjects, load_synthetic_windows


_SUBJECT_RANDOM_SPLITS = {"subject_random", "participant_random"}
_SUBJECT_HOLDOUT_SPLITS = {
    "subject_holdout",
    "participant_holdout",
    "subject_disjoint",
    "participant_disjoint",
    "subject_disjoint_holdout",
    "participant_disjoint_holdout",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_summary(history: dict, cfg: Config):
    print()
    print("=" * 50)
    print("Training complete")
    print(f"  Final train loss    : {history['train_loss'][-1]:.4f}")
    checkpoint_metric = history.get("best_checkpoint_metric", "zeta")
    checkpoint_score = history.get("best_checkpoint_score", history.get("best_val_zeta", float("nan")))
    print(f"  Best checkpoint metric ({checkpoint_metric}) : {checkpoint_score:.4f}")
    print(f"  Val ζ at checkpoint : {history['best_val_zeta']:.4f}")
    print(f"  Val loss at checkpoint : {history['best_checkpoint_val_loss']:.4f}")
    print(f"  Wall-clock time     : {history['elapsed_s']:.1f} s  ({history['elapsed_s']/60:.1f} min)")
    print(f"  Checkpoint          : {cfg.ckpt_dir}/best.pt")
    print(f"  Loss curve          : {cfg.out_dir}/loss_curve.png")
    print("=" * 50)


def _parse_subject_ids(spec: str) -> np.ndarray:
    spec = str(spec or "").strip()
    if not spec or spec.lower() in {"all", "none"}:
        return np.array([], dtype=np.int64)
    return np.array([int(item.strip()) for item in spec.split(",") if item.strip()], dtype=np.int64)


def _dataset_source_indices(ds) -> np.ndarray:
    """Return source-window indices for TensorDataset/Subset split objects."""
    if isinstance(ds, Subset):
        parent = _dataset_source_indices(ds.dataset)
        return parent[np.asarray(ds.indices, dtype=int)]
    return np.arange(len(ds), dtype=int)


def _load_dataset_split_counts(data_path: str, n_windows: int) -> dict[str, int]:
    sidecar_path = data_path + ".json"
    if not os.path.exists(sidecar_path):
        raise ValueError(
            "SYNTH_SPLIT=dataset requires a sidecar JSON next to SYNTH_DATA_PATH "
            f"with split_counts; missing {sidecar_path}"
        )
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    split_counts = sidecar.get("split_counts")
    if not isinstance(split_counts, dict) or "train" not in split_counts or "val" not in split_counts:
        raise ValueError(f"{sidecar_path} must contain split_counts with train and val entries")

    n_train = int(split_counts["train"])
    n_val = int(split_counts["val"])
    if n_train < 1 or n_val < 1:
        raise ValueError(f"dataset split requires positive train/val counts, got {split_counts}")
    if n_train + n_val != n_windows:
        if n_windows < n_train + n_val:
            n_train = min(n_train, n_windows)
            n_val = max(0, n_windows - n_train)
        if n_train + n_val != n_windows or n_val < 1:
            raise ValueError(
                f"dataset split counts train={split_counts['train']} val={split_counts['val']} "
                f"do not match loaded windows={n_windows}; remove SYNTH_MAX_TRIALS or regenerate sidecar"
            )
    return {"train": n_train, "val": n_val}


def _select_subject_sets(
    subjects: np.ndarray,
    rng: np.random.Generator,
    subject_count: int = 0,
    subject_ids: str = "",
    holdout_subject_count: int = 0,
    holdout_subject_ids: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select train/val and held-out subject sets from available subject IDs."""
    unique_subjects = np.unique(subjects)
    explicit_subjects = _parse_subject_ids(subject_ids)

    if explicit_subjects.size:
        missing = np.setdiff1d(explicit_subjects, unique_subjects)
        if missing.size:
            raise ValueError(f"SYNTH_SUBJECT_IDS contains unknown subjects: {missing.tolist()}")
        selected_subjects = np.sort(explicit_subjects)
    elif subject_count and subject_count > 0:
        if subject_count > len(unique_subjects):
            raise ValueError(
                f"SYNTH_SUBJECT_COUNT={subject_count} exceeds available subjects={len(unique_subjects)}"
            )
        selected_subjects = np.sort(rng.choice(unique_subjects, size=subject_count, replace=False))
    else:
        selected_subjects = unique_subjects

    explicit_holdout = _parse_subject_ids(holdout_subject_ids)
    if explicit_holdout.size:
        missing = np.setdiff1d(explicit_holdout, selected_subjects)
        if missing.size:
            raise ValueError(
                "SYNTH_HOLDOUT_SUBJECT_IDS must be within the selected subject pool; "
                f"unknown or unselected subjects: {missing.tolist()}"
            )
        holdout_subjects = np.sort(explicit_holdout)
    elif holdout_subject_count and holdout_subject_count > 0:
        if holdout_subject_count >= len(selected_subjects):
            raise ValueError(
                "SYNTH_HOLDOUT_SUBJECT_COUNT must leave at least one subject for train/val; "
                f"got holdout={holdout_subject_count}, selected={len(selected_subjects)}"
            )
        holdout_subjects = np.sort(
            rng.choice(selected_subjects, size=holdout_subject_count, replace=False)
        )
    else:
        holdout_subjects = np.array([], dtype=selected_subjects.dtype)

    trainval_subjects = np.setdiff1d(selected_subjects, holdout_subjects)
    return selected_subjects, trainval_subjects, holdout_subjects


def _select_validation_subjects(
    trainval_subjects: np.ndarray,
    rng: np.random.Generator,
    val_frac: float,
    val_subject_count: int = 0,
    val_subject_ids: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    explicit_val = _parse_subject_ids(val_subject_ids)
    if explicit_val.size:
        if len(np.unique(explicit_val)) != len(explicit_val):
            raise ValueError("SYNTH_VAL_SUBJECT_IDS must not contain duplicate subjects")
        missing = np.setdiff1d(explicit_val, trainval_subjects)
        if missing.size:
            raise ValueError(
                "SYNTH_VAL_SUBJECT_IDS must be within the selected non-test subject pool; "
                f"unknown, unselected, or test-overlapping subjects: {missing.tolist()}"
            )
        val_subjects = np.sort(explicit_val)
    else:
        if val_subject_count and val_subject_count > 0:
            n_val_subjects = int(val_subject_count)
        else:
            n_val_subjects = max(1, int(len(trainval_subjects) * val_frac))
        if n_val_subjects >= len(trainval_subjects):
            raise ValueError(
                "subject_holdout validation subjects must leave at least one training subject; "
                f"got val={n_val_subjects}, trainval_pool={len(trainval_subjects)}"
            )
        val_subjects = np.sort(rng.choice(trainval_subjects, size=n_val_subjects, replace=False))

    train_subjects = np.setdiff1d(trainval_subjects, val_subjects)
    if len(train_subjects) < 1:
        raise ValueError("subject_holdout split leaves no training subjects")
    return train_subjects, val_subjects


def train_val_split_synth(
    windows: np.ndarray,
    val_frac: float,
    seed: int,
    split: str = "random",
    subjects: Optional[np.ndarray] = None,
    subject_count: int = 0,
    subject_ids: str = "",
    val_subject_count: int = 0,
    val_subject_ids: str = "",
    holdout_subject_count: int = 0,
    holdout_subject_ids: str = "",
    dataset_split_counts: Optional[dict[str, int]] = None,
):
    """Split synthetic windows for training/validation."""
    tensor = torch.from_numpy(windows)
    full_ds = TensorDataset(tensor)

    split = split.lower()
    subject_split_modes = _SUBJECT_RANDOM_SPLITS | _SUBJECT_HOLDOUT_SPLITS
    if split not in subject_split_modes and (
        holdout_subject_count
        or _parse_subject_ids(holdout_subject_ids).size
        or val_subject_count
        or _parse_subject_ids(val_subject_ids).size
    ):
        raise ValueError("Subject holdout settings require SYNTH_SPLIT=subject_random or subject_holdout")
    if split in _SUBJECT_RANDOM_SPLITS and (
        val_subject_count or _parse_subject_ids(val_subject_ids).size
    ):
        raise ValueError("SYNTH_VAL_SUBJECT_COUNT/IDS require SYNTH_SPLIT=subject_holdout")

    if split in {"train_eq_val", "train_equals_val", "all", "none"}:
        return full_ds, full_ds, None, len(tensor), np.array([], dtype=np.int64), 0
    if split == "dataset":
        if dataset_split_counts is None:
            raise ValueError("SYNTH_SPLIT=dataset requires dataset_split_counts")
        n_train = int(dataset_split_counts["train"])
        n_val = int(dataset_split_counts["val"])
        if n_train + n_val != len(tensor):
            raise ValueError(
                f"dataset split counts train={n_train} val={n_val} "
                f"do not match windows length {len(tensor)}"
            )
        return (
            Subset(full_ds, list(range(n_train))),
            Subset(full_ds, list(range(n_train, n_train + n_val))),
            None,
            len(tensor),
            np.array([], dtype=np.int64),
            0,
        )
    if split in subject_split_modes:
        if subjects is None:
            raise ValueError(f"SYNTH_SPLIT={split} requires SYNTH_SUBJECTS_PATH")
        if len(subjects) != len(tensor):
            raise ValueError(
                f"subject IDs length ({len(subjects)}) must match windows length ({len(tensor)})"
            )

        rng = np.random.default_rng(seed)
        _, trainval_subjects, holdout_subjects = _select_subject_sets(
            subjects,
            rng,
            subject_count=subject_count,
            subject_ids=subject_ids,
            holdout_subject_count=holdout_subject_count,
            holdout_subject_ids=holdout_subject_ids,
        )

        if split in _SUBJECT_HOLDOUT_SPLITS:
            train_subjects, val_subjects = _select_validation_subjects(
                trainval_subjects,
                rng,
                val_frac,
                val_subject_count=val_subject_count,
                val_subject_ids=val_subject_ids,
            )
            train_idx = np.flatnonzero(np.isin(subjects, train_subjects))
            val_idx = np.flatnonzero(np.isin(subjects, val_subjects))
            holdout_idx = np.flatnonzero(np.isin(subjects, holdout_subjects))
            if len(train_idx) < 1 or len(val_idx) < 1:
                raise ValueError(
                    f"subject_holdout split produced train={len(train_idx)} val={len(val_idx)} trials"
                )
            return (
                Subset(full_ds, train_idx.tolist()),
                Subset(full_ds, val_idx.tolist()),
                trainval_subjects,
                len(train_idx) + len(val_idx),
                holdout_subjects,
                len(holdout_idx),
            )

        eligible = np.flatnonzero(np.isin(subjects, trainval_subjects))
        if len(eligible) < 2:
            raise ValueError("subject_random split requires at least two eligible trials")
        shuffled = rng.permutation(eligible)
        n_val = max(1, int(len(shuffled) * val_frac))
        n_train = len(shuffled) - n_val
        if n_train < 1:
            raise ValueError("subject_random split leaves no training trials")
        holdout_idx = np.flatnonzero(np.isin(subjects, holdout_subjects))
        return (
            Subset(full_ds, shuffled[:n_train].tolist()),
            Subset(full_ds, shuffled[n_train:].tolist()),
            trainval_subjects,
            len(eligible),
            holdout_subjects,
            len(holdout_idx),
        )
    if split != "random":
        raise ValueError("SYNTH_SPLIT must be one of: random, train_eq_val, subject_random, subject_holdout, dataset")

    n_val = max(1, int(len(tensor) * val_frac))
    n_train = len(tensor) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)
    return train_ds, val_ds, None, len(tensor), np.array([], dtype=np.int64), 0


def main():
    cfg = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(SYNTH_RUNS_DIR, f"{timestamp}_{cfg.run_name()}")
    cfg.ckpt_dir = os.path.join(run_dir, "checkpoints")
    cfg.out_dir  = os.path.join(run_dir, "outputs")
    cfg.save_about(run_dir)
    print(f"Run directory: {run_dir}")
    set_seed(cfg.seed)
    print("Random seed set to:", cfg.seed)

    temporal_context_bins = (
        int(getattr(cfg, "temporal_context_bins", 0))
        if getattr(cfg, "temporal_filters", 0) > 0
        else 0
    )
    print(f"Loading synthetic data from {cfg.synth_data_path} …")
    windows = load_synthetic_windows(cfg, context_bins=temporal_context_bins)
    labels = load_synthetic_labels(cfg)
    subjects = load_synthetic_subjects(cfg)
    print(f"  Data layout: {cfg.synth_data_layout}  |  normalize: {cfg.synth_normalize}")
    if cfg.synth_max_trials > 0:
        print(f"  Trial cap: first {cfg.synth_max_trials} trials")
    if cfg.synth_noise_std > 0:
        print(f"  Added deterministic Gaussian noise: std={cfg.synth_noise_std}")

    print(f"  Windows shape: {windows.shape}  (K, N, T including context)")
    if subjects is not None:
        unique_subjects, subject_counts = np.unique(subjects, return_counts=True)
        print(
            f"  Subjects: {len(unique_subjects)}  |  "
            f"Trials/subject: min={subject_counts.min()} "
            f"median={np.median(subject_counts):.0f} max={subject_counts.max()}"
        )
    if labels is not None:
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        print(f"  Labels: {dict(zip(unique_labels.tolist(), label_counts.tolist()))}")

    N = windows.shape[1]
    dataset_split_counts = (
        _load_dataset_split_counts(cfg.synth_data_path, len(windows))
        if cfg.synth_split.lower() == "dataset"
        else None
    )
    train_ds, val_ds, trainval_subjects, eligible_trials, holdout_subjects, holdout_trials = train_val_split_synth(
        windows,
        cfg.val_split,
        cfg.seed,
        cfg.synth_split,
        subjects=subjects,
        subject_count=getattr(cfg, "synth_subject_count", 0),
        subject_ids=getattr(cfg, "synth_subject_ids", ""),
        val_subject_count=getattr(cfg, "synth_val_subject_count", 0),
        val_subject_ids=getattr(cfg, "synth_val_subject_ids", ""),
        holdout_subject_count=getattr(cfg, "synth_holdout_subject_count", 0),
        holdout_subject_ids=getattr(cfg, "synth_holdout_subject_ids", ""),
        dataset_split_counts=dataset_split_counts,
    )
    norm_info = apply_train_zscore(windows, _dataset_source_indices(train_ds), cfg.synth_normalize)
    if norm_info is not None:
        print(
            "  Applied train-only z-score: "
            f"train_trials={norm_info['train_trials']} channels={norm_info['channels']}"
        )
    if trainval_subjects is not None:
        print(
            f"  Train/val subject subset: {len(trainval_subjects)} subjects, "
            f"{eligible_trials} eligible trials"
        )
        print(f"  Train/val subjects: {trainval_subjects.tolist()}")
        if subjects is not None and str(cfg.synth_split).lower() in _SUBJECT_HOLDOUT_SPLITS:
            train_subjects = np.unique(subjects[_dataset_source_indices(train_ds)])
            val_subjects = np.unique(subjects[_dataset_source_indices(val_ds)])
            print(f"  Train subjects: {train_subjects.tolist()}")
            print(f"  Validation held-out subjects: {val_subjects.tolist()}")
        if len(holdout_subjects):
            print(
                f"  Held-out subjects: {holdout_subjects.tolist()} "
                f"({holdout_trials} test-only trials)"
            )
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Split: {cfg.synth_split}")

    model = MLP(
        in_channels=N, d=cfg.d, hidden_dim=cfg.hidden_dim, depth=cfg.depth, dropout=cfg.dropout,
        temporal_filters=cfg.temporal_filters, temporal_kernel_size=cfg.temporal_kernel_size,
        temporal_frontend=getattr(cfg, "temporal_frontend", "symmetric"),
        residual_kernels=getattr(cfg, "residual_kernels", "3,7,15,31"),
        multiscale_symmetric_conv_layers=getattr(cfg, "multiscale_symmetric_conv_layers", 1),
        antisymmetric_planes=getattr(cfg, "antisymmetric_planes", 0),
        temporal_context_bins=temporal_context_bins,
    )
    if model.temporal_conv is not None:
        print(model.temporal_conv.weight.shape)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print(f"\nStarting training for {cfg.epochs} epochs …\n")
    history = train(model, train_ds, val_ds, cfg)

    if model.temporal_conv is not None:
        print(model.temporal_conv.weight)
    
    print_summary(history, cfg)

    skip_diagnostic_plots = os.environ.get("SYNTH_SKIP_DIAGNOSTIC_PLOTS", "").strip().lower()
    if True: #skip_diagnostic_plots in {"1", "true", "yes", "y"}:
        print("\nSkipping synthetic diagnostic plots (SYNTH_SKIP_DIAGNOSTIC_PLOTS=1).")
    else:
        print("\nGenerating synthetic diagnostic plots …")
        best_ckpt_path = os.path.join(cfg.ckpt_dir, "best.pt")
        if os.path.isfile(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(
                "Loaded best checkpoint for diagnostics: "
                f"epoch={ckpt.get('epoch')} "
                f"selection={ckpt.get('checkpoint_selection', 'unknown')} "
                f"val_zeta={ckpt.get('val_zeta', float('nan')):.4f}"
            )
        make_diagnostic_plots_synth(
            model=model,
            val_ds=val_ds,
            cfg=cfg,
            run_dir=run_dir,
            train_ds=train_ds,
            subjects=subjects,
            labels=labels,
        )

    append_best_model_metrics(run_dir, val_ds, cfg)



if __name__ == "__main__":
    main()
