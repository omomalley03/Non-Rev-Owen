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
from synth_data import load_synthetic_labels, load_synthetic_subjects, load_synthetic_windows


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
    print(f"  Best val ζ          : {history['best_val_zeta']:.4f}")
    print(f"  Val loss at best ζ  : {history['best_checkpoint_val_loss']:.4f}")
    print(f"  Wall-clock time     : {history['elapsed_s']:.1f} s  ({history['elapsed_s']/60:.1f} min)")
    print(f"  Checkpoint          : {cfg.ckpt_dir}/best.pt")
    print(f"  Loss curve          : {cfg.out_dir}/loss_curve.png")
    print("=" * 50)


def _parse_subject_ids(spec: str) -> np.ndarray:
    spec = str(spec or "").strip()
    if not spec or spec.lower() in {"all", "none"}:
        return np.array([], dtype=np.int64)
    return np.array([int(item.strip()) for item in spec.split(",") if item.strip()], dtype=np.int64)


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


def train_val_split_synth(
    windows: np.ndarray,
    val_frac: float,
    seed: int,
    split: str = "random",
    subjects: Optional[np.ndarray] = None,
    subject_count: int = 0,
    subject_ids: str = "",
    holdout_subject_count: int = 0,
    holdout_subject_ids: str = "",
):
    """Split synthetic windows for training/validation."""
    tensor = torch.from_numpy(windows)
    full_ds = TensorDataset(tensor)

    split = split.lower()
    if split not in {"subject_random", "participant_random"} and (
        holdout_subject_count or _parse_subject_ids(holdout_subject_ids).size
    ):
        raise ValueError("Subject holdout requires SYNTH_SPLIT=subject_random")

    if split in {"train_eq_val", "train_equals_val", "all", "none"}:
        return full_ds, full_ds, None, len(tensor), np.array([], dtype=np.int64), 0
    if split in {"subject_random", "participant_random"}:
        if subjects is None:
            raise ValueError("SYNTH_SPLIT=subject_random requires SYNTH_SUBJECTS_PATH")
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
        raise ValueError("SYNTH_SPLIT must be one of: random, train_eq_val, subject_random")

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

    print(f"Loading synthetic data from {cfg.synth_data_path} …")
    windows = load_synthetic_windows(cfg)
    labels = load_synthetic_labels(cfg)
    subjects = load_synthetic_subjects(cfg)
    print(f"  Data layout: {cfg.synth_data_layout}  |  normalize: {cfg.synth_normalize}")
    if cfg.synth_max_trials > 0:
        print(f"  Trial cap: first {cfg.synth_max_trials} trials")
    if cfg.synth_noise_std > 0:
        print(f"  Added deterministic Gaussian noise: std={cfg.synth_noise_std}")

    print(f"  Windows shape: {windows.shape}  (K, N, T)")
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
    if trainval_subjects is not None:
        print(
            f"  Train/val subject subset: {len(trainval_subjects)} subjects, "
            f"{eligible_trials} eligible trials"
        )
        print(f"  Train/val subjects: {trainval_subjects.tolist()}")
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
