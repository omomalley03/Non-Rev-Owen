import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

from config import Config
from paths import SYNTH_RUNS_DIR
from model import MLP
from train import train
from visualize_synth import make_diagnostic_plots_synth
from best_metrics import append_best_model_metrics
from synth_data import load_synthetic_windows


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
    print(f"  Best val loss       : {history['best_val_loss']:.4f}")
    print(f"  Wall-clock time     : {history['elapsed_s']:.1f} s  ({history['elapsed_s']/60:.1f} min)")
    print(f"  Checkpoint          : {cfg.ckpt_dir}/best.pt")
    print(f"  Loss curve          : {cfg.out_dir}/loss_curve.png")
    print("=" * 50)


def train_val_split_synth(windows: np.ndarray, val_frac: float, seed: int, split: str = "random"):
    """Split synthetic windows for training/validation."""
    tensor = torch.from_numpy(windows)
    full_ds = TensorDataset(tensor)

    split = split.lower()
    if split in {"train_eq_val", "train_equals_val", "all", "none"}:
        return full_ds, full_ds
    if split != "random":
        raise ValueError("SYNTH_SPLIT must be one of: random, train_eq_val")

    n_val = max(1, int(len(tensor) * val_frac))
    n_train = len(tensor) - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(full_ds, [n_train, n_val], generator=generator)


def main():
    cfg = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(SYNTH_RUNS_DIR, f"{timestamp}_{cfg.run_name()}")
    cfg.ckpt_dir = os.path.join(run_dir, "checkpoints")
    cfg.out_dir  = os.path.join(run_dir, "outputs")
    cfg.save_about(run_dir)
    print(f"Run directory: {run_dir}")
    set_seed(cfg.seed)

    print(f"Loading synthetic data from {cfg.synth_data_path} …")
    windows = load_synthetic_windows(cfg)
    print(f"  Data layout: {cfg.synth_data_layout}  |  normalize: {cfg.synth_normalize}")
    if cfg.synth_max_trials > 0:
        print(f"  Trial cap: first {cfg.synth_max_trials} trials")
    if cfg.synth_noise_std > 0:
        print(f"  Added deterministic Gaussian noise: std={cfg.synth_noise_std}")

    print(f"  Windows shape: {windows.shape}  (K, N, T)")

    N = windows.shape[1]
    train_ds, val_ds = train_val_split_synth(windows, cfg.val_split, cfg.seed, cfg.synth_split)
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Split: {cfg.synth_split}")

    model = MLP(in_channels=N, d=cfg.d, hidden_dim=cfg.hidden_dim, depth=cfg.depth, dropout=cfg.dropout,
                temporal_filters=cfg.temporal_filters, temporal_kernel_size=cfg.temporal_kernel_size)
    if model.temporal_conv is not None:
        print(model.temporal_conv.weight.shape)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print(f"\nStarting training for {cfg.epochs} epochs …\n")
    history = train(model, train_ds, val_ds, cfg)

    if model.temporal_conv is not None:
        print(model.temporal_conv.weight)
    
    print_summary(history, cfg)

    print("\nGenerating synthetic diagnostic plots …")
    make_diagnostic_plots_synth(
        model=model,
        val_ds=val_ds,
        cfg=cfg,
        run_dir=run_dir,
    )

    append_best_model_metrics(run_dir, val_ds, cfg)



if __name__ == "__main__":
    main()
