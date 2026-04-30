import os
import random
import time
from datetime import datetime

import numpy as np
import torch

from config import Config
from data import load_mcmaze, gaussian_smooth, make_windows, train_val_split
from model import MLP
from train import train


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
    print(f"  Final train S_ratio : {history['train_S_ratio'][-1]:.4f}")
    print(f"  Best val S_ratio    : {history['best_val_S_ratio']:.4f}")
    print(f"  Wall-clock time     : {history['elapsed_s']:.1f} s  ({history['elapsed_s']/60:.1f} min)")
    print(f"  Checkpoint          : {cfg.ckpt_dir}/best.pt")
    print(f"  Loss curve          : {cfg.out_dir}/loss_curve.png")
    print("=" * 50)


def main():
    cfg = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{timestamp}_{cfg.run_name()}")
    cfg.ckpt_dir = os.path.join(run_dir, "checkpoints")
    cfg.out_dir  = os.path.join(run_dir, "outputs")
    cfg.save_about(run_dir)
    print(f"Run directory: {run_dir}")
    set_seed(cfg.seed)

    print("Loading MC_Maze data …")
    spikes_raw, bin_width_s, trial_info, time_index_s = load_mcmaze(cfg.nwb_path, cfg.bin_ms)
    N = spikes_raw.shape[0]
    print(f"  Channels: {N}  |  Bin width: {cfg.bin_ms} ms  |  Trials: {len(trial_info)}")

    sigma_samples = round((cfg.sigma_ms * 1e-3) / bin_width_s)
    print(f"  Gaussian smoothing sigma: {cfg.sigma_ms} ms = {sigma_samples} bins")
    X_smooth = gaussian_smooth(spikes_raw, sigma_samples)

    print(f"Windowing ({cfg.window_strategy}, T={cfg.window_size}) …")
    windows = make_windows(
        X_smooth, trial_info, time_index_s, bin_width_s,
        strategy=cfg.window_strategy, window_size=cfg.window_size,
    )
    print(f"  Windows shape: {windows.shape}  (K, N, T)")

    train_ds, val_ds = train_val_split(windows, cfg.val_split, cfg.seed)
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    model = MLP(in_channels=N, d=cfg.d, hidden_dim=cfg.hidden_dim, depth=cfg.depth)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print(f"\nStarting training for {cfg.epochs} epochs …\n")
    history = train(model, train_ds, val_ds, cfg)

    print_summary(history, cfg)


if __name__ == "__main__":
    main()
