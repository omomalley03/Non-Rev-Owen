import os
import random
import time
from datetime import datetime

import numpy as np
import torch

from config import Config
from data import load_mcmaze, gaussian_smooth, soft_normalize, make_windows, train_val_split
from model import MLP
from train import train
from visualize import make_diagnostic_plots, _hand_windows_from_raw


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


def main():
    cfg = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("synth_runs", f"{timestamp}_{cfg.run_name()}")
    cfg.ckpt_dir = os.path.join(run_dir, "checkpoints")
    cfg.out_dir  = os.path.join(run_dir, "outputs")
    cfg.save_about(run_dir)
    print(f"Run directory: {run_dir}")
    set_seed(cfg.seed)

    # LOADING SYNTHETIC DATA
    print("Loading Synth Data")
    windows = np.load("rotations.npy")
    windows = windows.astype(np.float32)
    windows = np.transpose(windows, (0, 2, 1)) # (K,N,T)
    windows = windows + 0.5 * np.random.randn(*windows.shape).astype(np.float32) # add some noise


    # LOADING MONKEY DATA NPZ
    # print("Loading Monkey Data")
    # data = np.load("monkey_data.npz")
    # windows = data["windows"]  # (K, N, T)

    print(f"  Windows shape: {windows.shape}  (K, N, T)")
    
    N = windows.shape[1]
    train_ds, val_ds = train_val_split(windows, None, cfg.val_split, cfg.seed)
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    model = MLP(in_channels=N, d=cfg.d, hidden_dim=cfg.hidden_dim, depth=cfg.depth)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print(f"\nStarting training for {cfg.epochs} epochs …\n")
    history = train(model, train_ds, val_ds, cfg)

    print_summary(history, cfg)



if __name__ == "__main__":
    main()