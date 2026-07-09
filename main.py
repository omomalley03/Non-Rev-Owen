import os
import random
import time
from datetime import datetime

import numpy as np
import torch

from config import Config
from paths import RUNS_DIR
from data import load_mcmaze_cached, gaussian_smooth, soft_normalize, make_windows, train_val_split
from model import MLP
from train import train
from visualize import make_diagnostic_plots, _hand_windows_from_raw
from best_metrics import append_best_model_metrics


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
    print(f"  Loss curve          : '{cfg.out_dir}/loss_curve.png'")
    print("=" * 50)


def main():
    cfg = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"{timestamp}_{cfg.run_name()}")
    cfg.ckpt_dir = os.path.join(run_dir, "checkpoints")
    cfg.out_dir  = os.path.join(run_dir, "outputs")
    cfg.save_about(run_dir)
    print(f"Run directory: {run_dir}")
    set_seed(cfg.seed)

    print("Loading MC_Maze data …")
    spikes_raw, bin_width_s, trial_info, time_index_s, hand_pos_raw = load_mcmaze_cached(
        cfg.nwb_path, cfg.bin_ms
    )
    N = spikes_raw.shape[0]
    print(f"  Channels: {N}  |  Bin width: {cfg.bin_ms} ms  |  Trials: {len(trial_info)}")

    sigma_samples = round((cfg.sigma_ms * 1e-3) / bin_width_s)
    print(f"  Gaussian smoothing sigma: {cfg.sigma_ms} ms = {sigma_samples} bins")
    X_smooth = gaussian_smooth(spikes_raw, sigma_samples)
    # X_smooth = spikes_raw; print("NO GAUSSIAN SMOOTHING!!")


    if cfg.softnorm_method and cfg.softnorm_method != "none":
        print(f"  Soft-normalising per neuron (method={cfg.softnorm_method!r})")
        X_smooth = soft_normalize(X_smooth, method=cfg.softnorm_method)

    print(f"Windowing ({cfg.window_strategy}, align={cfg.align_field}, "
          f"pre={cfg.pre_ms}ms, T={cfg.window_size}) …")
    windows = make_windows(
        X_smooth, trial_info, time_index_s, bin_width_s,
        strategy=cfg.window_strategy, window_size=cfg.window_size,
        align_field=cfg.align_field, pre_ms=cfg.pre_ms,
    )

    # grand_mean = windows.mean(axis=(0,2), keepdims=True)  # (1, N, T)
    # windows = windows - grand_mean

    print(f"  Windows shape: {windows.shape}  (K, N, T)")
    if cfg.split == "random":
        trial_info = trial_info.drop(columns=["split"], errors="ignore") # drop split to force 90/10 random split
    train_ds, val_ds = train_val_split(windows, trial_info, cfg.val_split, cfg.seed)
    using_split_col = "split" in trial_info.columns
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}  "
          f"({'dataset split column' if using_split_col else f'random {cfg.val_split:.0%}'})")

    model = MLP(
        in_channels=N, d=cfg.d, hidden_dim=cfg.hidden_dim, depth=cfg.depth, dropout=cfg.dropout,
        temporal_filters=cfg.temporal_filters, temporal_kernel_size=cfg.temporal_kernel_size,
        temporal_frontend=getattr(cfg, "temporal_frontend", "symmetric"),
        residual_kernels=getattr(cfg, "residual_kernels", "3,7,15,31"),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print(f"\nStarting training for {cfg.epochs} epochs …\n")
    history = train(model, train_ds, val_ds, cfg)

    print_summary(history, cfg)

    if True:  # os.environ.get("SKIP_DIAGNOSTICS", "").lower() in {"1", "true", "yes"}:
        print("\nSkipping diagnostic plots because SKIP_DIAGNOSTICS=1.")
    else:
        print("\nGenerating diagnostic plots …")
        hand_windows = _hand_windows_from_raw(hand_pos_raw, cfg, trial_info, time_index_s, bin_width_s)
        make_diagnostic_plots(
            model=model,
            val_ds=val_ds,
            trial_info=trial_info,
            cfg=cfg,
            run_dir=run_dir,
            hand_windows=hand_windows,
        )

        append_best_model_metrics(run_dir, val_ds, cfg)

    # from evaluate import run_linear_probe, plot_confusion_matrix

    # probe_results = run_linear_probe(model, train_ds, val_ds, trial_info, cfg, run_dir)
    # plot_confusion_matrix(
    #     probe_results["y_val"], probe_results["y_pred"],
    #     probe_results["label_encoder"], probe_results["val_acc"],
    #     out_path=os.path.join(cfg.out_dir, "11_condition_classifier.png"),
    # )

    return run_dir, history


if __name__ == "__main__":
    main()
