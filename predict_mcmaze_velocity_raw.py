"""Predict future MC Maze hand velocity directly from neural input channels.

This is the raw-input counterpart to ``predict_mcmaze_velocity.py``.  Instead
of first passing the 137-channel MC Maze neural window through a frozen
embedding model, it trains the same per-snapshot MLP decoder architecture on
the neural vector at each time step:

    neural_channels(t) -> hand_vel(t + horizon)

The data split, windowing, smoothing, and soft-normalization come from the
current MC Maze ``Config`` / environment.
"""

import argparse
import os
from dataclasses import asdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from config import Config
from data import gaussian_smooth, load_mcmaze_cached, make_windows, soft_normalize, train_val_split
from paths import DATASET_NAME, RUNS_BASE
from predict_mcmaze_velocity import (
    flatten_samples,
    future_velocity_windows,
    load_hand_velocity_resampled,
    maybe_subsample,
    plot_predictions,
    regression_metrics,
    save_metrics,
    standardize_train_val,
    train_linear_regression,
    train_mlp_decoder,
)


def run_name(cfg: Config, args) -> str:
    return (
        f"raw_din{args.input_dim_tag}_h{args.mlp_hidden_dim}_dep{args.mlp_depth}"
        f"_ep{args.mlp_epochs}_hz{args.horizon_ms}_s{args.seed}"
    )


def make_out_dir(cfg: Config, args) -> str:
    root = os.path.join(RUNS_BASE, DATASET_NAME, "raw_velocity_runs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, f"{timestamp}_{run_name(cfg, args)}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_about(out_dir: str, cfg: Config, args, vel_meta: dict, input_dim: int):
    lines = [
        f"out_dir : {out_dir}",
        "",
        "[raw_velocity]",
        f"  input_dim            = {input_dim}",
        f"  horizon_ms           = {args.horizon_ms}",
        f"  velocity_scale       = {args.velocity_scale}",
        f"  velocity_units       = {vel_meta['units']}",
        f"  mlp_hidden_dim       = {args.mlp_hidden_dim}",
        f"  mlp_depth            = {args.mlp_depth}",
        f"  mlp_dropout          = {args.mlp_dropout}",
        f"  mlp_epochs           = {args.mlp_epochs}",
        f"  mlp_batch_size       = {args.mlp_batch_size}",
        f"  mlp_lr               = {args.mlp_lr}",
        f"  mlp_weight_decay     = {args.mlp_weight_decay}",
        f"  max_train_samples    = {args.max_train_samples}",
        f"  seed                 = {args.seed}",
        "",
        "[mcmaze_config]",
    ]
    for key, value in asdict(cfg).items():
        lines.append(f"  {key:<20} = {value}")
    with open(os.path.join(out_dir, "about.txt"), "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-ms", type=int, default=100)
    parser.add_argument("--velocity-scale", choices=["stored", "si"], default="stored")
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--mlp-batch-size", type=int, default=4096)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-plot-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config()
    print("Loading MC Maze data and building raw neural windows...")
    spikes_raw, bin_width_s, trial_info, time_index_s, _ = load_mcmaze_cached(cfg.nwb_path, cfg.bin_ms)
    print(f"Channels: {spikes_raw.shape[0]}  bin={cfg.bin_ms} ms  trials={len(trial_info)}")

    sigma_samples = round((cfg.sigma_ms * 1e-3) / bin_width_s)
    X = gaussian_smooth(spikes_raw, sigma_samples)
    softnorm = getattr(cfg, "softnorm_method", "none")
    if softnorm and softnorm != "none":
        X = soft_normalize(X, method=softnorm)

    windows = make_windows(
        X,
        trial_info,
        time_index_s,
        bin_width_s,
        strategy=getattr(cfg, "window_strategy", "trial_aligned"),
        window_size=cfg.window_size,
        align_field=getattr(cfg, "align_field", "move_onset_time"),
        pre_ms=getattr(cfg, "pre_ms", 100),
    )
    if getattr(cfg, "split", "dataset") == "random":
        trial_info = trial_info.drop(columns=["split"], errors="ignore")
    train_ds, val_ds = train_val_split(windows, trial_info, cfg.val_split, cfg.seed)
    input_dim = int(windows.shape[1])
    args.input_dim_tag = input_dim
    print(f"Trials: train={len(train_ds)} val={len(val_ds)}  input_dim={input_dim}  window={cfg.window_size}")

    print("Loading hand velocity and building future targets...")
    hand_vel_raw, vel_meta = load_hand_velocity_resampled(cfg.nwb_path, time_index_s, args.velocity_scale)
    targets, valid, horizon_bins = future_velocity_windows(
        hand_vel_raw, trial_info, time_index_s, bin_width_s, cfg, args.horizon_ms
    )
    print(
        f"Target: hand_vel +{args.horizon_ms} ms "
        f"({horizon_bins} bins), units={vel_meta['units']}"
    )
    print(f"Valid future samples: {valid.sum():,}/{valid.size:,}")

    X_train, y_train = flatten_samples(windows[train_ds.indices], targets, valid, train_ds.indices)
    X_val, y_val = flatten_samples(windows[val_ds.indices], targets, valid, val_ds.indices)
    X_train, y_train = maybe_subsample(X_train, y_train, args.max_train_samples, args.seed)
    X_train, X_val, feature_mean, feature_std = standardize_train_val(X_train, X_val)
    print(f"Decoder samples: train={len(X_train):,} val={len(X_val):,}  features={X_train.shape[1]}")

    out_dir = make_out_dir(cfg, args)
    save_about(out_dir, cfg, args, vel_meta, input_dim)
    print(f"Output directory: {out_dir}")

    print("Training linear regression baseline...")
    pred_linear = train_linear_regression(X_train, y_train, X_val)
    linear_metrics = {
        "model": "linear_regression_raw",
        **regression_metrics(y_val, pred_linear),
    }

    print("Training raw-input MLP decoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_mlp, mlp, mlp_info = train_mlp_decoder(
        X_train,
        y_train,
        X_val,
        y_val,
        hidden_dim=args.mlp_hidden_dim,
        depth=args.mlp_depth,
        dropout=args.mlp_dropout,
        epochs=args.mlp_epochs,
        batch_size=args.mlp_batch_size,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        seed=args.seed,
        device=device,
    )
    mlp_metrics = {
        "model": "mlp_raw",
        **regression_metrics(y_val, pred_mlp),
        "best_val_mse_z": float(mlp_info["best_val_mse_z"]),
    }

    save_metrics(out_dir, [linear_metrics, mlp_metrics])
    plot_predictions(
        out_dir,
        y_val,
        {"linear_raw": pred_linear, "mlp_raw": pred_mlp},
        max_points=args.max_plot_points,
        seed=args.seed,
    )

    torch.save(
        {
            "mlp_state_dict": mlp.state_dict(),
            "args": vars(args),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": mlp_info["target_mean"],
            "target_std": mlp_info["target_std"],
            "velocity_meta": vel_meta,
            "config": asdict(cfg),
            "metrics": [linear_metrics, mlp_metrics],
        },
        os.path.join(out_dir, "mlp_raw_decoder.pt"),
    )
    print(f"Saved raw-input MLP decoder: {os.path.join(out_dir, 'mlp_raw_decoder.pt')}")

    print()
    print("Validation metrics:")
    for row in [linear_metrics, mlp_metrics]:
        print(
            f"  {row['model']:<22} "
            f"R2 mean={row['r2_mean']:.4f}  "
            f"RMSE mean={row['rmse_mean']:.4f}  "
            f"corr x/y={row['corr_x']:.3f}/{row['corr_y']:.3f}"
        )


if __name__ == "__main__":
    main()
