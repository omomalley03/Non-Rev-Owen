from __future__ import annotations

"""Fine-tune an MC Maze embedder for future hand velocity prediction.

This is the end-to-end counterpart to ``predict_mcmaze_velocity.py``.  It
starts from a trained embedding checkpoint, trains a frozen-embedding decoder
baseline for pretrained embedders, then fine-tunes the embedder and the same
per-snapshot MLP decoder jointly.  With ``--embedder-init random`` it skips the
frozen baseline and trains the embedder plus decoder end to end from random
initialization on:

    embedding(raw_neural_window)[t] -> hand_vel(t + horizon)
"""

import argparse
import csv
import os
import random
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import Config
from data import gaussian_smooth, load_mcmaze_cached, make_windows, soft_normalize, train_val_split
from predict_mcmaze_velocity import (
    VelocityMLP,
    compute_features,
    flatten_samples,
    future_velocity_windows,
    load_hand_velocity_resampled,
    plot_predictions,
    regression_metrics,
    resolve_run_dir,
    save_decoder_loss_curve,
    save_metrics,
    standardize_train_val,
    train_mlp_decoder,
)
from model import MLP, infer_multiscale_symmetric_conv_layers


class WindowVelocityDataset(Dataset):
    """Trial-window dataset returning neural input, target velocity, and mask."""

    def __init__(self, windows: np.ndarray, targets: np.ndarray, valid: np.ndarray, indices):
        self.indices = list(indices)
        self.windows = torch.from_numpy(windows[self.indices].astype(np.float32))
        self.targets = torch.from_numpy(targets[self.indices].astype(np.float32))
        self.valid = torch.from_numpy(valid[self.indices].astype(bool))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.windows[idx], self.targets[idx], self.valid[idx]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint(run_dir: str, arg_checkpoint: str | None) -> str:
    if arg_checkpoint is None:
        return os.path.join(run_dir, "checkpoints", "best.pt")
    if os.path.isabs(arg_checkpoint):
        return arg_checkpoint
    direct = os.path.abspath(arg_checkpoint)
    if os.path.isfile(direct):
        return direct
    return os.path.join(run_dir, "checkpoints", arg_checkpoint)


def maybe_cap_indices(indices, max_count: int, seed: int):
    indices = list(indices)
    if max_count <= 0 or len(indices) <= max_count:
        return indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=max_count, replace=False)
    return sorted(int(i) for i in chosen)


def infer_temporal_frontend(cfg: Config, state_dict) -> str:
    """Infer checkpoint frontend variant so random/pretrained models match."""
    temporal_frontend = getattr(cfg, "temporal_frontend", "symmetric")
    if "temporal_conv.weight" in state_dict:
        return "symmetric"
    if any(k.startswith("temporal_conv.temporal_branches.0.norm.") for k in state_dict):
        return "residual"
    if any(k.startswith("temporal_conv.temporal_branches.0.conv.") for k in state_dict):
        return "multiscale_symmetric"
    return temporal_frontend


def build_embedder(cfg: Config, state_dict, in_channels: int, init: str):
    """Build the embedder architecture, optionally loading pretrained weights."""
    temporal_frontend = infer_temporal_frontend(cfg, state_dict)
    model = MLP(
        in_channels=in_channels,
        d=cfg.d,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        dropout=getattr(cfg, "dropout", 0.0),
        temporal_filters=getattr(cfg, "temporal_filters", 0),
        temporal_kernel_size=getattr(cfg, "temporal_kernel_size", 31),
        temporal_frontend=temporal_frontend,
        residual_kernels=getattr(cfg, "residual_kernels", "3,7,15,31"),
        multiscale_symmetric_conv_layers=infer_multiscale_symmetric_conv_layers(
            state_dict,
            getattr(cfg, "multiscale_symmetric_conv_layers", 1),
        ),
        antisymmetric_planes=getattr(cfg, "antisymmetric_planes", 0),
    )
    if init == "pretrained":
        model.load_state_dict(state_dict)
    elif init != "random":
        raise ValueError(f"embedder init must be 'pretrained' or 'random', got {init!r}")
    return model


def clone_state_dict_cpu(model: torch.nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def flatten_batch(F, targets, valid, feature_mean, feature_std, target_mean, target_std):
    """Return standardized decoder inputs/targets for valid samples in a batch."""
    F = (F - feature_mean) / feature_std
    y = (targets - target_mean) / target_std
    X_flat = F.permute(0, 2, 1).reshape(-1, F.shape[1])
    y_flat = y.permute(0, 2, 1).reshape(-1, 2)
    keep = valid.reshape(-1)
    return X_flat[keep], y_flat[keep]


def embedder_hidden_features(embedder, windows):
    """Differentiable penultimate-layer features with the final projection removed."""
    def hidden_net_without_projection(net, name):
        if net is None or len(net) <= 1 or not isinstance(net[-1], nn.Linear):
            raise ValueError(f"Cannot remove final linear layer: {name} does not end with nn.Linear")
        return net[:-1]

    def apply_pointwise_hidden(x, hidden_net):
        B, C, T = x.shape
        x = x.permute(0, 2, 1).reshape(B * T, C)
        H = hidden_net(x)
        hidden_dim = H.shape[1]
        return H.reshape(B, T, hidden_dim).permute(0, 2, 1)

    x = windows
    if getattr(embedder, "mixed_parity", False):
        x_sym, x_anti = embedder.temporal_conv(x)
        parts = []
        if embedder.sym_net is not None:
            parts.append(apply_pointwise_hidden(x_sym, hidden_net_without_projection(embedder.sym_net, "embedder.sym_net")))
        if embedder.anti_net is not None:
            parts.append(apply_pointwise_hidden(x_anti, hidden_net_without_projection(embedder.anti_net, "embedder.anti_net")))
        return torch.cat(parts, dim=1)

    if embedder.temporal_conv is not None:
        x = embedder.temporal_conv(x)
    return apply_pointwise_hidden(x, hidden_net_without_projection(embedder.net, "embedder.net"))


def embedder_features(embedder, windows, feature_layer: str):
    if feature_layer == "output":
        return embedder(windows)
    if feature_layer == "hidden":
        return embedder_hidden_features(embedder, windows)
    raise ValueError(f"unknown feature_layer {feature_layer!r}")


def train_finetuned_model(
    embedder,
    decoder,
    train_loader,
    val_loader,
    feature_mean,
    feature_std,
    target_mean,
    target_std,
    args,
    device,
):
    embedder = embedder.to(device)
    decoder = decoder.to(device)
    feature_mean = torch.from_numpy(feature_mean.astype(np.float32)).to(device).reshape(1, -1, 1)
    feature_std = torch.from_numpy(feature_std.astype(np.float32)).to(device).reshape(1, -1, 1)
    target_mean = torch.from_numpy(target_mean.astype(np.float32)).to(device).reshape(1, 2, 1)
    target_std = torch.from_numpy(target_std.astype(np.float32)).to(device).reshape(1, 2, 1)

    optimizer = torch.optim.AdamW(
        [
            {"params": embedder.parameters(), "lr": args.embedder_lr},
            {"params": decoder.parameters(), "lr": args.decoder_lr},
        ],
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_embedder = None
    best_decoder = None
    history = []

    def evaluate():
        embedder.eval()
        decoder.eval()
        losses = []
        preds = []
        truths = []
        with torch.no_grad():
            for windows, targets, valid in val_loader:
                windows = windows.to(device)
                targets = targets.to(device)
                valid = valid.to(device)
                F = embedder_features(embedder, windows, args.feature_layer)
                X_flat, y_flat = flatten_batch(
                    F, targets, valid, feature_mean, feature_std, target_mean, target_std
                )
                if X_flat.numel() == 0:
                    continue
                pred_z = decoder(X_flat)
                losses.append(loss_fn(pred_z, y_flat).item())
                pred = pred_z * target_std.reshape(1, 2) + target_mean.reshape(1, 2)
                true = y_flat * target_std.reshape(1, 2) + target_mean.reshape(1, 2)
                preds.append(pred.cpu().numpy())
                truths.append(true.cpu().numpy())

        pred_all = np.concatenate(preds, axis=0) if preds else np.empty((0, 2), dtype=np.float32)
        true_all = np.concatenate(truths, axis=0) if truths else np.empty((0, 2), dtype=np.float32)
        return (float(np.mean(losses)) if losses else float("nan")), true_all, pred_all

    for epoch in range(1, args.epochs + 1):
        embedder.train()
        decoder.train()
        train_losses = []
        for windows, targets, valid in train_loader:
            windows = windows.to(device)
            targets = targets.to(device)
            valid = valid.to(device)
            optimizer.zero_grad()
            F = embedder_features(embedder, windows, args.feature_layer)
            X_flat, y_flat = flatten_batch(
                F, targets, valid, feature_mean, feature_std, target_mean, target_std
            )
            if X_flat.numel() == 0:
                continue
            loss = loss_fn(decoder(X_flat), y_flat)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss, _, _ = evaluate()
        history.append({"epoch": epoch, "train_mse_z": train_loss, "val_mse_z": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_embedder = {k: v.detach().cpu().clone() for k, v in embedder.state_dict().items()}
            best_decoder = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
        print(
            f"Fine-tune epoch {epoch:3d}/{args.epochs}  "
            f"train mse(z)={train_loss:.4f}  val mse(z)={val_loss:.4f}"
        )

    if best_embedder is not None:
        embedder.load_state_dict(best_embedder)
    if best_decoder is not None:
        decoder.load_state_dict(best_decoder)
    _, y_val, pred_val = evaluate()
    return embedder, decoder, history, y_val, pred_val, {"best_val_mse_z": best_loss}


def write_train_log(path: str, history):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_mse_z", "val_mse_z"])
        writer.writeheader()
        writer.writerows(history)


def prepare_mcmaze_problem(cfg: Config, args):
    print("Loading MC Maze data and computing neural windows...")
    spikes_raw, bin_width_s, trial_info, time_index_s, _ = load_mcmaze_cached(cfg.nwb_path, cfg.bin_ms)
    sigma_samples = round((cfg.sigma_ms * 1e-3) / bin_width_s)
    X_smooth = gaussian_smooth(spikes_raw, sigma_samples)
    softnorm = getattr(cfg, "softnorm_method", "none")
    if softnorm and softnorm != "none":
        X_smooth = soft_normalize(X_smooth, method=softnorm)

    windows = make_windows(
        X_smooth,
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
    train_indices = maybe_cap_indices(train_ds.indices, args.max_train_trials, args.seed)
    val_indices = maybe_cap_indices(val_ds.indices, args.max_val_trials, args.seed + 1)
    print(
        f"Trials: train={len(train_indices)} val={len(val_indices)}  "
        f"channels={windows.shape[1]}  window={cfg.window_size} bins"
    )

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
    return spikes_raw, windows, targets, valid, train_indices, val_indices, vel_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="Integer rank, explicit run dir, or omit for newest.")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path/name. Defaults to checkpoints/best.pt under --run.")
    parser.add_argument("--embedder-init", choices=["pretrained", "random"], default="pretrained",
                        help="Initialize the embedder from checkpoint weights, or skip the frozen baseline and train end-to-end from random weights.")
    parser.add_argument("--output-suffix", default=None,
                        help="Optional suffix for the output directory name.")
    parser.add_argument("--horizon-ms", type=int, default=100)
    parser.add_argument("--velocity-scale", choices=["stored", "si"], default="stored")
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--feature-layer", choices=["output", "hidden"], default="output",
                        help="Use normal embedder output or remove the final linear layer and decode from hidden features.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trial-batch-size", type=int, default=64)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--decoder-lr", type=float, default=1e-3)
    parser.add_argument("--embedder-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-trials", type=int, default=0)
    parser.add_argument("--max-val-trials", type=int, default=0)
    parser.add_argument("--max-plot-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    run_dir = resolve_run_dir(args.run)
    ckpt_path = resolve_checkpoint(run_dir, args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path!r}")
    print(f"Using run: {run_dir}")
    print(f"Using checkpoint: {ckpt_path}")
    print(f"Embedder initialization: {args.embedder_init}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spikes_raw, windows, targets, valid, train_indices, val_indices, vel_meta = prepare_mcmaze_problem(cfg, args)
    train_ds = WindowVelocityDataset(windows, targets, valid, train_indices)
    val_ds = WindowVelocityDataset(windows, targets, valid, val_indices)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.trial_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.trial_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    initial_embedder = build_embedder(
        cfg,
        ckpt["model_state_dict"],
        spikes_raw.shape[0],
        init=args.embedder_init,
    )
    initial_embedder_state = clone_state_dict_cpu(initial_embedder)
    frozen_train_subset = torch.utils.data.Subset(
        torch.utils.data.TensorDataset(torch.from_numpy(windows)),
        train_indices,
    )
    frozen_val_subset = torch.utils.data.Subset(
        torch.utils.data.TensorDataset(torch.from_numpy(windows)),
        val_indices,
    )

    print(
        f"Computing {args.embedder_init}-init frozen {args.feature_layer} features "
        f"on {device} for feature statistics..."
    )
    frozen_embedder = initial_embedder
    F_train = compute_features(
        frozen_embedder, frozen_train_subset, args.embed_batch_size, device, args.feature_layer
    )
    F_val = compute_features(
        frozen_embedder, frozen_val_subset, args.embed_batch_size, device, args.feature_layer
    )
    X_train, y_train = flatten_samples(F_train, targets, valid, train_indices)
    X_val, y_val = flatten_samples(F_val, targets, valid, val_indices)
    X_train_std, X_val_std, feature_mean, feature_std = standardize_train_val(X_train, X_val)
    target_mean = y_train.mean(axis=0, keepdims=True)
    target_std = np.maximum(y_train.std(axis=0, keepdims=True), 1e-6)
    decoder_in_dim = int(X_train.shape[1])
    print(f"Decoder samples: train={len(X_train):,} val={len(X_val):,}  features=d{decoder_in_dim}")

    feature_suffix = "_hidden" if args.feature_layer == "hidden" else ""
    if args.output_suffix is not None:
        out_name = f"velocity_prediction_finetune_{args.output_suffix}_{args.horizon_ms}ms{feature_suffix}"
    elif args.embedder_init == "random":
        out_name = f"velocity_prediction_finetune_random_init_{args.horizon_ms}ms{feature_suffix}"
    else:
        out_name = f"velocity_prediction_finetune_{args.horizon_ms}ms{feature_suffix}"
    out_dir = os.path.join(run_dir, "outputs", out_name)
    os.makedirs(out_dir, exist_ok=True)

    run_frozen_baseline = args.embedder_init != "random"
    pred_frozen = None
    frozen_decoder = None
    frozen_info = None
    frozen_metrics = None

    if run_frozen_baseline:
        print(f"Training {args.embedder_init}-init frozen-embedding MLP baseline...")
        pred_frozen, frozen_decoder, frozen_info = train_mlp_decoder(
            X_train_std,
            y_train,
            X_val_std,
            y_val,
            hidden_dim=args.mlp_hidden_dim,
            depth=args.mlp_depth,
            dropout=args.mlp_dropout,
            epochs=args.epochs,
            batch_size=4096,
            lr=args.decoder_lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=device,
        )
        frozen_metrics = {
            "model": "frozen_mlp",
            "embedder_init": args.embedder_init,
            "feature_layer": args.feature_layer,
            "feature_dim": decoder_in_dim,
            **regression_metrics(y_val, pred_frozen),
            "best_val_mse_z": float(frozen_info["best_val_mse_z"]),
        }
        save_decoder_loss_curve(
            out_dir,
            frozen_info["history"],
            stem="frozen_mlp_loss_curve",
            title=f"{args.embedder_init}-init frozen-embedding MLP decoder loss",
        )
    else:
        print("Skipping frozen decoder baseline for random embedder init.")
        print("Training embedder and decoder end-to-end from random initialization.")

    print(f"Fine-tuning {args.embedder_init}-init embedder + MLP decoder...")
    finetune_embedder = build_embedder(
        cfg,
        ckpt["model_state_dict"],
        spikes_raw.shape[0],
        init=args.embedder_init,
    )
    finetune_embedder.load_state_dict(initial_embedder_state)
    finetune_decoder = VelocityMLP(
        decoder_in_dim,
        hidden_dim=args.mlp_hidden_dim,
        depth=args.mlp_depth,
        dropout=args.mlp_dropout,
    )
    if frozen_decoder is not None:
        finetune_decoder.load_state_dict(
            {k: v.detach().cpu().clone() for k, v in frozen_decoder.state_dict().items()}
        )
        print("Initialized fine-tune decoder from the trained frozen-embedding decoder.")
    finetune_embedder, finetune_decoder, history, y_val_ft, pred_ft, finetune_info = train_finetuned_model(
        finetune_embedder,
        finetune_decoder,
        train_loader,
        val_loader,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        args,
        device,
    )
    finetuned_metrics = {
        "model": "finetuned_mlp",
        "embedder_init": args.embedder_init,
        "feature_layer": args.feature_layer,
        "feature_dim": decoder_in_dim,
        **regression_metrics(y_val_ft, pred_ft),
        "best_val_mse_z": float(finetune_info["best_val_mse_z"]),
    }
    save_decoder_loss_curve(
        out_dir,
        history,
        stem="finetuned_mlp_loss_curve",
        title=f"{args.embedder_init}-init fine-tuned MLP decoder loss",
    )

    rows = ([frozen_metrics] if frozen_metrics is not None else []) + [finetuned_metrics]
    save_metrics(out_dir, rows)
    write_train_log(os.path.join(out_dir, "train_log.csv"), history)
    val_predictions = {"finetuned_mlp": pred_ft}
    if pred_frozen is not None:
        val_predictions = {"frozen_mlp": pred_frozen, **val_predictions}
    plot_predictions(
        out_dir,
        y_val_ft,
        val_predictions,
        max_points=args.max_plot_points,
        seed=args.seed,
    )

    torch.save(
        {
            "embedder_state_dict": finetune_embedder.cpu().state_dict(),
            "decoder_state_dict": finetune_decoder.cpu().state_dict(),
            "frozen_decoder_state_dict": (
                frozen_decoder.cpu().state_dict() if frozen_decoder is not None else None
            ),
            "ran_frozen_baseline": run_frozen_baseline,
            "args": vars(args),
            "checkpoint_run_dir": run_dir,
            "checkpoint_path": ckpt_path,
            "checkpoint_epoch": ckpt.get("epoch"),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_layer": args.feature_layer,
            "feature_dim": decoder_in_dim,
            "target_mean": target_mean,
            "target_std": target_std,
            "velocity_meta": vel_meta,
            "embedder_init": args.embedder_init,
            "config": asdict(cfg),
            "metrics": rows,
            "frozen_decoder_history": frozen_info["history"] if frozen_info is not None else None,
            "history": history,
        },
        os.path.join(out_dir, "finetuned_model.pt"),
    )
    print(f"Saved fine-tuned model: {os.path.join(out_dir, 'finetuned_model.pt')}")

    print()
    print("Validation metrics:")
    for row in rows:
        print(
            f"  {row['model']:<18} "
            f"R2 mean={row['r2_mean']:.4f}  "
            f"RMSE mean={row['rmse_mean']:.4f}  "
            f"corr x/y={row['corr_x']:.3f}/{row['corr_y']:.3f}"
        )


if __name__ == "__main__":
    main()
