"""Predict future MC Maze hand velocity from frozen neural embeddings.

This script loads a trained MC Maze checkpoint, recomputes the neural
embeddings for the same trial windows, and trains two simple decoders:

  1. Linear regression from embedding(t) -> hand_vel(t + horizon)
  2. MLP regression from embedding(t) -> hand_vel(t + horizon)

By default the horizon is 100 ms. Results are written under the selected run's
``outputs/velocity_prediction_{horizon}ms`` directory.

Examples
--------
    source mcmaze_config.sh
    python predict_mcmaze_velocity.py
    python predict_mcmaze_velocity.py --run 2 --horizon-ms 100
    python predict_mcmaze_velocity.py --run mcmaze/runs/my_run --mlp-epochs 100
"""

import argparse
import csv
import os
from dataclasses import asdict

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from data import (
    gaussian_smooth,
    load_mcmaze_cached,
    make_windows,
    soft_normalize,
    train_val_split,
)
from model import MLP, infer_multiscale_symmetric_conv_layers
from paths import RUNS_BASE, RUNS_DIR


class VelocityMLP(nn.Module):
    """Small per-snapshot decoder: embedding vector -> x/y velocity."""

    def __init__(self, in_dim: int, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        layers = []
        dim = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _unique_existing(paths):
    seen = set()
    out = []
    for path in paths:
        path = os.path.abspath(path)
        if path not in seen and os.path.isdir(path):
            seen.add(path)
            out.append(path)
    return out


def resolve_run_dir(arg_run):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    roots = _unique_existing(
        [
            RUNS_DIR,
            os.path.join(RUNS_BASE, "mcmaze", "runs"),
            os.path.join(repo_root, "mcmaze", "runs"),
            os.path.join(repo_root, "runs"),
        ]
    )
    completed = []
    for root in roots:
        for name in os.listdir(root):
            run_dir = os.path.join(root, name)
            ckpt = os.path.join(run_dir, "checkpoints", "best.pt")
            if os.path.isfile(ckpt):
                completed.append((os.path.getmtime(ckpt), run_dir))
    completed = [run_dir for _, run_dir in sorted(completed, reverse=True)]

    if arg_run is None:
        if not completed:
            raise FileNotFoundError("No completed runs with checkpoints/best.pt were found.")
        print("Using most recent run. Available runs:")
        for i, run_dir in enumerate(completed[:10], 1):
            print(f"  [{i}] {os.path.relpath(run_dir, repo_root)}")
        if len(completed) > 10:
            print(f"  ... {len(completed) - 10} more")
        print()
        return completed[0]

    if arg_run.isdigit():
        idx = int(arg_run) - 1
        if idx < 0 or idx >= len(completed):
            raise ValueError(f"--run {arg_run} out of range (1-{len(completed)})")
        return completed[idx]

    return os.path.abspath(arg_run)


def load_hand_velocity_resampled(nwb_path: str, time_index_s: np.ndarray, velocity_scale: str):
    """Read NWB hand_vel and interpolate it onto the spike/bin time grid."""
    with h5py.File(nwb_path, "r") as f:
        if "processing/behavior/hand_vel/data" not in f:
            raise KeyError("NWB does not contain processing/behavior/hand_vel/data")
        group = f["processing/behavior/hand_vel"]
        data = group["data"][:].astype(np.float64)
        timestamps = group["timestamps"][:].astype(np.float64)
        conversion = float(group["data"].attrs.get("conversion", 1.0))
        unit = group["data"].attrs.get("unit", "unknown")
        description = group.attrs.get("description", "")

    if velocity_scale == "si":
        data *= conversion
        units = str(unit)
    else:
        units = "stored NWB values"

    vel = np.vstack(
        [
            np.interp(time_index_s, timestamps, data[:, 0]),
            np.interp(time_index_s, timestamps, data[:, 1]),
        ]
    ).astype(np.float32)
    meta = {
        "source": "processing/behavior/hand_vel",
        "units": units,
        "conversion": conversion,
        "description": description,
    }
    return vel, meta


def future_velocity_windows(
    hand_vel_raw: np.ndarray,
    trial_info,
    time_index_s: np.ndarray,
    bin_width_s: float,
    cfg: Config,
    horizon_ms: int,
):
    """Return target velocity windows and a valid sample mask.

    targets[k, :, t] is hand velocity at feature time t + horizon. Mask entries
    are false if that future time falls outside the trial or global time range.
    """
    horizon_bins = int(round((horizon_ms * 1e-3) / bin_width_s))
    if horizon_bins < 0:
        raise ValueError("horizon_ms must be non-negative")

    target_source = make_windows(
        hand_vel_raw,
        trial_info,
        time_index_s,
        bin_width_s,
        strategy=getattr(cfg, "window_strategy", "trial_aligned"),
        window_size=int(cfg.window_size) + horizon_bins,
        align_field=getattr(cfg, "align_field", "move_onset_time"),
        pre_ms=getattr(cfg, "pre_ms", 100),
    )
    targets = target_source[:, :, horizon_bins:horizon_bins + int(cfg.window_size)]

    align_field = getattr(cfg, "align_field", "move_onset_time")
    if align_field in trial_info.columns:
        align_times = trial_info[align_field].astype(float).values
        pre_bins = int(round(getattr(cfg, "pre_ms", 100) * 1e-3 / bin_width_s))
    else:
        align_times = trial_info["start_time"].astype(float).values
        pre_bins = 0

    end_col = "end_time" if "end_time" in trial_info.columns else "stop_time"
    trial_starts = trial_info["start_time"].astype(float).values
    trial_ends = trial_info[end_col].astype(float).values

    T = int(cfg.window_size)
    all_bins = np.arange(T)
    valid = np.zeros((len(trial_info), T), dtype=bool)
    for k, align_time in enumerate(align_times):
        idx_align = int(np.searchsorted(time_index_s, align_time))
        idx_start = max(0, idx_align - pre_bins)
        target_idx = idx_start + all_bins + horizon_bins
        in_range = target_idx < len(time_index_s)
        clipped = np.minimum(target_idx, len(time_index_s) - 1)
        target_times = time_index_s[clipped]
        in_trial = (
            (target_times >= trial_starts[k] - 0.5 * bin_width_s)
            & (target_times <= trial_ends[k] + 0.5 * bin_width_s)
        )
        valid[k] = in_range & in_trial

    return targets.astype(np.float32), valid, horizon_bins


def build_model_from_checkpoint(cfg: Config, state_dict, in_channels: int):
    temporal_frontend = getattr(cfg, "temporal_frontend", "symmetric")
    if "temporal_conv.weight" in state_dict:
        temporal_frontend = "symmetric"
    elif any(k.startswith("temporal_conv.temporal_branches.0.norm.") for k in state_dict):
        temporal_frontend = "residual"
    elif any(k.startswith("temporal_conv.temporal_branches.0.conv.") for k in state_dict):
        temporal_frontend = "multiscale_symmetric"

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
    model.load_state_dict(state_dict)
    return model


def compute_embeddings(model, dataset, batch_size: int, device: torch.device):
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    with torch.no_grad():
        for (batch,) in loader:
            F = model(batch.to(device)).cpu().numpy()
            chunks.append(F)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _hidden_net_without_projection(net: nn.Module, name: str) -> nn.Module:
    if net is None or len(net) <= 1 or not isinstance(net[-1], nn.Linear):
        raise ValueError(f"Cannot remove final linear layer: {name} does not end with nn.Linear")
    return net[:-1]


def _apply_pointwise_hidden(x: torch.Tensor, hidden_net: nn.Module) -> torch.Tensor:
    B, C, T = x.shape
    x = x.permute(0, 2, 1).reshape(B * T, C)
    H = hidden_net(x)
    hidden_dim = H.shape[1]
    return H.reshape(B, T, hidden_dim).permute(0, 2, 1)


def compute_hidden_features(model, dataset, batch_size: int, device: torch.device):
    """Compute penultimate MLP features by removing the final projection layer."""
    model = model.to(device).eval()
    if getattr(model, "mixed_parity", False):
        sym_hidden_net = (
            _hidden_net_without_projection(model.sym_net, "model.sym_net")
            if model.sym_net is not None
            else None
        )
        anti_hidden_net = (
            _hidden_net_without_projection(model.anti_net, "model.anti_net")
            if model.anti_net is not None
            else None
        )
    else:
        hidden_net = _hidden_net_without_projection(model.net, "model.net")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    with torch.no_grad():
        for (batch,) in loader:
            x = batch.to(device)
            if getattr(model, "mixed_parity", False):
                x_sym, x_anti = model.temporal_conv(x)
                parts = []
                if sym_hidden_net is not None:
                    parts.append(_apply_pointwise_hidden(x_sym, sym_hidden_net))
                if anti_hidden_net is not None:
                    parts.append(_apply_pointwise_hidden(x_anti, anti_hidden_net))
                H = torch.cat(parts, dim=1)
            else:
                if model.temporal_conv is not None:
                    x = model.temporal_conv(x)
                H = _apply_pointwise_hidden(x, hidden_net)
            chunks.append(H.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def compute_features(model, dataset, batch_size: int, device: torch.device, feature_layer: str):
    if feature_layer == "output":
        return compute_embeddings(model, dataset, batch_size, device)
    if feature_layer == "hidden":
        return compute_hidden_features(model, dataset, batch_size, device)
    raise ValueError(f"unknown feature_layer {feature_layer!r}")


def flatten_samples(F: np.ndarray, targets: np.ndarray, valid: np.ndarray, indices):
    indices = np.asarray(list(indices), dtype=int)
    y = targets[indices].transpose(0, 2, 1).reshape(-1, 2)
    X = F.transpose(0, 2, 1).reshape(-1, F.shape[1])
    mask = valid[indices].reshape(-1)
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    keep = mask & finite
    return X[keep], y[keep]


def standardize_train_val(X_train, X_val, eps=1e-6):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (X_train - mean) / std, (X_val - mean) / std, mean, std


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    mse = np.mean(err ** 2, axis=0)
    rmse = np.sqrt(mse)
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    corr = []
    for j in range(y_true.shape[1]):
        if np.std(y_true[:, j]) < 1e-12 or np.std(y_pred[:, j]) < 1e-12:
            corr.append(float("nan"))
        else:
            corr.append(float(np.corrcoef(y_true[:, j], y_pred[:, j])[0, 1]))
    return {
        "mse_x": float(mse[0]),
        "mse_y": float(mse[1]),
        "rmse_x": float(rmse[0]),
        "rmse_y": float(rmse[1]),
        "rmse_mean": float(np.sqrt(np.mean(err ** 2))),
        "r2_x": float(r2[0]),
        "r2_y": float(r2[1]),
        "r2_mean": float(np.mean(r2)),
        "corr_x": corr[0],
        "corr_y": corr[1],
    }


def train_linear_regression(X_train, y_train, X_val):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg.predict(X_val)


def train_mlp_decoder(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_dim: int,
    depth: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
):
    torch.manual_seed(seed)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = np.maximum(y_train.std(axis=0, keepdims=True), 1e-6)
    y_train_z = (y_train - y_mean) / y_std # standardize target for training
    y_val_z = (y_val - y_mean) / y_std

    ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train_z.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_val_t = torch.from_numpy(y_val_z.astype(np.float32)).to(device)

    model = VelocityMLP(X_train.shape[1], hidden_dim, depth, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "train_mse_z": train_loss, "val_mse_z": float(val_loss)})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"MLP epoch {epoch:3d}/{epochs}  "
            f"train mse(z)={train_loss:.4f}  val mse(z)={val_loss:.4f}"
        )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_z = model(X_val_t).cpu().numpy()
    pred = pred_z * y_std + y_mean
    return pred, model, {
        "target_mean": y_mean,
        "target_std": y_std,
        "best_val_mse_z": best_loss,
        "history": history,
    }


def maybe_subsample(X, y, max_samples: int, seed: int):
    if max_samples <= 0 or len(X) <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx], y[idx]


def save_metrics(out_dir, rows):
    path = os.path.join(out_dir, "metrics.csv")
    fieldnames = ["model"]
    for row in rows:
        for key in row:
            if key != "model" and key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metrics: {path}")


def save_decoder_loss_curve(
    out_dir,
    history,
    stem: str = "mlp_loss_curve",
    title: str = "MLP velocity decoder loss",
    make_plot: bool = True,
):
    if not history:
        return

    csv_path = os.path.join(out_dir, f"{stem}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_mse_z", "val_mse_z"])
        writer.writeheader()
        writer.writerows(history)

    if not make_plot:
        print(f"Saved decoder loss log: {csv_path}")
        return

    epochs = [int(row["epoch"]) for row in history]
    train = [float(row["train_mse_z"]) for row in history]
    val = [float(row["val_mse_z"]) for row in history]

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(epochs, train, label="train MSE (z)", color="steelblue", linewidth=1.8)
    ax.plot(epochs, val, label="val MSE (z)", color="tomato", linewidth=1.8)
    best_idx = int(np.nanargmin(val)) if np.isfinite(val).any() else None
    if best_idx is not None:
        ax.scatter(
            [epochs[best_idx]],
            [val[best_idx]],
            s=42,
            color="goldenrod",
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
            label="best val",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE on standardized velocity")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved decoder loss curve: {png_path}")
    print(f"Saved decoder loss log: {csv_path}")


def plot_predictions(out_dir, y_true, predictions, max_points: int, seed: int):
    rng = np.random.default_rng(seed)
    n = min(max_points, len(y_true))
    idx = rng.choice(len(y_true), size=n, replace=False) if len(y_true) > n else np.arange(len(y_true))

    fig, axes = plt.subplots(len(predictions), 2, figsize=(8, 3.2 * len(predictions)), squeeze=False)
    for row, (name, pred) in enumerate(predictions.items()):
        for col, label in enumerate(["x velocity", "y velocity"]):
            ax = axes[row, col]
            ax.scatter(y_true[idx, col], pred[idx, col], s=4, alpha=0.25)
            lo = min(float(y_true[idx, col].min()), float(pred[idx, col].min()))
            hi = max(float(y_true[idx, col].max()), float(pred[idx, col].max()))
            ax.plot([lo, hi], [lo, hi], color="black", lw=1)
            ax.set_title(f"{name}: {label}")
            ax.set_xlabel("true")
            ax.set_ylabel("predicted")
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, "prediction_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def prediction_plots_disabled(args=None) -> bool:
    env_disabled = os.environ.get("NONREV_SKIP_PREDICTION_PLOTS", "").lower() in {"1", "true", "yes"}
    return env_disabled or bool(getattr(args, "disable_plots", False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="Integer rank, explicit run dir, or omit for newest.")
    parser.add_argument("--horizon-ms", type=int, default=100)
    parser.add_argument("--velocity-scale", choices=["stored", "si"], default="stored",
                        help="Use raw stored NWB values or apply NWB conversion to SI units.")
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--feature-layer", choices=["output", "hidden"], default="output",
                        help="Use normal embedder output or remove the final linear layer and use hidden features.")
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--mlp-batch-size", type=int, default=4096)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="Optional random cap for decoder training samples; 0 uses all.")
    parser.add_argument("--max-plot-points", type=int, default=5000)
    parser.add_argument("--disable-plots", action="store_true",
                        help="Skip prediction scatter PNG; decoder loss curve is still written.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path!r}")
    print(f"Using run: {run_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')}")

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
    print(f"Trials: train={len(train_ds)} val={len(val_ds)}  window={cfg.window_size} bins")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(cfg, ckpt["model_state_dict"], spikes_raw.shape[0])

    print(f"Computing frozen {args.feature_layer} features on {device}...")
    F_train = compute_features(model, train_ds, args.embed_batch_size, device, args.feature_layer)
    F_val = compute_features(model, val_ds, args.embed_batch_size, device, args.feature_layer)

    X_train, y_train = flatten_samples(F_train, targets, valid, train_ds.indices)
    X_val, y_val = flatten_samples(F_val, targets, valid, val_ds.indices)
    X_train, y_train = maybe_subsample(X_train, y_train, args.max_train_samples, args.seed)
    X_train, X_val, feature_mean, feature_std = standardize_train_val(X_train, X_val)
    print(f"Decoder samples: train={len(X_train):,} val={len(X_val):,}  features=d{X_train.shape[1]}")

    out_name = f"velocity_prediction_{args.horizon_ms}ms"
    if args.feature_layer == "hidden":
        out_name += "_hidden"
    out_dir = os.path.join(run_dir, "outputs", out_name)
    os.makedirs(out_dir, exist_ok=True)

    print("Training linear regression decoder...")
    pred_linear = train_linear_regression(X_train, y_train, X_val)
    metric_context = {"feature_layer": args.feature_layer, "feature_dim": int(X_train.shape[1])}
    linear_metrics = {
        "model": "linear_regression",
        **metric_context,
        **regression_metrics(y_val, pred_linear),
    }

    print("Training MLP decoder...")
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
        "model": "mlp",
        **metric_context,
        **regression_metrics(y_val, pred_mlp),
        "best_val_mse_z": float(mlp_info["best_val_mse_z"]),
    }
    disable_prediction_plot = prediction_plots_disabled(args)
    save_decoder_loss_curve(out_dir, mlp_info["history"])

    rows = [linear_metrics, mlp_metrics]
    save_metrics(out_dir, rows)
    if disable_prediction_plot:
        print("Skipping decoder prediction scatter because prediction plots are disabled.")
    else:
        plot_predictions(
            out_dir,
            y_val,
            {"linear": pred_linear, "mlp": pred_mlp},
            max_points=args.max_plot_points,
            seed=args.seed,
        )

    torch.save(
        {
            "mlp_state_dict": mlp.state_dict(),
            "args": vars(args),
            "checkpoint_run_dir": run_dir,
            "checkpoint_epoch": ckpt.get("epoch"),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": mlp_info["target_mean"],
            "target_std": mlp_info["target_std"],
            "decoder_history": mlp_info["history"],
            "velocity_meta": vel_meta,
            "config": asdict(cfg),
            "metrics": rows,
        },
        os.path.join(out_dir, "mlp_decoder.pt"),
    )
    print(f"Saved MLP decoder: {os.path.join(out_dir, 'mlp_decoder.pt')}")

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
