"""Evaluate MC Maze zeta-threshold and periodic embedder checkpoints on hand velocity prediction.

The intended workflow is:

    source mcmaze_config.sh
    python -u mcmaze_val_s_checkpoint_velocity_experiment.py

With no --run, this trains a new embedder using the current MC Maze config, but
saves validation-zeta threshold checkpoints every 0.1 increase and periodic
embedder checkpoints every 10 epochs. The script then trains the same
hand-velocity decoders on all of those checkpoints and plots embedder validation
loss vs downstream MLP RMSE, labeling points by validation zeta. It also plots
embedder checkpoint epoch vs downstream MLP RMSE.

To reuse an existing run that already has checkpoints/val_zeta_checkpoints.csv
and checkpoints/epoch_checkpoints.csv:

    python -u mcmaze_val_s_checkpoint_velocity_experiment.py --run mcmaze/runs/<run>
"""

import argparse
import csv
import os
from dataclasses import asdict


CHECKPOINT_METRIC = "zeta"
ZETA_CHECKPOINT_STEP = 0.1
ZETA_CHECKPOINT_MAX = 1.0
EPOCH_CHECKPOINT_INTERVAL = 10


def _threshold_sequence(step: float, max_value: float) -> str:
    count = int(round(max_value / step))
    return ",".join(f"{i * step:.1f}".rstrip("0").rstrip(".") for i in range(1, count + 1))


# Config reads environment-backed defaults at import time. Set this wrapper's
# checkpoint policy before importing Config/main so normal main.py runs are not
# affected.
os.environ["CHECKPOINT_EVERY_EPOCHS"] = str(EPOCH_CHECKPOINT_INTERVAL)
os.environ["VAL_CHECKPOINT_METRIC"] = CHECKPOINT_METRIC
os.environ["VAL_CHECKPOINTS"] = _threshold_sequence(ZETA_CHECKPOINT_STEP, ZETA_CHECKPOINT_MAX)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from data import gaussian_smooth, load_mcmaze_cached, make_windows, soft_normalize, train_val_split
from main import main as train_embedder
from predict_mcmaze_velocity import (
    build_model_from_checkpoint,
    compute_embeddings,
    flatten_samples,
    future_velocity_windows,
    load_hand_velocity_resampled,
    maybe_subsample,
    plot_predictions,
    regression_metrics,
    resolve_run_dir,
    save_decoder_loss_curve,
    save_metrics,
    standardize_train_val,
    train_linear_regression,
    train_mlp_decoder,
)


def _abs_checkpoint_path(run_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(run_dir, path))


def _float_or_nan(row: dict, key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return float("nan")
    return float(value)


def _metric_label(metric: str) -> str:
    return "zeta" if metric == "zeta" else "S"


def _record_metric_value(record: dict, metric: str) -> float:
    return float(record[f"val_{metric}"])


def _output_root(args) -> str:
    base = (
        f"velocity_prediction_zeta{ZETA_CHECKPOINT_STEP:g}_"
        f"epoch{EPOCH_CHECKPOINT_INTERVAL}_checkpoints_{args.horizon_ms}ms"
    )
    feature_layer = getattr(args, "feature_layer", "output")
    if feature_layer == "hidden":
        return f"{base}_hidden"
    return base


def _output_root_for_feature(args, feature_layer: str) -> str:
    old = getattr(args, "feature_layer", "output")
    args.feature_layer = feature_layer
    try:
        return _output_root(args)
    finally:
        args.feature_layer = old


def resolve_nwb_path(path: str) -> str:
    if path and os.path.isfile(path):
        return path

    repo_root = os.path.dirname(os.path.abspath(__file__))
    basename = os.path.basename(path) if path else "sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"
    candidates = [
        os.environ.get("NWB_PATH", ""),
        os.path.join(repo_root, basename),
        os.path.join(os.path.dirname(repo_root), basename),
        os.path.join(os.path.expanduser("~"), basename),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            print(f"Using NWB_PATH fallback: {candidate} (checkpoint had {path!r})")
            return candidate

    searched = ", ".join(candidate for candidate in candidates if candidate)
    raise FileNotFoundError(
        f"NWB file from checkpoint does not exist: {path!r}. "
        f"Also tried: {searched}. Set NWB_PATH to the correct NWB file."
    )


def load_zeta_checkpoint_manifest(run_dir: str):
    manifest_path = os.path.join(run_dir, "checkpoints", "val_zeta_checkpoints.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"No zeta checkpoint manifest at {manifest_path!r}. "
            f"Run embedder training with VAL_CHECKPOINT_METRIC=zeta and "
            f"VAL_CHECKPOINTS={os.environ['VAL_CHECKPOINTS']} first."
        )

    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            threshold = float(row["threshold"])
            ckpt_path = _abs_checkpoint_path(run_dir, row["path"])
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Manifest checkpoint does not exist: {ckpt_path}")
            rows.append(
                {
                    "checkpoint_label": f"zeta >= {threshold:g}",
                    "checkpoint_kind": "val_zeta_threshold",
                    "checkpoint_metric": CHECKPOINT_METRIC,
                    "checkpoint_epoch_interval": None,
                    "output_name": f"zeta_{threshold:g}",
                    "threshold": threshold,
                    "epoch": int(row["epoch"]),
                    "val_s": _float_or_nan(row, "val_s"),
                    "val_zeta": _float_or_nan(row, "val_zeta"),
                    "val_c_plus": _float_or_nan(row, "val_c_plus"),
                    "val_loss": _float_or_nan(row, "val_loss"),
                    "train_loss": _float_or_nan(row, "train_loss"),
                    "train_s": _float_or_nan(row, "train_s"),
                    "train_zeta": _float_or_nan(row, "train_zeta"),
                    "path": ckpt_path,
                }
            )

    if not rows:
        raise RuntimeError(f"No checkpoint rows found in {manifest_path}")
    return sorted(rows, key=lambda r: r["threshold"])


def load_epoch_checkpoint_manifest(run_dir: str):
    manifest_path = os.path.join(run_dir, "checkpoints", "epoch_checkpoints.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"No epoch checkpoint manifest at {manifest_path!r}. "
            f"Run embedder training with CHECKPOINT_EVERY_EPOCHS={EPOCH_CHECKPOINT_INTERVAL} first."
        )

    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            ckpt_path = _abs_checkpoint_path(run_dir, row["path"])
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Manifest checkpoint does not exist: {ckpt_path}")
            rows.append(
                {
                    "checkpoint_label": f"epoch {epoch}",
                    "checkpoint_kind": "epoch_interval",
                    "checkpoint_metric": CHECKPOINT_METRIC,
                    "checkpoint_epoch_interval": int(_float_or_nan(row, "interval")),
                    "output_name": f"epoch_{epoch:04d}",
                    "threshold": None,
                    "epoch": epoch,
                    "val_s": _float_or_nan(row, "val_s"),
                    "val_zeta": _float_or_nan(row, "val_zeta"),
                    "val_c_plus": _float_or_nan(row, "val_c_plus"),
                    "val_loss": _float_or_nan(row, "val_loss"),
                    "train_loss": _float_or_nan(row, "train_loss"),
                    "train_s": _float_or_nan(row, "train_s"),
                    "train_zeta": _float_or_nan(row, "train_zeta"),
                    "path": ckpt_path,
                }
            )

    if not rows:
        raise RuntimeError(f"No checkpoint rows found in {manifest_path}")
    return sorted(rows, key=lambda r: r["epoch"])


def load_experiment_checkpoint_records(run_dir: str):
    records = load_zeta_checkpoint_manifest(run_dir) + load_epoch_checkpoint_manifest(run_dir)
    kind_order = {"val_zeta_threshold": 0, "epoch_interval": 1}
    return sorted(
        records,
        key=lambda r: (
            int(r["epoch"]),
            kind_order.get(r.get("checkpoint_kind"), 99),
            float(r["threshold"]) if r.get("threshold") is not None else float("inf"),
            r["path"],
        ),
    )


def load_best_checkpoint_record(run_dir: str):
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No best checkpoint at {ckpt_path!r}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    selection = ckpt.get("checkpoint_selection", "best_val_loss")
    if selection == "best_val_zeta":
        label = "best val zeta"
        output_name = "best_val_zeta"
    else:
        label = "best val loss"
        output_name = "best_val_loss"
    return {
        "checkpoint_label": label,
        "checkpoint_kind": selection,
        "output_name": output_name,
        "threshold": None,
        "checkpoint_metric": None,
        "checkpoint_epoch_interval": None,
        "epoch": int(ckpt.get("epoch") or -1),
        "val_s": float(ckpt.get("val_s", "nan")),
        "val_zeta": float(ckpt.get("val_zeta", "nan")),
        "val_c_plus": float(ckpt.get("val_c_plus", "nan")),
        "val_loss": float(ckpt.get("val_loss", "nan")),
        "train_loss": float(ckpt.get("train_loss", "nan")),
        "train_s": float(ckpt.get("train_s", "nan")),
        "train_zeta": float(ckpt.get("train_zeta", "nan")),
        "path": ckpt_path,
    }


def prepare_velocity_problem(cfg: Config, args):
    cfg.nwb_path = resolve_nwb_path(cfg.nwb_path)
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

    return {
        "spikes_raw": spikes_raw,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "targets": targets,
        "valid": valid,
        "vel_meta": vel_meta,
    }


def mse_mean(metrics: dict) -> float:
    return 0.5 * (float(metrics["mse_x"]) + float(metrics["mse_y"]))


def compute_hidden_features(model, dataset, batch_size: int, device: torch.device):
    """Compute features from the embedder with the final projection removed."""
    model = model.to(device).eval()
    def hidden_net_without_projection(net, name):
        if net is None or len(net) <= 1 or not isinstance(net[-1], torch.nn.Linear):
            raise ValueError(f"Cannot remove final linear layer: {name} does not end with a hidden-to-output Linear")
        return net[:-1]

    def apply_pointwise_hidden(x, hidden_net):
        B, C, T = x.shape
        x = x.permute(0, 2, 1).reshape(B * T, C)
        H = hidden_net(x)
        hidden_dim = H.shape[1]
        return H.reshape(B, T, hidden_dim).permute(0, 2, 1)

    if getattr(model, "mixed_parity", False):
        sym_hidden_net = (
            hidden_net_without_projection(model.sym_net, "model.sym_net")
            if model.sym_net is not None
            else None
        )
        anti_hidden_net = (
            hidden_net_without_projection(model.anti_net, "model.anti_net")
            if model.anti_net is not None
            else None
        )
    else:
        hidden_net = hidden_net_without_projection(model.net, "model.net")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    with torch.no_grad():
        for (batch,) in loader:
            x = batch.to(device)
            if getattr(model, "mixed_parity", False):
                x_sym, x_anti = model.temporal_conv(x)
                parts = []
                if sym_hidden_net is not None:
                    parts.append(apply_pointwise_hidden(x_sym, sym_hidden_net))
                if anti_hidden_net is not None:
                    parts.append(apply_pointwise_hidden(x_anti, anti_hidden_net))
                H = torch.cat(parts, dim=1)
            else:
                if model.temporal_conv is not None:
                    x = model.temporal_conv(x)
                H = apply_pointwise_hidden(x, hidden_net)
            chunks.append(H.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def compute_checkpoint_features(model, dataset, batch_size: int, device: torch.device, feature_layer: str):
    if feature_layer == "output":
        return compute_embeddings(model, dataset, batch_size, device)
    if feature_layer == "hidden":
        return compute_hidden_features(model, dataset, batch_size, device)
    raise ValueError(f"Unknown feature layer {feature_layer!r}")


def evaluate_checkpoint(run_dir: str, record: dict, problem: dict, args, device: torch.device):
    label = record.get("checkpoint_label") or (
        f"{_metric_label(CHECKPOINT_METRIC)}>={record['threshold']:g}"
        if record.get("threshold") is not None else "checkpoint"
    )
    print()
    print(
        f"Evaluating checkpoint {label}: "
        f"epoch={record['epoch']} val_loss={record['val_loss']:.4f} "
        f"val_s={record['val_s']:.4f} val_zeta={record['val_zeta']:.4f}"
    )

    ckpt = torch.load(record["path"], map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]
    model = build_model_from_checkpoint(cfg, ckpt["model_state_dict"], problem["spikes_raw"].shape[0])

    print(f"Computing frozen {args.feature_layer} features on {device}...")
    F_train = compute_checkpoint_features(model, problem["train_ds"], args.embed_batch_size, device, args.feature_layer)
    F_val = compute_checkpoint_features(model, problem["val_ds"], args.embed_batch_size, device, args.feature_layer)

    X_train, y_train = flatten_samples(
        F_train, problem["targets"], problem["valid"], problem["train_ds"].indices
    )
    X_val, y_val = flatten_samples(
        F_val, problem["targets"], problem["valid"], problem["val_ds"].indices
    )
    X_train, y_train = maybe_subsample(X_train, y_train, args.max_train_samples, args.seed)
    X_train, X_val, feature_mean, feature_std = standardize_train_val(X_train, X_val)
    print(f"Decoder samples: train={len(X_train):,} val={len(X_val):,}  features=d{X_train.shape[1]}")

    out_dir = os.path.join(
        run_dir,
        "outputs",
        _output_root(args),
        record.get("output_name", label.replace(" ", "_")),
    )
    os.makedirs(out_dir, exist_ok=True)

    print("Training linear regression decoder...")
    pred_linear = train_linear_regression(X_train, y_train, X_val)
    linear_metrics = {
        "model": "linear_regression",
        **regression_metrics(y_val, pred_linear),
    }
    linear_metrics["mse_mean"] = mse_mean(linear_metrics)

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
        **regression_metrics(y_val, pred_mlp),
        "best_val_mse_z": float(mlp_info["best_val_mse_z"]),
    }
    mlp_metrics["mse_mean"] = mse_mean(mlp_metrics)
    save_decoder_loss_curve(out_dir, mlp_info["history"])

    save_metrics(out_dir, [linear_metrics, mlp_metrics])
    if args.plot_predictions:
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
            "checkpoint_path": record["path"],
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_threshold": record.get("threshold"),
            "checkpoint_metric": record.get("checkpoint_metric"),
            "checkpoint_metric_value": _record_metric_value(record, CHECKPOINT_METRIC),
            "checkpoint_epoch_interval": record.get("checkpoint_epoch_interval"),
            "checkpoint_label": label,
            "checkpoint_kind": record.get("checkpoint_kind"),
            "feature_layer": args.feature_layer,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": mlp_info["target_mean"],
            "target_std": mlp_info["target_std"],
            "decoder_history": mlp_info["history"],
            "velocity_meta": problem["vel_meta"],
            "config": asdict(cfg),
            "metrics": [linear_metrics, mlp_metrics],
        },
        os.path.join(out_dir, "mlp_decoder.pt"),
    )

    return {
        "checkpoint_label": label,
        "checkpoint_kind": record.get("checkpoint_kind", "unknown"),
        "checkpoint_metric": record.get("checkpoint_metric"),
        "threshold": record.get("threshold"),
        "checkpoint_metric_value": _record_metric_value(record, CHECKPOINT_METRIC),
        "checkpoint_epoch_interval": record.get("checkpoint_epoch_interval"),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_path": record["path"],
        "feature_layer": args.feature_layer,
        "feature_dim": int(X_train.shape[1]),
        "embedding_train_loss": float(ckpt.get("train_loss", record["train_loss"])),
        "embedding_val_loss": float(ckpt.get("val_loss", record["val_loss"])),
        "embedding_val_s": float(ckpt.get("val_s", record["val_s"])),
        "embedding_val_zeta": float(ckpt.get("val_zeta", record["val_zeta"])),
        "embedding_val_c_plus": float(ckpt.get("val_c_plus", record["val_c_plus"])),
        "linear_mse_mean": linear_metrics["mse_mean"],
        "linear_rmse_mean": linear_metrics["rmse_mean"],
        "linear_r2_mean": linear_metrics["r2_mean"],
        "linear_corr_x": linear_metrics["corr_x"],
        "linear_corr_y": linear_metrics["corr_y"],
        "mlp_mse_mean": mlp_metrics["mse_mean"],
        "mlp_rmse_mean": mlp_metrics["rmse_mean"],
        "mlp_r2_mean": mlp_metrics["r2_mean"],
        "mlp_corr_x": mlp_metrics["corr_x"],
        "mlp_corr_y": mlp_metrics["corr_y"],
        "mlp_best_val_mse_z": mlp_metrics["best_val_mse_z"],
    }


def save_summary_csv(path: str, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {path}")


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def save_scatter(out_dir: str, rows, x_key: str, xlabel: str, title: str, filename: str):
    x = np.asarray([row[x_key] for row in rows], dtype=float)
    y = np.asarray([row["mlp_rmse_mean"] for row in rows], dtype=float)
    checkpoint_epoch = np.asarray([row["checkpoint_epoch"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sc = ax.scatter(x, y, c=checkpoint_epoch, s=45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MLP mean RMSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Checkpoint epoch")
    fig.tight_layout()

    png_path = os.path.join(out_dir, f"{filename}.png")
    svg_path = os.path.join(out_dir, f"{filename}.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {svg_path}")


def plot_checkpoint_scatters(out_dir: str, rows):
    save_scatter(
        out_dir,
        rows,
        "checkpoint_epoch",
        "Embedder checkpoint epoch",
        "MC Maze checkpoint epoch vs decoder RMSE",
        "embedding_checkpoint_epoch_vs_velocity_rmse",
    )
    save_scatter(
        out_dir,
        rows,
        "embedding_val_loss",
        "Embedder validation loss",
        "MC Maze embedder validation loss vs decoder RMSE",
        "embedding_val_loss_vs_velocity_rmse",
    )
    save_scatter(
        out_dir,
        rows,
        "embedding_val_zeta",
        "Embedder validation ζ",
        "MC Maze embedder validation ζ vs decoder RMSE",
        "embedding_val_zeta_vs_velocity_rmse",
    )


def save_hidden_vs_output_comparison(run_dir: str, args, hidden_rows):
    if args.feature_layer != "hidden":
        return

    baseline_dir = os.path.join(run_dir, "outputs", _output_root_for_feature(args, "output"))
    baseline_path = os.path.join(baseline_dir, "summary.csv")
    if not os.path.isfile(baseline_path):
        print(f"No output-layer baseline summary at {baseline_path}; skipping hidden-vs-output comparison.")
        return

    with open(baseline_path, newline="") as f:
        baseline_rows = list(csv.DictReader(f))
    baseline_by_path = {os.path.abspath(row["checkpoint_path"]): row for row in baseline_rows}

    comparison_rows = []
    for hidden in hidden_rows:
        key = os.path.abspath(hidden["checkpoint_path"])
        base = baseline_by_path.get(key)
        if base is None:
            continue
        hidden_rmse = float(hidden["mlp_rmse_mean"])
        output_rmse = float(base["mlp_rmse_mean"])
        comparison_rows.append(
            {
                "checkpoint_label": hidden["checkpoint_label"],
                "checkpoint_kind": hidden["checkpoint_kind"],
                "checkpoint_epoch": hidden["checkpoint_epoch"],
                "checkpoint_path": hidden["checkpoint_path"],
                "embedding_val_loss": hidden["embedding_val_loss"],
                "embedding_val_zeta": hidden["embedding_val_zeta"],
                "output_feature_dim": base.get("feature_dim", ""),
                "hidden_feature_dim": hidden["feature_dim"],
                "output_mlp_rmse_mean": output_rmse,
                "hidden_mlp_rmse_mean": hidden_rmse,
                "hidden_minus_output_rmse": hidden_rmse - output_rmse,
                "hidden_better": hidden_rmse < output_rmse,
            }
        )

    if not comparison_rows:
        print("No matching output-layer baseline rows found; skipping hidden-vs-output comparison.")
        return

    out_dir = os.path.join(run_dir, "outputs", _output_root(args))
    path = os.path.join(out_dir, "hidden_vs_output_comparison.csv")
    save_summary_csv(path, comparison_rows)

    deltas = np.asarray([row["hidden_minus_output_rmse"] for row in comparison_rows], dtype=float)
    hidden_better = int(np.sum(deltas < 0))
    print()
    print("Hidden vs output-layer MLP RMSE:")
    print(f"  matched checkpoints: {len(comparison_rows)}")
    print(f"  hidden better: {hidden_better}/{len(comparison_rows)}")
    print(f"  mean hidden-output RMSE delta: {float(np.mean(deltas)):.4f}")
    print(f"  best hidden-output RMSE delta: {float(np.min(deltas)):.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="Existing run dir. If omitted, train a new embedder first.")
    parser.add_argument("--horizon-ms", type=int, default=100)
    parser.add_argument("--velocity-scale", choices=["stored", "si"], default="stored")
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--mlp-batch-size", type=int, default=4096)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-plot-points", type=int, default=5000)
    parser.add_argument("--plot-predictions", action="store_true")
    parser.add_argument("--include-best", action="store_true",
                        help="Also evaluate checkpoints/best.pt from the embedder run.")
    parser.add_argument("--feature-layer", choices=["output", "hidden"], default="output",
                        help="Use normal embedder output or remove the final linear layer and use hidden features.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.run is None:
        run_dir, _ = train_embedder()
    else:
        run_dir = resolve_run_dir(args.run)
    print(f"Using embedder run: {run_dir}")

    records = load_experiment_checkpoint_records(run_dir)
    if args.include_best:
        records.append(load_best_checkpoint_record(run_dir))
    print("Experiment checkpoints:")
    for record in records:
        label = record.get("checkpoint_label") or f"{_metric_label(CHECKPOINT_METRIC)}>={record['threshold']:g}"
        print(
            f"  {label} [{record.get('checkpoint_kind')}]: epoch={record['epoch']} "
            f"train_loss={record['train_loss']:.4f} "
            f"val_loss={record['val_loss']:.4f} "
            f"val_s={record['val_s']:.4f} val_zeta={record['val_zeta']:.4f}"
        )

    first_ckpt = torch.load(records[0]["path"], map_location="cpu", weights_only=False)
    cfg: Config = first_ckpt["config"]
    problem = prepare_velocity_problem(cfg, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [evaluate_checkpoint(run_dir, record, problem, args, device) for record in records]

    out_dir = os.path.join(run_dir, "outputs", _output_root(args))
    os.makedirs(out_dir, exist_ok=True)
    save_summary_csv(os.path.join(out_dir, "summary.csv"), rows)
    plot_checkpoint_scatters(out_dir, rows)
    save_hidden_vs_output_comparison(run_dir, args, rows)

    r_val_loss = _pearson(
        [row["embedding_val_loss"] for row in rows],
        [row["mlp_rmse_mean"] for row in rows],
    )
    r_epoch = _pearson(
        [row["checkpoint_epoch"] for row in rows],
        [row["mlp_rmse_mean"] for row in rows],
    )
    r_zeta = _pearson(
        [row["embedding_val_zeta"] for row in rows],
        [row["mlp_rmse_mean"] for row in rows],
    )
    print()
    print(f"Pearson r(embedder val loss, MLP velocity RMSE): {r_val_loss:.4f}")
    print(f"Pearson r(embedder checkpoint epoch, MLP velocity RMSE): {r_epoch:.4f}")
    print(f"Pearson r(embedder val zeta, MLP velocity RMSE): {r_zeta:.4f}")


if __name__ == "__main__":
    main()
