"""Evaluate MC Maze val-S embedder checkpoints on hand velocity prediction.

The intended workflow is:

    source mcmaze_config.sh
    python -u mcmaze_val_s_checkpoint_velocity_experiment.py

With no --run, this trains a new embedder using the current MC Maze config. The
training loop saves checkpoints when validation S first reaches each value in
VAL_S_CHECKPOINTS, then this script trains the same hand-velocity decoders on
each checkpoint and plots embedder validation loss vs downstream MLP RMSE.

To reuse an existing run that already has checkpoints/val_s_checkpoints.csv:

    python -u mcmaze_val_s_checkpoint_velocity_experiment.py --run mcmaze/runs/<run>
"""

import argparse
import csv
import os
from dataclasses import asdict

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
    save_metrics,
    standardize_train_val,
    train_linear_regression,
    train_mlp_decoder,
)


def _abs_checkpoint_path(run_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(run_dir, path))


def load_val_s_checkpoint_manifest(run_dir: str):
    manifest_path = os.path.join(run_dir, "checkpoints", "val_s_checkpoints.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"No val-S checkpoint manifest at {manifest_path!r}. "
            "Run embedder training with VAL_S_CHECKPOINTS set first."
        )

    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ckpt_path = _abs_checkpoint_path(run_dir, row["path"])
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Manifest checkpoint does not exist: {ckpt_path}")
            rows.append(
                {
                    "checkpoint_label": f"S>={float(row['threshold']):g}",
                    "checkpoint_kind": "val_s_threshold",
                    "output_name": f"s_{float(row['threshold']):g}",
                    "threshold": float(row["threshold"]),
                    "epoch": int(row["epoch"]),
                    "val_s": float(row["val_s"]),
                    "val_loss": float(row["val_loss"]),
                    "train_loss": float(row["train_loss"]),
                    "train_s": float(row["train_s"]),
                    "path": ckpt_path,
                }
            )

    if not rows:
        raise RuntimeError(f"No checkpoint rows found in {manifest_path}")
    return sorted(rows, key=lambda r: r["threshold"])


def load_best_checkpoint_record(run_dir: str):
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No best checkpoint at {ckpt_path!r}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "checkpoint_label": "best val loss",
        "checkpoint_kind": "best_val_loss",
        "output_name": "best_val_loss",
        "threshold": None,
        "epoch": int(ckpt.get("epoch") or -1),
        "val_s": float(ckpt.get("val_s", "nan")),
        "val_loss": float(ckpt.get("val_loss", "nan")),
        "train_loss": float(ckpt.get("train_loss", "nan")),
        "train_s": float(ckpt.get("train_s", "nan")),
        "path": ckpt_path,
    }


def warn_missing_thresholds(cfg: Config, records):
    requested = sorted(set(float(x) for x in getattr(cfg, "val_s_checkpoint_thresholds", ()) or ()))
    saved = {float(row["threshold"]) for row in records if row.get("threshold") is not None}
    missing = [threshold for threshold in requested if threshold not in saved]
    if missing:
        print(
            "Warning: no checkpoints were saved for requested VAL_S_CHECKPOINTS "
            f"{', '.join(f'{x:g}' for x in missing)}. "
            "The embedder did not reach those validation S thresholds in this run."
        )


def prepare_velocity_problem(cfg: Config, args):
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


def evaluate_checkpoint(run_dir: str, record: dict, problem: dict, args, device: torch.device):
    label = record.get("checkpoint_label") or (
        f"S>={record['threshold']:g}" if record.get("threshold") is not None else "checkpoint"
    )
    print()
    print(
        f"Evaluating checkpoint {label}: "
        f"epoch={record['epoch']} val_loss={record['val_loss']:.4f} val_s={record['val_s']:.4f}"
    )

    ckpt = torch.load(record["path"], map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]
    model = build_model_from_checkpoint(cfg, ckpt["model_state_dict"], problem["spikes_raw"].shape[0])

    print(f"Computing frozen embeddings on {device}...")
    F_train = compute_embeddings(model, problem["train_ds"], args.embed_batch_size, device)
    F_val = compute_embeddings(model, problem["val_ds"], args.embed_batch_size, device)

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
        f"velocity_prediction_val_s_checkpoints_{args.horizon_ms}ms",
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
            "checkpoint_label": label,
            "checkpoint_kind": record.get("checkpoint_kind"),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": mlp_info["target_mean"],
            "target_std": mlp_info["target_std"],
            "velocity_meta": problem["vel_meta"],
            "config": asdict(cfg),
            "metrics": [linear_metrics, mlp_metrics],
        },
        os.path.join(out_dir, "mlp_decoder.pt"),
    )

    return {
        "checkpoint_label": label,
        "checkpoint_kind": record.get("checkpoint_kind", "unknown"),
        "threshold": record.get("threshold"),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_path": record["path"],
        "embedding_val_loss": float(ckpt.get("val_loss", record["val_loss"])),
        "embedding_val_s": float(ckpt.get("val_s", record["val_s"])),
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


def plot_loss_vs_rmse(out_dir: str, rows):
    x = np.asarray([row["embedding_val_loss"] for row in rows], dtype=float)
    y = np.asarray([row["mlp_rmse_mean"] for row in rows], dtype=float)
    labels = [row.get("checkpoint_label") or f"S>={row['threshold']:g}" for row in rows]
    r = _pearson(x, y)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(x, y, s=58, color="steelblue")
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=8)
    if len(x) >= 2 and np.std(x) >= 1e-12:
        m, b = np.polyfit(x, y, deg=1)
        xx = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(xx, m * xx + b, color="black", lw=1, alpha=0.75)
    ax.set_xlabel("Embedder validation loss")
    ax.set_ylabel("Hand velocity prediction RMSE (MLP)")
    ax.set_title(f"MC Maze checkpoint quality vs downstream RMSE\nPearson r={r:.3f}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = os.path.join(out_dir, "embedding_val_loss_vs_velocity_rmse.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.run is None:
        run_dir, _ = train_embedder()
    else:
        run_dir = resolve_run_dir(args.run)
    print(f"Using embedder run: {run_dir}")

    records = load_val_s_checkpoint_manifest(run_dir)
    if args.include_best:
        records.append(load_best_checkpoint_record(run_dir))
    print("Val-S checkpoints:")
    for record in records:
        label = record.get("checkpoint_label") or f"S>={record['threshold']:g}"
        print(
            f"  {label}: epoch={record['epoch']} "
            f"val_loss={record['val_loss']:.4f} val_s={record['val_s']:.4f}"
        )

    first_ckpt = torch.load(records[0]["path"], map_location="cpu", weights_only=False)
    cfg: Config = first_ckpt["config"]
    warn_missing_thresholds(cfg, records)
    problem = prepare_velocity_problem(cfg, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [evaluate_checkpoint(run_dir, record, problem, args, device) for record in records]

    out_dir = os.path.join(run_dir, "outputs", f"velocity_prediction_val_s_checkpoints_{args.horizon_ms}ms")
    os.makedirs(out_dir, exist_ok=True)
    save_summary_csv(os.path.join(out_dir, "summary.csv"), rows)
    plot_loss_vs_rmse(out_dir, rows)

    r = _pearson(
        [row["embedding_val_loss"] for row in rows],
        [row["mlp_rmse_mean"] for row in rows],
    )
    print()
    print(f"Pearson r(embedder val loss, MLP velocity RMSE): {r:.4f}")


if __name__ == "__main__":
    main()
