"""Train MC Maze velocity decoders and export predictions for the test NWB.
[OBSOLETE: this script isn't useful]
The test NWB currently contains ecephys/trial intervals but no hand velocity
targets, so this script trains/selects decoder models on the labelled training
NWB and then writes predictions for every trial in the test NWB.

Examples
--------
    source configs/mcmaze_config.sh
    python evaluate_mcmaze_test_velocity.py \
        --model-path mcmaze/runs/.../checkpoints/best.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model.py").is_file())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import csv
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from config import Config
from data import gaussian_smooth, load_mcmaze, load_mcmaze_cached, make_windows, soft_normalize, train_val_split
from decoders.predict_mcmaze_velocity import (
    VelocityMLP,
    compute_features,
    flatten_samples,
    future_velocity_windows,
    load_hand_velocity_resampled,
    plot_predictions,
    regression_metrics,
    save_decoder_loss_curve,
    save_metrics,
    standardize_train_val,
    train_mlp_decoder,
)
from decoders.predict_mcmaze_velocity_finetune import (
    WindowVelocityDataset,
    build_embedder,
    clone_state_dict_cpu,
    train_finetuned_model,
)


DEFAULT_TEST_NWB = REPO_ROOT / "sub-Jenkins_ses-full_desc-test_ecephys.nwb"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cfg_to_dict(cfg) -> dict:
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(getattr(cfg, "__dict__", {}))


def resolve_output_dir(model_path: Path, out_dir: Path | None, horizon_ms: int, feature_layer: str) -> Path:
    if out_dir is not None:
        return out_dir
    if model_path.parent.name == "checkpoints":
        run_dir = model_path.parent.parent
        return run_dir / "outputs" / f"test_velocity_prediction_{horizon_ms}ms_{feature_layer}"
    return Path("outputs") / f"test_velocity_prediction_{model_path.stem}_{horizon_ms}ms_{feature_layer}"


def load_checkpoint(model_path: Path) -> tuple[Config, dict, dict]:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"{model_path} does not look like an embedder checkpoint: "
            "expected key 'model_state_dict'."
        )
    return ckpt["config"], ckpt["model_state_dict"], ckpt


def preprocess_mcmaze(
    nwb_path: str,
    cfg: Config,
    *,
    use_cache: bool,
    label: str,
) -> tuple[np.ndarray, float, object, np.ndarray, np.ndarray]:
    loader = load_mcmaze_cached if use_cache else load_mcmaze
    spikes_raw, bin_width_s, trial_info, time_index_s, _ = loader(nwb_path, cfg.bin_ms)
    sigma_samples = round((float(cfg.sigma_ms) * 1e-3) / bin_width_s)
    X = gaussian_smooth(spikes_raw, sigma_samples)
    softnorm = getattr(cfg, "softnorm_method", "none")
    if softnorm and softnorm != "none":
        X = soft_normalize(X, method=softnorm)

    temporal_context_bins = (
        int(getattr(cfg, "temporal_context_bins", 0))
        if int(getattr(cfg, "temporal_filters", 0)) > 0
        else 0
    )
    windows = make_windows(
        X,
        trial_info,
        time_index_s,
        bin_width_s,
        strategy=getattr(cfg, "window_strategy", "trial_aligned"),
        window_size=int(cfg.window_size),
        align_field=getattr(cfg, "align_field", "move_onset_time"),
        pre_ms=int(getattr(cfg, "pre_ms", 100)),
        context_bins=temporal_context_bins,
    )
    print(
        f"{label}: trials={len(trial_info)} channels={windows.shape[1]} "
        f"window={windows.shape[-1]} bins target_window={cfg.window_size} "
        f"context={temporal_context_bins} bins/side"
    )
    return spikes_raw, bin_width_s, trial_info, time_index_s, windows


def trial_ids(trial_info) -> np.ndarray:
    if "id" in trial_info.columns:
        return trial_info["id"].to_numpy()
    return np.arange(len(trial_info), dtype=np.int64)


def write_prediction_csv(path: Path, trial_id: np.ndarray, time_ms: np.ndarray, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_index", "trial_id", "time_bin", "time_from_alignment_ms", "pred_vx", "pred_vy"])
        for k in range(pred.shape[0]):
            for t in range(pred.shape[1]):
                writer.writerow([k, trial_id[k], t, float(time_ms[t]), float(pred[k, t, 0]), float(pred[k, t, 1])])
    print(f"Saved test prediction CSV: {path}")


def predict_windows(
    embedder,
    decoder,
    windows: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    feature_layer: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(windows.astype(np.float32)))
    F = compute_features(embedder, dataset, batch_size, device, feature_layer)
    X = F.transpose(0, 2, 1).reshape(-1, F.shape[1])
    X = (X - feature_mean) / feature_std
    decoder = decoder.to(device).eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size].astype(np.float32)).to(device)
            pred_z = decoder(xb).cpu().numpy()
            preds.append(pred_z)
    pred_z = np.concatenate(preds, axis=0)
    pred = pred_z * target_std + target_mean
    return pred.reshape(F.shape[0], F.shape[2], 2).astype(np.float32)


def save_prediction_artifacts(
    out_dir: Path,
    stem: str,
    pred: np.ndarray,
    trial_info,
    time_ms: np.ndarray,
    meta: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = trial_ids(trial_info)
    npz_path = out_dir / f"{stem}_test_predictions.npz"
    np.savez(
        npz_path,
        prediction=pred,
        trial_id=ids,
        time_from_alignment_ms=time_ms,
        **meta,
    )
    print(f"Saved test prediction arrays: {npz_path}")
    write_prediction_csv(out_dir / f"{stem}_test_predictions.csv", ids, time_ms, pred)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path, help="Trained embedder checkpoint, e.g. checkpoints/best.pt.")
    parser.add_argument("--train-nwb", default=None, help="Labelled train NWB. Defaults to checkpoint config nwb_path.")
    parser.add_argument("--test-nwb", default=str(DEFAULT_TEST_NWB), help="Unlabelled test NWB to embed and predict.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--horizon-ms", type=int, default=100)
    parser.add_argument("--velocity-scale", choices=["stored", "si"], default="stored")
    parser.add_argument("--feature-layer", choices=["output", "hidden"], default="hidden")
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--trial-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--mlp-dropout", type=float, default=0.3)
    parser.add_argument("--decoder-lr", type=float, default=1e-3)
    parser.add_argument("--embedder-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-trials", type=int, default=0)
    parser.add_argument("--max-val-trials", type=int, default=0)
    parser.add_argument("--max-plot-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    model_path = args.model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"No model checkpoint at {model_path}")
    test_nwb = Path(args.test_nwb).resolve()
    if not test_nwb.is_file():
        raise FileNotFoundError(f"No test NWB at {test_nwb}")

    cfg, state_dict, ckpt = load_checkpoint(model_path)
    train_nwb = args.train_nwb or cfg.nwb_path
    if not os.path.isfile(train_nwb):
        raise FileNotFoundError(f"No labelled train NWB at {train_nwb}")

    out_dir = resolve_output_dir(model_path, args.out_dir, args.horizon_ms, args.feature_layer)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using model checkpoint: {model_path}")
    print(f"Using labelled train NWB: {train_nwb}")
    print(f"Using test NWB: {test_nwb}")
    print(f"Writing outputs to: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_spikes, train_bin_width_s, train_trial_info, train_time_index_s, train_windows = preprocess_mcmaze(
        train_nwb, cfg, use_cache=True, label="train NWB"
    )
    test_spikes, test_bin_width_s, test_trial_info, test_time_index_s, test_windows = preprocess_mcmaze(
        str(test_nwb), cfg, use_cache=False, label="test NWB"
    )
    if len(test_trial_info) != 574:
        print(f"Warning: expected 574 test trials, found {len(test_trial_info)}")
    if train_spikes.shape[0] != test_spikes.shape[0]:
        raise ValueError(
            f"Train/test channel mismatch: train={train_spikes.shape[0]}, test={test_spikes.shape[0]}"
        )

    trial_info_for_split = train_trial_info
    if getattr(cfg, "split", "dataset") == "random":
        trial_info_for_split = train_trial_info.drop(columns=["split"], errors="ignore")
    train_ds_base, val_ds_base = train_val_split(train_windows, trial_info_for_split, cfg.val_split, cfg.seed)
    train_indices = list(train_ds_base.indices)
    val_indices = list(val_ds_base.indices)
    if args.max_train_trials > 0:
        rng = np.random.default_rng(args.seed)
        train_indices = sorted(int(i) for i in rng.choice(train_indices, size=min(args.max_train_trials, len(train_indices)), replace=False))
    if args.max_val_trials > 0:
        rng = np.random.default_rng(args.seed + 1)
        val_indices = sorted(int(i) for i in rng.choice(val_indices, size=min(args.max_val_trials, len(val_indices)), replace=False))
    print(f"Decoder train/val trials: train={len(train_indices)} val={len(val_indices)}")

    print("Loading hand velocity targets from labelled train NWB...")
    hand_vel_raw, vel_meta = load_hand_velocity_resampled(train_nwb, train_time_index_s, args.velocity_scale)
    targets, valid, horizon_bins = future_velocity_windows(
        hand_vel_raw, train_trial_info, train_time_index_s, train_bin_width_s, cfg, args.horizon_ms
    )
    print(
        f"Target: hand_vel +{args.horizon_ms} ms ({horizon_bins} bins), "
        f"valid samples={valid.sum():,}/{valid.size:,}"
    )

    initial_embedder = build_embedder(cfg, state_dict, train_spikes.shape[0], init="pretrained")
    initial_embedder_state = clone_state_dict_cpu(initial_embedder)
    frozen_train_subset = Subset(TensorDataset(torch.from_numpy(train_windows)), train_indices)
    frozen_val_subset = Subset(TensorDataset(torch.from_numpy(train_windows)), val_indices)

    print(f"Computing frozen {args.feature_layer} features on {device}...")
    F_train = compute_features(initial_embedder, frozen_train_subset, args.embed_batch_size, device, args.feature_layer)
    F_val = compute_features(initial_embedder, frozen_val_subset, args.embed_batch_size, device, args.feature_layer)
    X_train, y_train = flatten_samples(F_train, targets, valid, train_indices)
    X_val, y_val = flatten_samples(F_val, targets, valid, val_indices)
    X_train_std, X_val_std, feature_mean, feature_std = standardize_train_val(X_train, X_val)
    target_mean = y_train.mean(axis=0, keepdims=True)
    target_std = np.maximum(y_train.std(axis=0, keepdims=True), 1e-6)
    decoder_in_dim = int(X_train.shape[1])
    print(f"Decoder samples: train={len(X_train):,} val={len(X_val):,} features=d{decoder_in_dim}")

    print("Training frozen-embedding MLP decoder...")
    pred_frozen_val, frozen_decoder, frozen_info = train_mlp_decoder(
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
        "feature_layer": args.feature_layer,
        "feature_dim": decoder_in_dim,
        **regression_metrics(y_val, pred_frozen_val),
        "best_val_mse_z": float(frozen_info["best_val_mse_z"]),
    }
    save_decoder_loss_curve(out_dir, frozen_info["history"], stem="frozen_mlp_loss_curve")

    train_ds = WindowVelocityDataset(train_windows, targets, valid, train_indices)
    val_ds = WindowVelocityDataset(train_windows, targets, valid, val_indices)
    train_loader = DataLoader(train_ds, batch_size=args.trial_batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.trial_batch_size, shuffle=False, drop_last=False, num_workers=0)

    print("Fine-tuning embedder + MLP decoder...")
    finetune_embedder = build_embedder(cfg, state_dict, train_spikes.shape[0], init="pretrained")
    finetune_embedder.load_state_dict(initial_embedder_state)
    finetune_decoder = VelocityMLP(
        decoder_in_dim,
        hidden_dim=args.mlp_hidden_dim,
        depth=args.mlp_depth,
        dropout=args.mlp_dropout,
    )
    finetune_decoder.load_state_dict({k: v.detach().cpu().clone() for k, v in frozen_decoder.state_dict().items()})
    finetune_embedder, finetune_decoder, ft_history, y_val_ft, pred_ft_val, finetune_info = train_finetuned_model(
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
        "feature_layer": args.feature_layer,
        "feature_dim": decoder_in_dim,
        **regression_metrics(y_val_ft, pred_ft_val),
        "best_val_mse_z": float(finetune_info["best_val_mse_z"]),
    }
    save_decoder_loss_curve(out_dir, ft_history, stem="finetuned_mlp_loss_curve")
    save_metrics(out_dir, [frozen_metrics, finetuned_metrics])
    plot_predictions(
        out_dir,
        y_val_ft,
        {"frozen_mlp": pred_frozen_val, "finetuned_mlp": pred_ft_val},
        max_points=args.max_plot_points,
        seed=args.seed,
    )

    pre_bins = int(round(int(getattr(cfg, "pre_ms", 100)) * 1e-3 / test_bin_width_s))
    time_ms = (np.arange(int(cfg.window_size)) - pre_bins + horizon_bins) * test_bin_width_s * 1000.0
    prediction_meta = {
        "model_path": np.asarray(str(model_path), dtype=object),
        "train_nwb": np.asarray(str(train_nwb), dtype=object),
        "test_nwb": np.asarray(str(test_nwb), dtype=object),
        "horizon_ms": np.asarray(args.horizon_ms, dtype=np.int64),
        "velocity_units": np.asarray(vel_meta.get("units", ""), dtype=object),
        "feature_layer": np.asarray(args.feature_layer, dtype=object),
    }

    print("Predicting test set with frozen decoder...")
    frozen_test_pred = predict_windows(
        initial_embedder,
        frozen_decoder,
        test_windows,
        feature_mean,
        feature_std,
        frozen_info["target_mean"],
        frozen_info["target_std"],
        args.feature_layer,
        args.embed_batch_size,
        device,
    )
    save_prediction_artifacts(out_dir, "frozen_mlp", frozen_test_pred, test_trial_info, time_ms, prediction_meta)

    print("Predicting test set with fine-tuned model...")
    finetuned_test_pred = predict_windows(
        finetune_embedder,
        finetune_decoder,
        test_windows,
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        args.feature_layer,
        args.embed_batch_size,
        device,
    )
    save_prediction_artifacts(out_dir, "finetuned_mlp", finetuned_test_pred, test_trial_info, time_ms, prediction_meta)

    torch.save(
        {
            "args": vars(args),
            "checkpoint_path": str(model_path),
            "checkpoint_epoch": ckpt.get("epoch"),
            "feature_layer": args.feature_layer,
            "feature_dim": decoder_in_dim,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "frozen_decoder_state_dict": frozen_decoder.cpu().state_dict(),
            "finetuned_embedder_state_dict": finetune_embedder.cpu().state_dict(),
            "finetuned_decoder_state_dict": finetune_decoder.cpu().state_dict(),
            "frozen_decoder_history": frozen_info["history"],
            "finetuned_history": ft_history,
            "velocity_meta": vel_meta,
            "config": cfg_to_dict(cfg),
            "validation_metrics": [frozen_metrics, finetuned_metrics],
        },
        out_dir / "test_velocity_models.pt",
    )
    print(f"Saved trained decoder/fine-tune bundle: {out_dir / 'test_velocity_models.pt'}")

    print()
    print("Validation metrics on labelled train-NWB validation split:")
    for row in (frozen_metrics, finetuned_metrics):
        print(
            f"  {row['model']:<14} R2 mean={row['r2_mean']:.4f} "
            f"RMSE mean={row['rmse_mean']:.4f} corr x/y={row['corr_x']:.3f}/{row['corr_y']:.3f}"
        )
    print(f"Test predictions exported for {len(test_trial_info)} trials.")


if __name__ == "__main__":
    main()
