"""Predict PhysioNetMI condition labels from frozen full-trial embeddings.

This script loads a trained PhysioNetMI embedding checkpoint, recomputes frozen
embeddings for each EEG trial, and trains trial-level condition decoders:

  1. Logistic regression from flattened embedding trajectory ``d*T`` -> class
  2. MLP classifier from flattened embedding trajectory ``d*T`` -> class

Unlike the MC Maze velocity decoder, this is a trial-level classifier: each
sample is one full single-trial embedding trajectory.

Examples
--------
    source physionetmi_config.sh
    python predict_physionet_condition.py
    python predict_physionet_condition.py --run 2 --mlp-epochs 100
    python predict_physionet_condition.py --decoder-split random
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from config import Config
from model import MLP
from paths import RUNS_BASE, SYNTH_RUNS_DIR
from synth_data import load_synthetic_labels, load_synthetic_subjects, load_synthetic_windows
from visualize_synth import _dataset_source_indices, train_val_split_synth


CONDITION_NAMES = {
    0: "imagined left fist",
    1: "imagined right fist",
    2: "imagined both fists",
    3: "imagined both feet",
}


class TrajectoryConditionMLP(nn.Module):
    """Trial-level decoder: flattened embedding trajectory -> condition logits."""

    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        layers = []
        dim = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, n_classes))
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
            SYNTH_RUNS_DIR,
            os.path.join(RUNS_BASE, "physionetmi", "synth_runs"),
            os.path.join(repo_root, "physionetmi", "synth_runs"),
            os.path.join(repo_root, "synth_runs"),
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
            raise FileNotFoundError("No completed synthetic runs with checkpoints/best.pt were found.")
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
    )
    model.load_state_dict(state_dict)
    return model


def compute_embeddings(model, dataset, batch_size: int, device: torch.device) -> np.ndarray:
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    with torch.no_grad():
        for (batch,) in loader:
            chunks.append(model(batch.to(device)).cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32)


def trajectory_features(F_train: np.ndarray, F_val: np.ndarray, standardize: bool):
    """Flatten (K, d, T) embeddings to trial-level d*T features."""
    # Use the training embedding mean for the same batch/time centering convention
    # as the main embedding objective, without leaking validation statistics.
    emb_mean = F_train.mean(axis=(0, 2), keepdims=True)
    F_train = F_train - emb_mean
    F_val = F_val - emb_mean

    X_train = F_train.reshape(F_train.shape[0], -1)
    X_val = F_val.reshape(F_val.shape[0], -1)
    if not standardize:
        return X_train, X_val, emb_mean, None, None

    feature_mean = X_train.mean(axis=0, keepdims=True)
    feature_std = np.maximum(X_train.std(axis=0, keepdims=True), 1e-6)
    return (X_train - feature_mean) / feature_std, (X_val - feature_mean) / feature_std, emb_mean, feature_mean, feature_std


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def train_logistic_regression(X_train, y_train, X_val, seed: int):
    clf = LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs")
    clf.fit(X_train, y_train)
    return clf.predict(X_val), clf


def train_mlp_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    n_classes: int,
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
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.int64)).to(device)

    model = TrajectoryConditionMLP(
        in_dim=X_train.shape[1],
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        correct = 0
        total = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += int(yb.numel())

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()
            val_pred = val_logits.argmax(dim=1)
            val_acc = float((val_pred == y_val_t).float().mean().item())
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc = val_acc
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"MLP epoch {epoch:3d}/{epochs}  "
            f"train ce={np.mean(losses):.4f} acc={correct / max(total, 1):.1%}  "
            f"val ce={val_loss:.4f} acc={val_acc:.1%}"
        )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(X_val_t).argmax(dim=1).cpu().numpy()
    return pred, model, {"best_val_acc": best_acc, "best_val_ce": best_loss}


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


def plot_confusions(out_dir, y_true, predictions, labels, label_names):
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4.5), squeeze=False)
    for ax, (name, y_pred) in zip(axes.ravel(), predictions.items()):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{name} confusion")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(label_names, rotation=35, ha="right", fontsize=7)
        ax.set_yticklabels(label_names, fontsize=7)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(out_dir, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="Integer rank, explicit run dir, or omit for newest.")
    parser.add_argument("--data", default=None, help="Override synthetic data .npy path from checkpoint config.")
    parser.add_argument("--decoder-split", choices=["checkpoint", "random", "subject_random"], default="checkpoint",
                        help="Split for the decoder. 'checkpoint' uses cfg.synth_split.")
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--mlp-dropout", type=float, default=0.2)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--mlp-batch-size", type=int, default=128)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-standardize", action="store_true")
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

    data_path = args.data or getattr(cfg, "synth_data_path", "")
    print(f"Loading data from {data_path}...")
    windows = load_synthetic_windows(cfg, data_path=data_path)
    labels = load_synthetic_labels(cfg, data_path=data_path)
    subjects = load_synthetic_subjects(cfg)
    if labels is None:
        raise ValueError("No labels found. Set SYNTH_LABELS_PATH or use a *_labels.npy cache next to the data file.")
    if len(labels) != len(windows):
        raise ValueError(f"labels length ({len(labels)}) must match windows ({len(windows)})")
    if subjects is not None and len(subjects) != len(windows):
        raise ValueError(f"subjects length ({len(subjects)}) must match windows ({len(windows)})")

    split = getattr(cfg, "synth_split", "random") if args.decoder_split == "checkpoint" else args.decoder_split
    train_ds, val_ds = train_val_split_synth(
        windows,
        cfg.val_split,
        cfg.seed,
        split,
        subjects=subjects,
        subject_count=getattr(cfg, "synth_subject_count", 0),
        subject_ids=getattr(cfg, "synth_subject_ids", ""),
    )
    train_idx = _dataset_source_indices(train_ds)
    val_idx = _dataset_source_indices(val_ds)
    y_train_raw = labels[train_idx]
    y_val_raw = labels[val_idx]
    classes = np.array(sorted(np.unique(np.concatenate([y_train_raw, y_val_raw]))), dtype=np.int64)
    class_to_idx = {label: i for i, label in enumerate(classes)}
    y_train = np.array([class_to_idx[int(label)] for label in y_train_raw], dtype=np.int64)
    y_val = np.array([class_to_idx[int(label)] for label in y_val_raw], dtype=np.int64)
    label_names = [CONDITION_NAMES.get(int(label), f"condition {int(label)}") for label in classes]

    if len(set(train_idx.tolist()) & set(val_idx.tolist())):
        print("Warning: train and validation source indices overlap; decoder accuracy is not a held-out estimate.")
    print(f"Decoder split: {split}  train={len(train_ds)} val={len(val_ds)}")
    print(f"Classes: {dict(zip(label_names, np.bincount(y_train, minlength=len(classes)).tolist()))} train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = build_model_from_checkpoint(cfg, ckpt["model_state_dict"], windows.shape[1])
    print(f"Computing frozen embeddings on {device}...")
    F_train = compute_embeddings(embedder, train_ds, args.embed_batch_size, device)
    F_val = compute_embeddings(embedder, val_ds, args.embed_batch_size, device)
    X_train, X_val, emb_mean, feature_mean, feature_std = trajectory_features(
        F_train,
        F_val,
        standardize=not args.no_standardize,
    )
    print(f"Decoder features: train={X_train.shape} val={X_val.shape}  raw embedding={F_train.shape[1:]} per trial")

    out_dir = os.path.join(run_dir, "outputs", "condition_prediction")
    os.makedirs(out_dir, exist_ok=True)

    print("Training logistic regression baseline...")
    pred_lr, lr_model = train_logistic_regression(X_train, y_train, X_val, seed=args.seed)
    lr_metrics = {"model": "logistic_regression", **classification_metrics(y_val, pred_lr)}

    print("Training MLP condition decoder...")
    pred_mlp, mlp, mlp_info = train_mlp_classifier(
        X_train,
        y_train,
        X_val,
        y_val,
        n_classes=len(classes),
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
        **classification_metrics(y_val, pred_mlp),
        "best_val_acc": float(mlp_info["best_val_acc"]),
        "best_val_ce": float(mlp_info["best_val_ce"]),
    }

    rows = [lr_metrics, mlp_metrics]
    save_metrics(out_dir, rows)
    plot_confusions(
        out_dir,
        y_val,
        {"logistic": pred_lr, "mlp": pred_mlp},
        labels=np.arange(len(classes)),
        label_names=label_names,
    )

    torch.save(
        {
            "mlp_state_dict": mlp.state_dict(),
            "args": vars(args),
            "checkpoint_run_dir": run_dir,
            "checkpoint_epoch": ckpt.get("epoch"),
            "classes": classes,
            "label_names": label_names,
            "embedding_mean": emb_mean,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "config": asdict(cfg),
            "metrics": rows,
        },
        os.path.join(out_dir, "mlp_condition_decoder.pt"),
    )
    print(f"Saved MLP decoder: {os.path.join(out_dir, 'mlp_condition_decoder.pt')}")

    print()
    print("Validation metrics:")
    for row in rows:
        print(
            f"  {row['model']:<20} "
            f"acc={row['accuracy']:.1%}  "
            f"balanced_acc={row['balanced_accuracy']:.1%}  "
            f"macro_f1={row['macro_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
