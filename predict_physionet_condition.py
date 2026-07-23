"""Predict PhysioNetMI condition labels from frozen full-trial embeddings.

This script loads a trained PhysioNetMI embedding checkpoint, recomputes frozen
embeddings for each EEG trial, and trains trial-level condition decoders:

  1. Logistic regression from flattened embedding trajectory ``d*T`` -> class
  2. MLP classifier from flattened embedding trajectory ``d*T`` -> class
  3. Temporal Conv1d classifier from embedding trajectory ``d x T`` -> class

Unlike the MC Maze velocity decoder, this is a trial-level classifier: each
sample is one full single-trial embedding trajectory.

If the checkpoint config defines SYNTH_HOLDOUT_SUBJECT_COUNT or
SYNTH_HOLDOUT_SUBJECT_IDS, those participants are excluded from decoder
train/validation and evaluated as a held-out subject test set.

Examples
--------
    source physionetmi_config.sh
    python predict_physionet_condition.py
    python predict_physionet_condition.py --run 2 --mlp-epochs 100
    python predict_physionet_condition.py --decoder-split random
    python predict_physionet_condition.py --run 2 --feature-layer hidden
    python predict_physionet_condition.py --run 2 --embedder-init random --feature-layer hidden
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
from model import MLP, infer_multiscale_symmetric_conv_layers
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


class EmbeddingTemporalConvClassifier(nn.Module):
    """Trial-level decoder: temporal Conv1d over frozen embedding trajectories."""

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hidden_dim: int,
        depth: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        padding = kernel_size // 2
        layers = []
        dim = in_channels
        for _ in range(depth):
            layers += [nn.Conv1d(dim, hidden_dim, kernel_size, padding=padding), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.temporal = nn.Sequential(*layers)
        self.classifier = nn.Linear(dim, n_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.temporal(x)
        x = x.mean(dim=-1)
        return self.classifier(x)


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


def build_model_from_checkpoint(cfg: Config, state_dict, in_channels: int, init: str = "pretrained"):
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
    if init == "pretrained":
        model.load_state_dict(state_dict)
    elif init != "random":
        raise ValueError(f"embedder init must be 'pretrained' or 'random', got {init!r}")
    return model


def compute_embeddings(model, dataset, batch_size: int, device: torch.device) -> np.ndarray:
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    with torch.no_grad():
        for (batch,) in loader:
            chunks.append(model(batch.to(device)).cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32)


def compute_hidden_features(model, dataset, batch_size: int, device: torch.device) -> np.ndarray:
    """Compute penultimate MLP features by removing the final projection layer."""
    model = model.to(device).eval()
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
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
            chunks.append(H.cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32)


def compute_features(model, dataset, batch_size: int, device: torch.device, feature_layer: str) -> np.ndarray:
    if feature_layer == "output":
        return compute_embeddings(model, dataset, batch_size, device)
    if feature_layer == "hidden":
        return compute_hidden_features(model, dataset, batch_size, device)
    raise ValueError(f"unknown feature_layer {feature_layer!r}")


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


def transform_trajectory_features(
    F: np.ndarray,
    emb_mean: np.ndarray,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
) -> np.ndarray:
    """Apply train-derived trajectory centering/standardization to another split."""
    X = (F - emb_mean).reshape(F.shape[0], -1)
    if feature_mean is None or feature_std is None:
        return X
    return (X - feature_mean) / feature_std


def sequence_features(
    F_train: np.ndarray,
    F_val: np.ndarray,
    standardize: bool,
    emb_mean: np.ndarray | None = None,
):
    """Return centered (K, d, T) features for temporal-conv decoding."""
    if emb_mean is None:
        emb_mean = F_train.mean(axis=(0, 2), keepdims=True)
    F_train = F_train - emb_mean
    F_val = F_val - emb_mean
    if not standardize:
        return F_train, F_val, emb_mean, None, None

    feature_mean = F_train.mean(axis=(0, 2), keepdims=True)
    feature_std = np.maximum(F_train.std(axis=(0, 2), keepdims=True), 1e-6)
    return (
        (F_train - feature_mean) / feature_std,
        (F_val - feature_mean) / feature_std,
        emb_mean,
        feature_mean,
        feature_std,
    )


def transform_sequence_features(
    F: np.ndarray,
    emb_mean: np.ndarray,
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
) -> np.ndarray:
    """Apply train-derived sequence centering/standardization to another split."""
    F = F - emb_mean
    if feature_mean is None or feature_std is None:
        return F
    return (F - feature_mean) / feature_std


def classification_metrics(y_true, y_pred, prefix: str = ""):
    return {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}macro_f1": float(f1_score(y_true, y_pred, average="macro")),
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


def predict_mlp_classifier(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32)).to(device)
        return model(X_t).argmax(dim=1).cpu().numpy()


def train_temporal_conv_classifier(
    F_train,
    y_train,
    F_val,
    y_val,
    n_classes: int,
    hidden_dim: int,
    depth: int,
    kernel_size: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    early_stop_epoch=None,
    early_stop_min_acc=None,
):
    torch.manual_seed(seed)
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(F_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    F_val_t = torch.from_numpy(F_val.astype(np.float32)).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.int64)).to(device)

    model = EmbeddingTemporalConvClassifier(
        in_channels=F_train.shape[1],
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        depth=depth,
        kernel_size=kernel_size,
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
            val_logits = model(F_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()
            val_pred = val_logits.argmax(dim=1)
            val_acc = float((val_pred == y_val_t).float().mean().item())
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc = val_acc
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"TemporalConv epoch {epoch:3d}/{epochs}  "
            f"train ce={np.mean(losses):.4f} acc={correct / max(total, 1):.1%}  "
            f"val ce={val_loss:.4f} acc={val_acc:.1%}"
        )
        if (
            early_stop_epoch is not None
            and early_stop_min_acc is not None
            and epoch >= int(early_stop_epoch)
            and best_acc < float(early_stop_min_acc)
        ):
            print(
                f"TemporalConv early stop at epoch {epoch}: "
                f"best val acc={best_acc:.1%} below {float(early_stop_min_acc):.1%}"
            )
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(F_val_t).argmax(dim=1).cpu().numpy()
    return pred, model, {"best_val_acc": best_acc, "best_val_ce": best_loss, "epochs_ran": epoch}


def predict_temporal_conv_classifier(model: nn.Module, F: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        F_t = torch.from_numpy(F.astype(np.float32)).to(device)
        return model(F_t).argmax(dim=1).cpu().numpy()


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


def plot_confusions(out_dir, y_true, predictions, labels, label_names, filename="confusion_matrices.png"):
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
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="Integer rank, explicit run dir, or omit for newest.")
    parser.add_argument("--data", default=None, help="Override synthetic data .npy path from checkpoint config.")
    parser.add_argument("--decoder-split", choices=["checkpoint", "random", "subject_random"], default="checkpoint",
                        help="Split for the decoder. 'checkpoint' uses cfg.synth_split.")
    parser.add_argument("--embedder-init", choices=["pretrained", "random"], default="pretrained",
                        help="Initialize the frozen embedder from checkpoint weights or random weights with the same architecture.")
    parser.add_argument("--decoder-type", choices=["all", "linear", "mlp", "temporal_conv"], default="all",
                        help="Which downstream condition decoder(s) to train.")
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--feature-layer", choices=["output", "hidden"], default="output",
                        help="Use normal embedder output or remove the final linear layer and decode from hidden features.")
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--mlp-dropout", type=float, default=0.2)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--mlp-batch-size", type=int, default=128)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--conv-hidden-dim", type=int, default=128)
    parser.add_argument("--conv-depth", type=int, default=2)
    parser.add_argument("--conv-kernel-size", type=int, default=31)
    parser.add_argument("--conv-dropout", type=float, default=None,
                        help="Temporal-conv dropout. Defaults to --mlp-dropout when omitted.")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.mlp_depth < 1:
        parser.error("--mlp-depth must be at least 1")
    if args.mlp_hidden_dim < 1:
        parser.error("--mlp-hidden-dim must be positive")
    if args.mlp_epochs < 1:
        parser.error("--mlp-epochs must be at least 1")
    if args.mlp_batch_size < 1:
        parser.error("--mlp-batch-size must be positive")
    if args.conv_depth < 1:
        parser.error("--conv-depth must be at least 1")
    if args.conv_hidden_dim < 1:
        parser.error("--conv-hidden-dim must be positive")
    if args.conv_kernel_size < 1 or args.conv_kernel_size % 2 == 0:
        parser.error("--conv-kernel-size must be a positive odd integer")
    conv_dropout = args.mlp_dropout if args.conv_dropout is None else args.conv_dropout

    run_dir = resolve_run_dir(args.run)
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path!r}")
    print(f"Using run: {run_dir}")
    print(f"Embedder initialization: {args.embedder_init}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]
    if not hasattr(cfg, "synth_holdout_subject_count"):
        cfg.synth_holdout_subject_count = 0
    if not hasattr(cfg, "synth_holdout_subject_ids"):
        cfg.synth_holdout_subject_ids = ""
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
    train_ds, val_ds, holdout_ds, trainval_subjects, holdout_subjects = train_val_split_synth(
        windows,
        cfg.val_split,
        cfg.seed,
        split,
        subjects=subjects,
        subject_count=getattr(cfg, "synth_subject_count", 0),
        subject_ids=getattr(cfg, "synth_subject_ids", ""),
        holdout_subject_count=getattr(cfg, "synth_holdout_subject_count", 0),
        holdout_subject_ids=getattr(cfg, "synth_holdout_subject_ids", ""),
        return_holdout=True,
    )
    train_idx = _dataset_source_indices(train_ds)
    val_idx = _dataset_source_indices(val_ds)
    holdout_idx = _dataset_source_indices(holdout_ds) if holdout_ds is not None else None
    y_train_raw = labels[train_idx]
    y_val_raw = labels[val_idx]
    y_test_raw = labels[holdout_idx] if holdout_idx is not None else None
    class_arrays = [y_train_raw, y_val_raw]
    if y_test_raw is not None:
        class_arrays.append(y_test_raw)
    classes = np.array(sorted(np.unique(np.concatenate(class_arrays))), dtype=np.int64)
    class_to_idx = {label: i for i, label in enumerate(classes)}
    y_train = np.array([class_to_idx[int(label)] for label in y_train_raw], dtype=np.int64)
    y_val = np.array([class_to_idx[int(label)] for label in y_val_raw], dtype=np.int64)
    y_test = (
        np.array([class_to_idx[int(label)] for label in y_test_raw], dtype=np.int64)
        if y_test_raw is not None
        else None
    )
    label_names = [CONDITION_NAMES.get(int(label), f"condition {int(label)}") for label in classes]

    if len(set(train_idx.tolist()) & set(val_idx.tolist())):
        print("Warning: train and validation source indices overlap; decoder accuracy is not a held-out estimate.")
    if holdout_idx is not None:
        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())
        holdout_set = set(holdout_idx.tolist())
        if train_set & holdout_set or val_set & holdout_set:
            raise RuntimeError("Held-out subject indices overlap decoder train/validation indices.")
    print(
        f"Decoder split: {split}  train={len(train_ds)} val={len(val_ds)} "
        f"test={len(holdout_ds) if holdout_ds is not None else 0}"
    )
    if trainval_subjects is not None:
        print(f"Decoder train/val subjects: {np.asarray(trainval_subjects).tolist()}")
    if holdout_subjects is not None and len(holdout_subjects):
        print(f"Decoder held-out subjects: {np.asarray(holdout_subjects).tolist()}")
    print(f"Classes: {dict(zip(label_names, np.bincount(y_train, minlength=len(classes)).tolist()))} train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = build_model_from_checkpoint(
        cfg,
        ckpt["model_state_dict"],
        windows.shape[1],
        init=args.embedder_init,
    )
    print(f"Computing {args.embedder_init}-init frozen {args.feature_layer} features on {device}...")
    F_train = compute_features(embedder, train_ds, args.embed_batch_size, device, args.feature_layer)
    F_val = compute_features(embedder, val_ds, args.embed_batch_size, device, args.feature_layer)
    F_test = (
        compute_features(embedder, holdout_ds, args.embed_batch_size, device, args.feature_layer)
        if holdout_ds is not None
        else None
    )
    X_train, X_val, emb_mean, feature_mean, feature_std = trajectory_features(
        F_train,
        F_val,
        standardize=not args.no_standardize,
    )
    X_test = (
        transform_trajectory_features(F_test, emb_mean, feature_mean, feature_std)
        if F_test is not None
        else None
    )
    F_train_seq, F_val_seq, _, seq_feature_mean, seq_feature_std = sequence_features(
        F_train,
        F_val,
        standardize=not args.no_standardize,
        emb_mean=emb_mean,
    )
    F_test_seq = (
        transform_sequence_features(F_test, emb_mean, seq_feature_mean, seq_feature_std)
        if F_test is not None
        else None
    )
    feature_context = {
        "embedder_init": args.embedder_init,
        "feature_layer": args.feature_layer,
        "feature_dim": int(F_train.shape[1]),
        "flat_feature_dim": int(X_train.shape[1]),
    }
    print(f"Decoder features: train={X_train.shape} val={X_val.shape}  raw features={F_train.shape[1:]} per trial")
    print(f"Temporal-conv features: train={F_train_seq.shape} val={F_val_seq.shape}")
    if X_test is not None and F_test_seq is not None:
        print(f"Held-out test features: flat={X_test.shape} temporal={F_test_seq.shape}")

    out_name = "condition_prediction"
    if args.embedder_init == "random":
        out_name += "_random_init"
    if args.feature_layer == "hidden":
        out_name += "_hidden"
    out_dir = os.path.join(run_dir, "outputs", out_name)
    os.makedirs(out_dir, exist_ok=True)
    if F_test is not None:
        heldout_path = os.path.join(out_dir, "heldout_embeddings.npz")
        np.savez_compressed(
            heldout_path,
            embeddings=F_test,
            labels=y_test_raw,
            subjects=subjects[holdout_idx] if subjects is not None else None,
            source_indices=holdout_idx,
            heldout_subjects=np.asarray(holdout_subjects),
            classes=classes,
            label_names=np.asarray(label_names),
            embedder_init=np.asarray(args.embedder_init),
            feature_layer=np.asarray(args.feature_layer),
            feature_dim=np.asarray(int(F_test.shape[1])),
        )
        print(f"Saved held-out embeddings: {heldout_path}")

    run_linear = args.decoder_type in {"all", "linear"}
    run_mlp = args.decoder_type in {"all", "mlp"}
    run_temporal_conv = args.decoder_type in {"all", "temporal_conv"}
    rows = []
    predictions = {}
    test_predictions = {}
    mlp = None
    temporal_conv = None

    if run_linear:
        print("Training logistic regression baseline...")
        pred_lr, lr_model = train_logistic_regression(X_train, y_train, X_val, seed=args.seed)
        row = {
            "model": "logistic_regression",
            **feature_context,
            **classification_metrics(y_val, pred_lr),
        }
        if X_test is not None and y_test is not None:
            pred_lr_test = lr_model.predict(X_test)
            row.update(classification_metrics(y_test, pred_lr_test, prefix="test_"))
            test_predictions["logistic"] = pred_lr_test
        rows.append(row)
        predictions["logistic"] = pred_lr

    if run_mlp:
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
        row = {
            "model": "mlp",
            **feature_context,
            **classification_metrics(y_val, pred_mlp),
            "best_val_acc": float(mlp_info["best_val_acc"]),
            "best_val_ce": float(mlp_info["best_val_ce"]),
        }
        if X_test is not None and y_test is not None:
            pred_mlp_test = predict_mlp_classifier(mlp, X_test, device)
            row.update(classification_metrics(y_test, pred_mlp_test, prefix="test_"))
            test_predictions["mlp"] = pred_mlp_test
        rows.append(row)
        predictions["mlp"] = pred_mlp

    if run_temporal_conv:
        print("Training temporal-conv condition decoder...")
        pred_conv, temporal_conv, conv_info = train_temporal_conv_classifier(
            F_train_seq,
            y_train,
            F_val_seq,
            y_val,
            n_classes=len(classes),
            hidden_dim=args.conv_hidden_dim,
            depth=args.conv_depth,
            kernel_size=args.conv_kernel_size,
            dropout=conv_dropout,
            epochs=args.mlp_epochs,
            batch_size=args.mlp_batch_size,
            lr=args.mlp_lr,
            weight_decay=args.mlp_weight_decay,
            seed=args.seed,
            device=device,
        )
        row = {
            "model": "temporal_conv",
            **feature_context,
            **classification_metrics(y_val, pred_conv),
            "best_val_acc": float(conv_info["best_val_acc"]),
            "best_val_ce": float(conv_info["best_val_ce"]),
        }
        if F_test_seq is not None and y_test is not None:
            pred_conv_test = predict_temporal_conv_classifier(temporal_conv, F_test_seq, device)
            row.update(classification_metrics(y_test, pred_conv_test, prefix="test_"))
            test_predictions["temporal_conv"] = pred_conv_test
        rows.append(row)
        predictions["temporal_conv"] = pred_conv

    save_metrics(out_dir, rows)
    plot_confusions(
        out_dir,
        y_val,
        predictions,
        labels=np.arange(len(classes)),
        label_names=label_names,
    )
    if y_test is not None and test_predictions:
        plot_confusions(
            out_dir,
            y_test,
            test_predictions,
            labels=np.arange(len(classes)),
            label_names=label_names,
            filename="confusion_matrices_holdout.png",
        )

    torch.save(
        {
            "mlp_state_dict": mlp.state_dict() if mlp is not None else None,
            "temporal_conv_state_dict": temporal_conv.state_dict() if temporal_conv is not None else None,
            "args": vars(args),
            "resolved_conv_dropout": conv_dropout,
            "checkpoint_run_dir": run_dir,
            "checkpoint_epoch": ckpt.get("epoch"),
            "embedder_init": args.embedder_init,
            "classes": classes,
            "label_names": label_names,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "holdout_indices": holdout_idx,
            "trainval_subjects": np.asarray(trainval_subjects) if trainval_subjects is not None else None,
            "holdout_subjects": np.asarray(holdout_subjects) if holdout_subjects is not None else None,
            "embedding_mean": emb_mean,
            "feature_layer": args.feature_layer,
            "feature_dim": int(F_train.shape[1]),
            "flat_feature_dim": int(X_train.shape[1]),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "seq_feature_mean": seq_feature_mean,
            "seq_feature_std": seq_feature_std,
            "config": asdict(cfg),
            "metrics": rows,
        },
        os.path.join(out_dir, "mlp_condition_decoder.pt"),
    )
    print(f"Saved decoder checkpoint: {os.path.join(out_dir, 'mlp_condition_decoder.pt')}")

    print()
    print("Validation metrics:")
    for row in rows:
        msg = (
            f"  {row['model']:<20} "
            f"acc={row['accuracy']:.1%}  "
            f"balanced_acc={row['balanced_accuracy']:.1%}  "
            f"macro_f1={row['macro_f1']:.3f}"
        )
        if "test_accuracy" in row:
            msg += (
                f"  |  heldout_test_acc={row['test_accuracy']:.1%}  "
                f"test_balanced_acc={row['test_balanced_accuracy']:.1%}  "
                f"test_macro_f1={row['test_macro_f1']:.3f}"
            )
        print(msg)


if __name__ == "__main__":
    main()
