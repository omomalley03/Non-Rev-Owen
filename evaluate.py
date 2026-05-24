"""
Linear probe: predict (trial_type, trial_version) from frozen embeddings.

Tests whether the learned embedding retains task-relevant condition
information.  The probe is a multinomial logistic regression on the
time-averaged embedding vector (d features per trial).

Usage
-----
    python evaluate.py                   # most recent run
    python evaluate.py --run 2           # 2nd most recent run
    python evaluate.py --run runs/foo    # explicit path
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from config import Config
from data import load_mcmaze, gaussian_smooth, make_windows, train_val_split
from model import MLP
from visualize import _resolve_run_dir


def _condition_labels(trial_info, indices):
    """Extract (trial_type, trial_version) for each trial index."""
    return [
        f"{int(trial_info.iloc[i]['trial_type'])}_{int(trial_info.iloc[i]['trial_version'])}"
        for i in indices
    ]


def run_linear_probe(model, train_ds, val_ds, trial_info, cfg: Config, run_dir: str) -> dict:
    model = model.cpu().eval()

    with torch.no_grad():
        train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
        (train_tensor,) = next(iter(train_loader))
        F_train = model(train_tensor).numpy()
        F_train = F_train - F_train.mean(axis=0, keepdims=True)  # zero-mean per dim across batch and time

        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
        (val_tensor,) = next(iter(val_loader))
        F_val = model(val_tensor).numpy()
        F_val = F_val - F_val.mean(axis=0, keepdims=True)  # zero-mean per dim across batch and time
    X_train = F_train.mean(axis=2)
    X_val = F_val.mean(axis=2)

    train_labels = _condition_labels(trial_info, list(train_ds.indices))
    val_labels = _condition_labels(trial_info, list(val_ds.indices))

    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_val = le.transform(val_labels)

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=cfg.seed,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    y_pred = clf.predict(X_val)
    n_classes = len(le.classes_)

    print()
    print("=" * 50)
    print("Linear probe: condition classification")
    print(f"  Features       : time-averaged embedding (d={cfg.d})")
    print(f"  Classes        : {n_classes}  (trial_type, trial_version)")
    print(f"  Train accuracy : {train_acc:.1%}  ({len(y_train)} trials)")
    print(f"  Val accuracy   : {val_acc:.1%}  ({len(y_val)} trials)")
    print(f"  Chance level   : {1.0 / n_classes:.1%}")
    print("=" * 50)

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "n_classes": n_classes,
        "y_val": y_val,
        "y_pred": y_pred,
        "label_encoder": le,
    }


def plot_confusion_matrix(y_val, y_pred, label_encoder, val_acc, out_path):
    cm = confusion_matrix(y_val, y_pred, labels=np.arange(len(label_encoder.classes_)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted condition", fontsize=9)
    ax.set_ylabel("True condition", fontsize=9)
    ax.set_title(
        f"Condition classification  (val acc = {val_acc:.1%})\n"
        f"{len(label_encoder.classes_)} classes, chance = {1.0 / len(label_encoder.classes_):.1%}",
        fontsize=10,
    )
    ax.tick_params(labelbottom=False, labelleft=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None,
                        help="Integer (1=most recent) or explicit path.")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run)
    print(f"Using run: {os.path.basename(run_dir)}")

    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: Config = ckpt["config"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    print("Loading data…")
    spikes_raw, bin_width_s, trial_info, time_index_s, _ = load_mcmaze(
        cfg.nwb_path, cfg.bin_ms
    )
    N = spikes_raw.shape[0]
    sigma_samples = round((cfg.sigma_ms * 1e-3) / bin_width_s)
    X_smooth = gaussian_smooth(spikes_raw, sigma_samples)
    softnorm = getattr(cfg, "softnorm_method", "none")
    if softnorm and softnorm != "none":
        from data import soft_normalize
        X_smooth = soft_normalize(X_smooth, method=softnorm)
    windows = make_windows(
        X_smooth, trial_info, time_index_s, bin_width_s,
        strategy=cfg.window_strategy, window_size=cfg.window_size,
        align_field=getattr(cfg, "align_field", "move_onset_time"),
        pre_ms=getattr(cfg, "pre_ms", 100),
    )
    windows = windows - windows.mean(axis=0, keepdims=True)

    train_ds, val_ds = train_val_split(windows, trial_info, cfg.val_split, cfg.seed)

    dropout = getattr(cfg, "dropout", 0.0)
    model = MLP(in_channels=N, d=cfg.d, hidden_dim=cfg.hidden_dim,
                depth=cfg.depth, dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"])

    out_dir = os.path.join(run_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    results = run_linear_probe(model, train_ds, val_ds, trial_info, cfg, run_dir)
    plot_confusion_matrix(
        results["y_val"], results["y_pred"],
        results["label_encoder"], results["val_acc"],
        out_path=os.path.join(out_dir, "11_condition_classifier.png"),
    )


if __name__ == "__main__":
    main()
