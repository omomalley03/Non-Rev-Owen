import os
import csv
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from loss import loss_fn, S_ratio


def _make_loader(dataset, batch_size: int, shuffle: bool, drop_last: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )


def train(model, train_ds, val_ds, cfg: Config) -> dict:
    """Run the full training loop.

    Returns a history dict with keys:
      'train_S_ratio' : list of mean train S_ratio per epoch
      'val_S_ratio'   : list of mean val S_ratio per epoch
      'best_val_S_ratio' : float
      'elapsed_s'     : total wall-clock seconds
    """
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = _make_loader(train_ds, cfg.batch_size, shuffle=True,  drop_last=True)
    val_loader   = _make_loader(val_ds,   cfg.batch_size, shuffle=False, drop_last=False)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult
    )

    best_val_ratio = -float("inf")
    history = {"train_S_ratio": [], "val_S_ratio": []}
    log_path = os.path.join(cfg.out_dir, "log.csv")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_S_ratio", "val_S_ratio"])

    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        # --- train ---
        model.train()
        epoch_ratios = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]", leave=False)
        for (batch,) in pbar:
            batch = batch.to(device)                    # (K, N, T)
            optimizer.zero_grad()
            F = model(batch)                            # (K, d, T)
            loss = loss_fn(F, cfg.lambda_bt)
            loss.backward()
            optimizer.step()
            # S_ratio reuses F (already computed); cheap relative to the backward pass
            with torch.no_grad():
                r = S_ratio(F).item()
            epoch_ratios.append(r)
            pbar.set_postfix(S_ratio=f"{r:.4f}")

        scheduler.step(epoch)
        mean_train_ratio = sum(epoch_ratios) / len(epoch_ratios)

        # --- validate ---
        model.eval()
        val_ratios = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                if batch.shape[0] < 2:          # S_ratio needs at least 2 windows
                    continue
                F = model(batch)
                val_ratios.append(S_ratio(F).item())

        mean_val_ratio = sum(val_ratios) / len(val_ratios) if val_ratios else float("nan")

        history["train_S_ratio"].append(mean_train_ratio)
        history["val_S_ratio"].append(mean_val_ratio)

        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"train S_ratio={mean_train_ratio:.4f}  val S_ratio={mean_val_ratio:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, mean_train_ratio, mean_val_ratio])

        if mean_val_ratio > best_val_ratio:
            best_val_ratio = mean_val_ratio
            torch.save(
                {"model_state_dict": model.state_dict(), "config": cfg, "epoch": epoch},
                os.path.join(cfg.ckpt_dir, "best.pt"),
            )

    history["best_val_S_ratio"] = best_val_ratio
    history["elapsed_s"] = time.time() - t0

    # --- loss curve ---
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, cfg.epochs + 1)
    ax.plot(epochs, history["train_S_ratio"], label="train S_ratio")
    ax.plot(epochs, history["val_S_ratio"],   label="val S_ratio")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("S_ratio  ∈ [0, 1]")
    ax.set_title("MC_Maze MLP — non-reversibility pretraining")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    return history
