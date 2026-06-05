import os
import csv
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from loss import loss_fn, non_reversibility_S, _batch_rms_normalize


def _make_loader(dataset, batch_size: int, shuffle: bool, drop_last: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )

# Unused learned plane-predictor adversary removed from the active training path
# while the current sweep focuses on BT, plane-aware BT, and block-CCA.


def train(model, train_ds, val_ds, cfg: Config, loss_function=loss_fn) -> dict:
    """Run the full training loop.

    Returns a history dict with keys:
      'train_loss' : list of mean train loss per epoch
      'val_loss'   : list of mean val loss per epoch
      'best_val_loss' : float
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

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [],
               "train_s": [], "val_s": [],
               "train_reg": [], "val_reg": []}
    log_path = os.path.join(cfg.out_dir, "log.csv")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss",
                                 "train_s", "val_s", "train_reg", "val_reg"])

    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        # --- train ---
        model.train()
        epoch_losses, epoch_s, epoch_reg = [], [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]", leave=False)
        for (batch,) in pbar:
            batch = batch.to(device)                    # (K, N, T)
            optimizer.zero_grad()
            F = model(batch)                            # (K, d, T)
            F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
            loss = loss_function(F, cfg=cfg, training=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                s = non_reversibility_S(_batch_rms_normalize(F)).item()
            l = loss.item()
            epoch_losses.append(l)
            epoch_s.append(s)
            epoch_reg.append(l + s)     # total = -S + reg  →  reg = total + S
            pbar.set_postfix(loss=f"{l:.4f}", S=f"{s:.4f}")

        scheduler.step(epoch)
        mean_train_loss = sum(epoch_losses) / len(epoch_losses)
        mean_train_s    = sum(epoch_s)      / len(epoch_s)
        mean_train_reg  = sum(epoch_reg)    / len(epoch_reg)

        # --- validate ---
        model.eval()
        val_losses, val_s, val_reg = [], [], []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                if batch.shape[0] < 2:
                    continue
                F = model(batch)
                F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
                loss = loss_function(F, cfg=cfg, training=False)
                s = non_reversibility_S(_batch_rms_normalize(F)).item()
                val_losses.append(loss.item())
                val_s.append(s)
                val_reg.append(loss.item() + s)

        mean_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
        mean_val_s    = sum(val_s)      / len(val_s)      if val_s      else float("nan")
        mean_val_reg  = sum(val_reg)    / len(val_reg)    if val_reg    else float("nan")

        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(mean_val_loss)
        history["train_s"].append(mean_train_s)
        history["val_s"].append(mean_val_s)
        history["train_reg"].append(mean_train_reg)
        history["val_reg"].append(mean_val_reg)

        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"train loss={mean_train_loss:.4f}  val loss={mean_val_loss:.4f}  "
            f"S={mean_val_s:.4f}  reg={mean_val_reg:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, mean_train_loss, mean_val_loss,
                                     mean_train_s, mean_val_s,
                                     mean_train_reg, mean_val_reg])

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(
                {"model_state_dict": model.state_dict(), "config": cfg, "epoch": epoch},
                os.path.join(cfg.ckpt_dir, "best.pt"),
            )

    history["best_val_loss"] = best_val_loss
    history["elapsed_s"] = time.time() - t0

    # --- loss curve ---
    ep = range(1, cfg.epochs + 1)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(ep, history["val_s"],   label="S  (non-rev, ↑)",   color="steelblue")
    ax.plot(ep, history["val_reg"], label="λ·reg  (penalty, ↓)", color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_title("Training dynamics")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    return history
