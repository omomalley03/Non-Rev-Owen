"""
Run a regularizer/objective sweep and summarize comparable validation metrics.

This script trains full jobs for combinations of:
  - S objective: "sum" vs "softmin"
  - redundancy regularizer: standard BT, plane-aware BT, block CCA

Raw losses are logged, but ranking should use objective-independent metrics
because each configuration optimizes a different scalar loss.

Usage
-----
    python run_regularizer_sweep.py
    python run_regularizer_sweep.py --epochs 80 --max-runs 30
    python run_regularizer_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from loss import (
    S_ratio,
    _batch_rms_normalize,
    barlow_twins_reg,
    block_cca_reg,
    loss_fn,
    non_reversibility_S,
    non_reversibility_S_per_plane,
    plane_barlow_twins_reg,
    plane_s_balance_reg,
)
from model import MLP
from train import train


SUMMARY_COLUMNS = [
    "rank",
    "experiment",
    "run_dir",
    "best_epoch",
    "best_train_loss",
    "best_val_loss",
    "val_loss_recomputed",
    "S_ratio",
    "S_total",
    "S_mean",
    "S_min",
    "S_max",
    "S_cv",
    "S_balance_raw",
    "bt_raw",
    "plane_bt_raw",
    "block_cca_raw",
    "pearson_offdiag_mean_abs",
    "pearson_offdiag_max_abs",
    "d",
    "epochs",
    "s_objective",
    "s_softmin_tau",
    "lambda_bt",
    "lambda_plane_bt",
    "lambda_block_cca",
    "lambda_s_balance",
    "dropout",
    "seed",
    "elapsed_s",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_experiment_grid(base_cfg: Config) -> list[tuple[str, Config]]:
    """Build a compact grid capped by --max-runs in main()."""
    experiments: list[tuple[str, Config]] = []

    objectives = [
        # ("sum", None),
        ("softmin", 0.1),
        ("softmin", 0.2),
        ("softmin", 0.3),
    ]
    s_balance_by_objective = {
        # "sum": 0.0,
        "softmin": 0.05,
    }

    def add(name: str, cfg: Config) -> None:
        cfg.lambda_xp = 0.0
        experiments.append((name, cfg))

    for objective, tau in objectives:
        cfg = deepcopy(base_cfg)
        cfg.s_objective = objective
        if tau is not None:
            cfg.s_softmin_tau = tau
        cfg.lambda_bt = 0.0
        cfg.lambda_plane_bt = 0.0
        cfg.lambda_block_cca = 0.0
        cfg.lambda_s_balance = s_balance_by_objective[objective]
        add(f"{objective}_noreg", cfg)

        for weight in [0.02, 0.05, 0.12]:
            cfg = deepcopy(base_cfg)
            cfg.s_objective = objective
            if tau is not None:
                cfg.s_softmin_tau = tau
            cfg.lambda_bt = weight
            cfg.lambda_plane_bt = 0.0
            cfg.lambda_block_cca = 0.0
            cfg.lambda_s_balance = s_balance_by_objective[objective]
            add(f"{objective}_bt_{weight:g}", cfg)

        for weight in [0.002, 0.005, 0.01]:
            cfg = deepcopy(base_cfg)
            cfg.s_objective = objective
            if tau is not None:
                cfg.s_softmin_tau = tau
            cfg.lambda_bt = 0.0
            cfg.lambda_plane_bt = weight
            cfg.lambda_block_cca = 0.0
            cfg.lambda_s_balance = s_balance_by_objective[objective]
            add(f"{objective}_plane_bt_{weight:g}", cfg)

        for weight in [0.5, 1.0, 2.0]:
            cfg = deepcopy(base_cfg)
            cfg.s_objective = objective
            if tau is not None:
                cfg.s_softmin_tau = tau
            cfg.lambda_bt = 0.0
            cfg.lambda_plane_bt = 0.0
            cfg.lambda_block_cca = weight
            cfg.lambda_s_balance = s_balance_by_objective[objective]
            add(f"{objective}_block_cca_{weight:g}", cfg)

    return experiments


def prepare_data(cfg: Config):
    from data import load_mcmaze_cached, gaussian_smooth, soft_normalize, make_windows, train_val_split

    spikes_raw, bin_width_s, trial_info, time_index_s, _ = load_mcmaze_cached(
        cfg.nwb_path, cfg.bin_ms
    )
    sigma_samples = round((cfg.sigma_ms * 1e-3) / bin_width_s)
    X_smooth = gaussian_smooth(spikes_raw, sigma_samples)
    if cfg.softnorm_method and cfg.softnorm_method != "none":
        X_smooth = soft_normalize(X_smooth, method=cfg.softnorm_method)

    windows = make_windows(
        X_smooth,
        trial_info,
        time_index_s,
        bin_width_s,
        strategy=cfg.window_strategy,
        window_size=cfg.window_size,
        align_field=cfg.align_field,
        pre_ms=cfg.pre_ms,
    )
    if cfg.split == "random":
        trial_info = trial_info.drop(columns=["split"], errors="ignore")
    train_ds, val_ds = train_val_split(windows, trial_info, cfg.val_split, cfg.seed)
    return windows.shape[1], train_ds, val_ds


def pearson_corr_metrics(F: torch.Tensor) -> tuple[float, float]:
    K, d, T = F.shape
    Z = F.permute(0, 2, 1).reshape(K * T, d)
    Z = Z - Z.mean(dim=0, keepdim=True)
    Z = Z / (Z.std(dim=0, keepdim=True) + 1e-6)
    corr = (Z.T @ Z) / Z.shape[0]
    offdiag = corr[~torch.eye(d, dtype=torch.bool, device=F.device)].abs()
    return float(offdiag.mean()), float(offdiag.max())


def evaluate_checkpoint(run_dir: str, cfg: Config, in_channels: int, train_ds, val_ds) -> dict:
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = MLP(
        in_channels=in_channels,
        d=cfg.d,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        dropout=cfg.dropout,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    (batch,) = next(iter(loader))
    with torch.no_grad():
        F = model(batch)
        F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
        F_hat = _batch_rms_normalize(F)

        S_p = non_reversibility_S_per_plane(F_hat)
        S_mean = S_p.mean()
        S_std = S_p.std(unbiased=False)
        val_loss = loss_fn(F, cfg=cfg, training=False)
        pearson_mean, pearson_max = pearson_corr_metrics(F)

        metrics = {
            "best_epoch": ckpt["epoch"],
            "val_loss_recomputed": float(val_loss),
            "S_ratio": float(S_ratio(F_hat)),
            "S_total": float(non_reversibility_S(F_hat)),
            "S_mean": float(S_mean),
            "S_min": float(S_p.min()),
            "S_max": float(S_p.max()),
            "S_cv": float(S_std / (S_mean.abs() + 1e-8)),
            "S_balance_raw": float(plane_s_balance_reg(S_p)),
            "bt_raw": float(barlow_twins_reg(F)),
            "plane_bt_raw": float(plane_barlow_twins_reg(F)),
            "block_cca_raw": float(block_cca_reg(F, eps=cfg.block_cca_eps)),
            "pearson_offdiag_mean_abs": pearson_mean,
            "pearson_offdiag_max_abs": pearson_max,
        }

    log_path = os.path.join(run_dir, "outputs", "log.csv")
    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))
    best_row = min(rows, key=lambda r: float(r["val_loss"]))
    metrics["best_train_loss"] = float(best_row["train_loss"])
    metrics["best_val_loss"] = float(best_row["val_loss"])
    return metrics


def score_for_ranking(row: dict) -> float:
    """Lower is better; avoids raw val loss because objectives differ."""
    return (
        -row["S_min"]
        + 0.25 * row["S_cv"]
        + 0.5 * row["block_cca_raw"]
        + 0.5 * row["pearson_offdiag_mean_abs"]
    )


def write_summary(summary_path: str, rows: list[dict]) -> None:
    ranked = sorted(rows, key=score_for_ranking)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in ranked:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_COLUMNS})


def print_top(rows: list[dict], n: int = 10) -> None:
    ranked = sorted(rows, key=score_for_ranking)
    print("\nTop runs by comparable score:")
    print("rank  experiment                 S_min   S_cv   blockCCA  |r|mean  S_ratio")
    for i, row in enumerate(ranked[:n], start=1):
        print(
            f"{i:>4}  {row['experiment']:<26} "
            f"{row['S_min']:>6.3f}  {row['S_cv']:>5.3f}  "
            f"{row['block_cca_raw']:>8.4f}  {row['pearson_offdiag_mean_abs']:>7.4f}  "
            f"{row['S_ratio']:>7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--sweep-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_cfg = Config()
    base_cfg.epochs = args.epochs
    base_cfg.seed = args.seed
    base_cfg.dropout = args.dropout

    experiments = make_experiment_grid(base_cfg)[:args.max_runs]
    if args.dry_run:
        for idx, (name, cfg) in enumerate(experiments, start=1):
            print(
                f"{idx:02d} {name}: s={cfg.s_objective} "
                f"bt={cfg.lambda_bt:g} plane_bt={cfg.lambda_plane_bt:g} "
                f"block_cca={cfg.lambda_block_cca:g} s_balance={cfg.lambda_s_balance:g}"
            )
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = args.sweep_dir or os.path.join("runs", f"regularizer_sweep_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)
    summary_path = os.path.join(sweep_dir, "summary.csv")
    manifest_path = os.path.join(sweep_dir, "manifest.json")

    with open(manifest_path, "w") as f:
        json.dump(
            [{"experiment": name, "config": asdict(cfg)} for name, cfg in experiments],
            f,
            indent=2,
        )

    print("Preparing data once for all experiments...")
    in_channels, train_ds, val_ds = prepare_data(base_cfg)
    print(f"Train={len(train_ds)}  Val={len(val_ds)}  in_channels={in_channels}")

    rows: list[dict] = []
    for idx, (name, cfg) in enumerate(experiments, start=1):
        run_dir = os.path.join(sweep_dir, f"{idx:02d}_{name}")
        cfg.ckpt_dir = os.path.join(run_dir, "checkpoints")
        cfg.out_dir = os.path.join(run_dir, "outputs")
        os.makedirs(run_dir, exist_ok=True)
        cfg.save_about(run_dir)

        if os.path.exists(os.path.join(cfg.ckpt_dir, "best.pt")):
            print(f"\n[{idx}/{len(experiments)}] Skipping completed {name}")
        else:
            print(f"\n[{idx}/{len(experiments)}] Training {name}")
            set_seed(cfg.seed)
            model = MLP(
                in_channels=in_channels,
                d=cfg.d,
                hidden_dim=cfg.hidden_dim,
                depth=cfg.depth,
                dropout=cfg.dropout,
            )
            t0 = time.time()
            train(model, train_ds, val_ds, cfg)
            print(f"Finished {name} in {(time.time() - t0) / 60:.1f} min")

        metrics = evaluate_checkpoint(run_dir, cfg, in_channels, train_ds, val_ds)
        row = {
            "experiment": name,
            "run_dir": run_dir,
            "d": cfg.d,
            "epochs": cfg.epochs,
            "s_objective": cfg.s_objective,
            "s_softmin_tau": cfg.s_softmin_tau,
            "lambda_bt": cfg.lambda_bt,
            "lambda_plane_bt": cfg.lambda_plane_bt,
            "lambda_block_cca": cfg.lambda_block_cca,
            "lambda_s_balance": cfg.lambda_s_balance,
            "dropout": cfg.dropout,
            "seed": cfg.seed,
            "elapsed_s": "",
        }
        row.update(metrics)
        rows.append(row)
        write_summary(summary_path, rows)
        print_top(rows, n=min(5, len(rows)))

    write_summary(summary_path, rows)
    print(f"\nWrote summary: {summary_path}")
    print_top(rows)


if __name__ == "__main__":
    main()
