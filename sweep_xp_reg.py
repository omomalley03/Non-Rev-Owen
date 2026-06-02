"""
Sweep the cross-plane non-reversibility regularizer (non_rev_regularizer)
in isolation — no BT, no plane_bt, no block_cca.

The only regularizer active is lambda_xp * non_rev_regularizer(F_hat),
applied planes//2 times per step with random dim shuffles.

After all runs, generates pairwise S diagnostic figures for the top 5.

Usage
-----
    python sweep_xp_reg.py --dry-run
    python sweep_xp_reg.py
    python sweep_xp_reg.py --epochs 100
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
)
from model import MLP
from train import train
from visualize_pairwise_s import plot_pairwise_s


SUMMARY_COLUMNS = [
    "rank",
    "experiment",
    "run_dir",
    "best_epoch",
    "best_train_loss",
    "best_val_loss",
    "S_ratio",
    "S_total",
    "S_mean",
    "S_min",
    "S_max",
    "S_cv",
    "within_S_mean",
    "cross_S_mean",
    "cross_S_max",
    "within_cross_ratio",
    "block_cca_raw",
    "pearson_offdiag_mean_abs",
    "pearson_offdiag_max_abs",
    "lambda_xp",
    "epochs",
    "seed",
    "elapsed_s",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_base(base_cfg: Config) -> Config:
    cfg = deepcopy(base_cfg)
    cfg.s_objective = "sum"
    cfg.s_softmin_tau = 0.0
    cfg.lambda_xp = 0.0
    cfg.lambda_bt = 0.0
    cfg.lambda_plane_bt = 0.0
    cfg.lambda_block_cca = 0.0
    cfg.lambda_s_balance = 0.0
    cfg.block_cca_eps = 1e-4
    return cfg


def make_experiment_grid(base_cfg: Config) -> list[tuple[str, Config]]:
    experiments: list[tuple[str, Config]] = []

    def add(name: str, cfg: Config) -> None:
        experiments.append((name, cfg))

    # Baseline: no regularization at all
    cfg = make_base(base_cfg)
    add("baseline_noreg", cfg)

    # Sweep lambda_xp
    for lam in [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]:
        cfg = make_base(base_cfg)
        cfg.lambda_xp = lam
        add(f"xp_{lam:g}", cfg)

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
        X_smooth, trial_info, time_index_s, bin_width_s,
        strategy=cfg.window_strategy, window_size=cfg.window_size,
        align_field=cfg.align_field, pre_ms=cfg.pre_ms,
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


def pairwise_S_stats(F_hat: torch.Tensor) -> dict:
    """Compute within-plane vs cross-plane S statistics."""
    K, d, T = F_hat.shape
    D = d // 2

    within = []
    cross = []
    for i in range(d):
        for j in range(i + 1, d):
            s = non_reversibility_S(F_hat[:, [i, j], :]).item()
            if i // 2 == j // 2:
                within.append(s)
            else:
                cross.append(s)

    within = np.array(within)
    cross = np.array(cross)
    return {
        "within_S_mean": float(within.mean()),
        "cross_S_mean": float(cross.mean()),
        "cross_S_max": float(cross.max()),
        "within_cross_ratio": float(within.mean() / (cross.mean() + 1e-8)),
    }


def evaluate_checkpoint(run_dir: str, cfg: Config, in_channels: int, val_ds) -> dict:
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = MLP(
        in_channels=in_channels, d=cfg.d, hidden_dim=cfg.hidden_dim,
        depth=cfg.depth, dropout=getattr(cfg, "dropout", 0.0),
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
        pearson_mean, pearson_max = pearson_corr_metrics(F)

        metrics = {
            "best_epoch": ckpt["epoch"],
            "S_ratio": float(S_ratio(F_hat)),
            "S_total": float(non_reversibility_S(F_hat)),
            "S_mean": float(S_mean),
            "S_min": float(S_p.min()),
            "S_max": float(S_p.max()),
            "S_cv": float(S_std / (S_mean.abs() + 1e-8)),
            "block_cca_raw": float(block_cca_reg(F, eps=1e-4)),
            "pearson_offdiag_mean_abs": pearson_mean,
            "pearson_offdiag_max_abs": pearson_max,
        }

        # Pairwise S breakdown
        metrics.update(pairwise_S_stats(F_hat))

    log_path = os.path.join(run_dir, "outputs", "log.csv")
    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))
    best_row = min(rows, key=lambda r: float(r["val_loss"]))
    metrics["best_train_loss"] = float(best_row["train_loss"])
    metrics["best_val_loss"] = float(best_row["val_loss"])
    return metrics


def score_for_ranking(row: dict) -> float:
    """Lower is better.

    Primary: high S_min (all planes active).
    Secondary: high within/cross ratio (regularizer doing its job),
               low cross-plane S, low Pearson.
    """
    return (
        -row["S_min"]
        + 0.25 * row["S_cv"]
        - 0.3 * min(row["within_cross_ratio"], 20) / 20  # reward separation, cap at 20x
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
    hdr = (
        f"{'rank':>4}  {'experiment':<20} "
        f"{'S_min':>6}  {'S_cv':>5}  {'W/C':>5}  "
        f"{'cross_S':>7}  {'|r|':>6}  {'S_ratio':>7}"
    )
    print(f"\nTop runs:\n{hdr}")
    for i, row in enumerate(ranked[:n], start=1):
        print(
            f"{i:>4}  {row['experiment']:<20} "
            f"{row['S_min']:>6.3f}  {row['S_cv']:>5.3f}  "
            f"{row['within_cross_ratio']:>5.1f}  "
            f"{row['cross_S_mean']:>7.4f}  {row['pearson_offdiag_mean_abs']:>6.4f}  "
            f"{row['S_ratio']:>7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep-dir", default=None)
    parser.add_argument("--top-vis", type=int, default=5,
                        help="Generate pairwise S figures for the top N runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_cfg = Config()
    base_cfg.epochs = args.epochs
    base_cfg.seed = args.seed

    experiments = make_experiment_grid(base_cfg)[:args.max_runs]

    if args.dry_run:
        print(f"Cross-plane reg sweep: {len(experiments)} experiments, {args.epochs} epochs\n")
        print(f"{'#':>3}  {'experiment':<20} {'lambda_xp':>10}")
        print("-" * 38)
        for idx, (name, cfg) in enumerate(experiments, start=1):
            print(f"{idx:>3}  {name:<20} {cfg.lambda_xp:>10g}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = args.sweep_dir or os.path.join("runs", f"xp_reg_sweep_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)
    summary_path = os.path.join(sweep_dir, "summary.csv")
    manifest_path = os.path.join(sweep_dir, "manifest.json")

    with open(manifest_path, "w") as f:
        json.dump(
            [{"experiment": name, "config": asdict(cfg)} for name, cfg in experiments],
            f, indent=2,
        )

    print("Preparing data once for all experiments...")
    in_channels, train_ds, val_ds = prepare_data(base_cfg)
    print(f"Train={len(train_ds)}  Val={len(val_ds)}  in_channels={in_channels}\n")

    rows: list[dict] = []
    for idx, (name, cfg) in enumerate(experiments, start=1):
        run_dir = os.path.join(sweep_dir, f"{idx:02d}_{name}")
        cfg.ckpt_dir = os.path.join(run_dir, "checkpoints")
        cfg.out_dir = os.path.join(run_dir, "outputs")
        os.makedirs(run_dir, exist_ok=True)
        cfg.save_about(run_dir)

        if os.path.exists(os.path.join(cfg.ckpt_dir, "best.pt")):
            print(f"\n[{idx}/{len(experiments)}] Skipping completed: {name}")
        else:
            print(f"\n[{idx}/{len(experiments)}] Training: {name}")
            set_seed(cfg.seed)
            model = MLP(
                in_channels=in_channels, d=cfg.d, hidden_dim=cfg.hidden_dim,
                depth=cfg.depth, dropout=getattr(cfg, "dropout", 0.0),
            )
            t0 = time.time()
            train(model, train_ds, val_ds, cfg)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed / 60:.1f} min")

        print(f"  Evaluating (includes pairwise S computation)...")
        metrics = evaluate_checkpoint(run_dir, cfg, in_channels, val_ds)
        row = {
            "experiment": name,
            "run_dir": run_dir,
            "lambda_xp": cfg.lambda_xp,
            "epochs": cfg.epochs,
            "seed": cfg.seed,
            "elapsed_s": "",
        }
        row.update(metrics)
        rows.append(row)
        write_summary(summary_path, rows)
        print_top(rows, n=min(5, len(rows)))

    write_summary(summary_path, rows)
    print(f"\nFinal summary: {summary_path}")
    print_top(rows, n=len(rows))

    # ── Generate pairwise S figures for top N ─────────────────────────
    if args.top_vis > 0:
        ranked = sorted(rows, key=score_for_ranking)
        top_n = ranked[:args.top_vis]
        print(f"\n{'='*60}")
        print(f"Generating pairwise S figures for top {len(top_n)} runs...")
        print(f"{'='*60}")
        for i, row in enumerate(top_n, start=1):
            run_dir = row["run_dir"]
            out_path = os.path.join(sweep_dir, f"top{i}_{row['experiment']}_pairwise_s.png")
            try:
                plot_pairwise_s(run_dir, out_path=out_path)
            except Exception as e:
                print(f"  Warning: failed on {row['experiment']}: {e}")
        print(f"\nPairwise S figures saved to: {sweep_dir}/top*_pairwise_s.png")


if __name__ == "__main__":
    main()
