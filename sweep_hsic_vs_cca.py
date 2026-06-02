"""
Head-to-head sweep: HSIC vs block CCA plane-redundancy regularizers.

Both penalize cross-plane dependence, but differ in what they can see:
  - block CCA  : linear (canonical) correlation between whole 2D planes
  - HSIC       : arbitrary nonlinear dependence via RBF-kernel alignment

Only one regularizer is active per run (sum-S objective, no BT/plane-BT).
Every run reports BOTH block_cca_raw and hsic_raw so each method can be
judged on the other's metric. Ranking uses objective-neutral metrics
(S_min, S_cv, Pearson off-diagonal) so neither method is favoured by
construction.

After all runs, generates CCA diagnostic figures for the top 5.

Usage
-----
    python sweep_hsic_vs_cca.py --dry-run
    python sweep_hsic_vs_cca.py
    python sweep_hsic_vs_cca.py --epochs 100
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
    block_cca_reg,
    non_reversibility_S,
    non_reversibility_S_per_plane,
    plane_hsic_reg,
)
from model import MLP
from train import train
from visualize_cca import plot_cca_diagnostic


# Deterministic settings for the post-hoc HSIC estimate so it is comparable
# across runs (HSIC subsamples snapshots stochastically).
EVAL_HSIC_SEED = 0
EVAL_HSIC_MAX_SAMPLES = 2048


SUMMARY_COLUMNS = [
    "rank",
    "experiment",
    "reg_type",
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
    "block_cca_raw",
    "hsic_raw",
    "pearson_offdiag_mean_abs",
    "pearson_offdiag_max_abs",
    "lambda_block_cca",
    "lambda_hsic",
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
    cfg.lambda_s_balance = 0.0
    cfg.lambda_bt = 0.0
    cfg.lambda_plane_bt = 0.0
    cfg.lambda_block_cca = 0.0
    cfg.lambda_hsic = 0.0
    cfg.block_cca_eps = 1e-4
    cfg.hsic_sigma = 0.0          # median-distance heuristic
    return cfg


def reg_type_of(cfg: Config) -> str:
    if cfg.lambda_block_cca > 0:
        return "cca"
    if cfg.lambda_hsic > 0:
        return "hsic"
    return "none"


def make_experiment_grid(base_cfg: Config) -> list[tuple[str, Config]]:
    experiments: list[tuple[str, Config]] = []

    def add(name: str, cfg: Config) -> None:
        experiments.append((name, cfg))

    # ── Baseline: no redundancy regularizer ───────────────────────────
    add("baseline_noreg", make_base(base_cfg))

    # ── block CCA sweep (linear) ──────────────────────────────────────
    for lam in [0.5, 1.0, 2.0, 3.0, 5.0]:
        cfg = make_base(base_cfg)
        cfg.lambda_block_cca = lam
        add(f"cca_{lam:g}", cfg)

    # ── HSIC sweep (nonlinear) ────────────────────────────────────────
    for lam in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        cfg = make_base(base_cfg)
        cfg.lambda_hsic = lam
        add(f"hsic_{lam:g}", cfg)

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

        # Deterministic HSIC estimate (fixed seed + sample budget) so it is
        # comparable across runs regardless of which regularizer was active.
        torch.manual_seed(EVAL_HSIC_SEED)
        hsic_raw = float(plane_hsic_reg(F, max_samples=EVAL_HSIC_MAX_SAMPLES, sigma=None))

        metrics = {
            "reg_type": reg_type_of(cfg),
            "best_epoch": ckpt["epoch"],
            "S_ratio": float(S_ratio(F_hat)),
            "S_total": float(non_reversibility_S(F_hat)),
            "S_mean": float(S_mean),
            "S_min": float(S_p.min()),
            "S_max": float(S_p.max()),
            "S_cv": float(S_std / (S_mean.abs() + 1e-8)),
            "block_cca_raw": float(block_cca_reg(F, eps=max(cfg.block_cca_eps, 1e-4))),
            "hsic_raw": hsic_raw,
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
    """Lower is better.

    Objective-neutral so neither CCA nor HSIC is favoured by construction:
      - keep every plane active        -> high S_min
      - keep planes balanced           -> low S_cv
      - linearly decorrelate planes    -> low Pearson off-diagonal
    Each method's own raw objective (block_cca_raw, hsic_raw) is reported in
    the summary for inspection but kept out of the ranking.
    """
    return (
        -row["S_min"]
        + 0.25 * row["S_cv"]
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
        f"{'rank':>4}  {'experiment':<20} {'type':>5} "
        f"{'S_min':>6}  {'S_cv':>5}  {'CCA':>7}  {'HSIC':>7}  {'|r|':>6}  {'S_ratio':>7}"
    )
    print(f"\nTop runs:\n{hdr}")
    for i, row in enumerate(ranked[:n], start=1):
        print(
            f"{i:>4}  {row['experiment']:<20} {row['reg_type']:>5} "
            f"{row['S_min']:>6.3f}  {row['S_cv']:>5.3f}  "
            f"{row['block_cca_raw']:>7.4f}  {row['hsic_raw']:>7.4f}  "
            f"{row['pearson_offdiag_mean_abs']:>6.4f}  {row['S_ratio']:>7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep-dir", default=None)
    parser.add_argument("--top-vis", type=int, default=5,
                        help="Generate CCA diagnostic figures for the top N runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_cfg = Config()
    base_cfg.epochs = args.epochs
    base_cfg.seed = args.seed

    experiments = make_experiment_grid(base_cfg)[:args.max_runs]

    if args.dry_run:
        print(f"HSIC vs CCA sweep: {len(experiments)} experiments, {args.epochs} epochs\n")
        print(f"{'#':>3}  {'experiment':<20} {'type':>5} {'cca':>6} {'hsic':>6}")
        print("-" * 46)
        for idx, (name, cfg) in enumerate(experiments, start=1):
            print(
                f"{idx:>3}  {name:<20} {reg_type_of(cfg):>5} "
                f"{cfg.lambda_block_cca:>6g} {cfg.lambda_hsic:>6g}"
            )
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = args.sweep_dir or os.path.join("runs", f"hsic_vs_cca_{timestamp}")
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
            print(f"  Done in {(time.time() - t0) / 60:.1f} min")

        metrics = evaluate_checkpoint(run_dir, cfg, in_channels, val_ds)
        row = {
            "experiment": name,
            "run_dir": run_dir,
            "lambda_block_cca": cfg.lambda_block_cca,
            "lambda_hsic": cfg.lambda_hsic,
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

    # ── Generate CCA diagnostic figures for top N ─────────────────────
    if args.top_vis > 0:
        ranked = sorted(rows, key=score_for_ranking)
        top_n = ranked[:args.top_vis]
        print(f"\n{'='*60}")
        print(f"Generating CCA diagnostic figures for top {len(top_n)} runs...")
        print(f"{'='*60}")
        for i, row in enumerate(top_n, start=1):
            run_dir = row["run_dir"]
            out_path = os.path.join(
                sweep_dir, f"top{i}_{row['reg_type']}_{row['experiment']}_cca.png"
            )
            try:
                plot_cca_diagnostic(run_dir, out_path=out_path)
            except Exception as e:
                print(f"  Warning: failed on {row['experiment']}: {e}")
        print(f"\nDiagnostic figures saved to: {sweep_dir}/top*_cca.png")


if __name__ == "__main__":
    main()
