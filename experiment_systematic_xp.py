"""
Experiment: train with the SYSTEMATIC cross-plane non-reversibility regularizer
(loss.non_rev_regularizer_systematic), then dump the full d x d matrix of
pairwise non-reversibility scores as a heatmap (PNG) and a CSV.

Hypothesis under test
---------------------
The systematic regularizer penalises every dimension pair that is NOT a native
rotation plane -- i.e. all pairs except (0,1),(2,3),...  So after training the
d x d non-rev matrix should be strongly block-diagonal:
    * HIGH  at native planes  (0,1),(2,3),...   (driven up by the -S loss)
    * LOW   at crossed pairs   (0,2),(1,3),...   (driven down by the regularizer)

To isolate the effect, only the primary -S term and the systematic penalty are
active here (Barlow-Twins / plane-BT / block-CCA are switched off).

Run from an environment that has nlb_tools (e.g. the notebook kernel):
    python experiment_systematic_xp.py

Outputs (under runs/<timestamp>_systematic_xp_.../outputs/):
    pairwise_s.png          heatmap + within-vs-cross comparison
    pairwise_s_matrix.csv   the d x d matrix, labelled d0..d{d-1}
"""
import os
import random
from datetime import datetime

import numpy as np
import torch

from config import Config
from data import load_mcmaze_cached, gaussian_smooth, soft_normalize, make_windows, train_val_split
from model import MLP
from train import train
from loss import _batch_rms_normalize, non_reversibility_S, non_rev_regularizer_systematic
from visualize_pairwise_s import plot_pairwise_s

# ── tunables ────────────────────────────────────────────────────────────────
EPOCHS = 100
LAMBDA_XP = 0.2          # weight on the systematic cross-plane penalty
SEED = 0


def loss_fn_systematic(F, cfg=None, training=True):
    """Primary -S (rewards native-plane rotation) + systematic cross-plane penalty.

    Mirrors loss_fn's primary term but swaps the random cross-plane regularizer
    for the exhaustive one and drops the other regularizers, so the matrix test
    reflects the systematic penalty alone. Both terms use the RMS-normalised
    embedding, matching loss_fn.
    """
    F_hat = _batch_rms_normalize(F)
    loss = -non_reversibility_S(F_hat)
    lam = getattr(cfg, "lambda_xp", LAMBDA_XP)
    if lam > 0:
        loss = loss + lam * non_rev_regularizer_systematic(F_hat)
    return loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    cfg = Config()
    cfg.epochs = EPOCHS
    cfg.seed = SEED
    cfg.lambda_xp = LAMBDA_XP
    # isolate the systematic penalty: turn the other regularizers off
    cfg.lambda_bt = 0.0
    cfg.lambda_plane_bt = 0.0
    cfg.lambda_block_cca = 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{timestamp}_systematic_xp_{cfg.run_name()}")
    cfg.ckpt_dir = os.path.join(run_dir, "checkpoints")
    cfg.out_dir = os.path.join(run_dir, "outputs")
    cfg.save_about(run_dir)
    print(f"Run directory: {run_dir}")
    set_seed(cfg.seed)

    # ── data (same pipeline as main.py) ──────────────────────────────────────
    print("Loading MC_Maze data …")
    spikes_raw, bin_width_s, trial_info, time_index_s, _ = load_mcmaze_cached(cfg.nwb_path, cfg.bin_ms)
    N = spikes_raw.shape[0]
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
    print(f"  Channels: {N}  |  Windows: {tuple(windows.shape)}  |  Train/Val: {len(train_ds)}/{len(val_ds)}")

    # ── model + training with the systematic loss injected ───────────────────
    model = MLP(in_channels=N, d=cfg.d, hidden_dim=cfg.hidden_dim, depth=cfg.depth, dropout=cfg.dropout)
    print(f"Training {cfg.epochs} epochs  (systematic xp reg, lambda_xp={cfg.lambda_xp}) …\n")
    train(model, train_ds, val_ds, cfg, loss_function=loss_fn_systematic)

    # ── pairwise non-rev matrix: heatmap (PNG) + CSV ─────────────────────────
    print("\nComputing pairwise non-reversibility matrix …")
    plot_pairwise_s(run_dir)   # writes outputs/pairwise_s.png AND pairwise_s_matrix.csv
    _print_hypothesis_check(os.path.join(cfg.out_dir, "pairwise_s_matrix.csv"))


def _print_hypothesis_check(csv_path: str):
    """Read back the matrix CSV and report native-plane vs crossed-pair scores."""
    M = np.genfromtxt(csv_path, delimiter=",", skip_header=1)[:, 1:]   # drop header row + label col
    d = M.shape[0]
    D = d // 2
    within = np.array([M[2 * p, 2 * p + 1] for p in range(D)])
    cross = np.array([M[i, j] for i in range(d) for j in range(i + 1, d) if i // 2 != j // 2])
    print("\n" + "=" * 56)
    print("Pairwise non-reversibility: native planes vs crossed pairs")
    print(f"  native  (0,1),(2,3),… : mean={within.mean():.3f}  min={within.min():.3f}  (n={len(within)})")
    print(f"  crossed (0,2),(1,3),… : mean={cross.mean():.3f}  max={cross.max():.3f}  (n={len(cross)})")
    print(f"  native / crossed ratio : {within.mean() / (cross.mean() + 1e-8):.1f}x")
    print("=" * 56)


if __name__ == "__main__":
    main()
