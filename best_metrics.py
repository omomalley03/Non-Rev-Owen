"""Compute validation-set metrics for the best checkpoint and append to about.txt."""

import os

import torch
from torch.utils.data import DataLoader

from config import Config
from model import MLP
from loss import (
    non_reversibility_S,
    non_reversibility_S_per_plane,
    barlow_twins_reg,
    non_rev_regularizer,
    block_cca_reg,
    _batch_rms_normalize,
)


def append_best_model_metrics(run_dir: str, val_ds, cfg: Config, n_xp_perms: int = 30) -> None:
    """Load best.pt, evaluate it on the full val set, and append metrics to about.txt.

    Metrics (all on the validation set):
      - non-reversibility score averaged across all native planes
      - the non-reversibility score for each native plane
      - raw magnitude of each regularizer we consider: Barlow Twins,
        cross-plane non-rev regularizer (averaged over `n_xp_perms` random
        dimension permutations), and block-CCA.
    """
    ckpt_path = os.path.join(cfg.ckpt_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
    (val_tensor,) = next(iter(loader))
    N_in = val_tensor.shape[1]

    model = MLP(
        in_channels=N_in, d=cfg.d, hidden_dim=cfg.hidden_dim,
        depth=cfg.depth, dropout=cfg.dropout,
        temporal_filters=getattr(cfg, "temporal_filters", 0),
        temporal_kernel_size=getattr(cfg, "temporal_kernel_size", 31),
        temporal_frontend=getattr(cfg, "temporal_frontend", "symmetric"),
        residual_kernels=getattr(cfg, "residual_kernels", "3,7,15,31"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        F = model(val_tensor)                                 # (K, d, T)
        F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
        F_hat = _batch_rms_normalize(F)

        objective = getattr(cfg, "s_objective", "sum")
        # always report mean S (per-plane average) as the diagnostic metric regardless
        # of training objective — "softmin" falls through to "sum" in non_reversibility_S
        agg_S = non_reversibility_S(F_hat, objective="mean").item()
        per_plane = non_reversibility_S_per_plane(F_hat).tolist()      # (D,)

        bt_mag = barlow_twins_reg(F).item()
        cca_mag = block_cca_reg(F).item()

        # always use "mean" for cross-plane non-rev so runs with different
        # s_objective (e.g. "softmin") remain directly comparable
        torch.manual_seed(cfg.seed)  # reproducible permutations
        xp_vals = [non_reversibility_S(
                       F_hat[:, torch.randperm(F_hat.shape[1]), :], objective="mean"
                   ).item() for _ in range(n_xp_perms)]
        xp_mag = sum(xp_vals) / len(xp_vals)

    lines = [
        "",
        "[best model — validation metrics]",
        f"  best_epoch                       = {ckpt.get('epoch')}",
        f"  s_objective                      = {objective}",
        f"  non_rev_S [mean/plane]               = {agg_S:.6f}",
        "  non_rev_S per plane              = [" + ", ".join(f"{v:.6f}" for v in per_plane) + "]",
        f"  barlow_twins (raw)               = {bt_mag:.6f}",
        f"  cross_plane_non_rev (raw, {n_xp_perms}-perm avg) = {xp_mag:.6f}",
        f"  block_cca (raw)                  = {cca_mag:.6f}",
        "",
    ]
    with open(os.path.join(run_dir, "about.txt"), "a") as f:
        f.write("\n".join(lines))
    print(f"Appended best-model val metrics → {os.path.join(run_dir, 'about.txt')}")
