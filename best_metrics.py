"""Compute validation-set metrics for the best checkpoint and append to about.txt."""

import os

import torch
from torch.utils.data import DataLoader

from config import Config
from model import MLP, infer_multiscale_symmetric_conv_layers
from loss import (
    non_reversibility_S_per_plane,
    non_reversibility_components,
    _batch_rms_normalize,
    loss_fn,
)


def append_best_model_metrics(run_dir: str, val_ds, cfg: Config, n_xp_perms: int = 30) -> None:
    """Load best.pt, evaluate validation batches, and append metrics to about.txt.

    Metrics (all on the validation set):
      - non-reversibility score averaged across all native planes
      - the non-reversibility score for each native plane
      - active regularizer magnitudes from loss_fn, using the same lambda
        weights and lambda warm-up scale as the saved best epoch.
    """
    ckpt_path = os.path.join(cfg.ckpt_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    probe_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    (probe_tensor,) = next(iter(probe_loader))
    N_in = probe_tensor.shape[1]

    model = MLP(
        in_channels=N_in, d=cfg.d, hidden_dim=cfg.hidden_dim,
        depth=cfg.depth, dropout=cfg.dropout,
        temporal_filters=getattr(cfg, "temporal_filters", 0),
        temporal_kernel_size=getattr(cfg, "temporal_kernel_size", 31),
        temporal_frontend=getattr(cfg, "temporal_frontend", "symmetric"),
        residual_kernels=getattr(cfg, "residual_kernels", "3,7,15,31"),
        multiscale_symmetric_conv_layers=infer_multiscale_symmetric_conv_layers(
            ckpt["model_state_dict"],
            getattr(cfg, "multiscale_symmetric_conv_layers", 1),
        ),
        antisymmetric_planes=getattr(cfg, "antisymmetric_planes", 0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    objective = getattr(cfg, "s_objective", "sum")
    best_epoch = int(ckpt.get("epoch", cfg.epochs))
    lambda_start = float(getattr(cfg, "lambda_start_frac", 1.0))
    if cfg.epochs <= 1:
        lambda_scale = 1.0
    else:
        lambda_scale = lambda_start + (1.0 - lambda_start) * (best_epoch - 1) / (cfg.epochs - 1)

    # Mirror train.py validation exactly: fixed-size validation batches,
    # unweighted mean over batches, and skip singleton batches.
    loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    val_losses = []
    val_s = []
    val_c_plus = []
    val_zeta = []
    val_reg_log_equiv = []
    per_plane_vals = []
    raw_vals = {}
    scaled_vals = {}
    whole_batch_reg_log_equiv = float("nan")
    whole_batch_reg_scaled_total = float("nan")

    with torch.no_grad():
        for (batch,) in loader:
            if batch.shape[0] < 2:
                continue
            F = model(batch)                                 # (K, d, T)
            F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
            F_hat = _batch_rms_normalize(F)

            loss, reg_info = loss_fn(
                F,
                cfg=cfg,
                training=False,
                lambda_scale=lambda_scale,
                return_components=True,
            )
            c_minus, c_plus, zeta = non_reversibility_components(F_hat, "mean")
            s = c_minus.item()

            val_losses.append(loss.item())
            val_s.append(s)
            val_c_plus.append(c_plus.item())
            val_zeta.append(zeta.item())
            val_reg_log_equiv.append(loss.item() + s)
            per_plane_vals.append(non_reversibility_S_per_plane(F_hat).cpu())
            for name, value in reg_info["reg_raw"].items():
                raw_vals.setdefault(name, []).append(value)
            for name, value in reg_info["reg_scaled"].items():
                scaled_vals.setdefault(name, []).append(value)

        whole_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
        (whole_batch,) = next(iter(whole_loader))
        if whole_batch.shape[0] >= 2:
            F = model(whole_batch)
            F = F - F.mean(dim=cfg.F_mean_axis, keepdim=True)
            F_hat = _batch_rms_normalize(F)
            loss, reg_info = loss_fn(
                F,
                cfg=cfg,
                training=False,
                lambda_scale=lambda_scale,
                return_components=True,
            )
            c_minus, _, _ = non_reversibility_components(F_hat, "mean")
            whole_batch_reg_log_equiv = loss.item() + c_minus.item()
            whole_batch_reg_scaled_total = sum(reg_info["reg_scaled"].values())

    if not val_losses:
        raise ValueError("Validation set produced no batches with at least two trials.")

    agg_S = sum(val_s) / len(val_s)
    per_plane = torch.stack(per_plane_vals).mean(dim=0).tolist()
    validation_loss = sum(val_losses) / len(val_losses)
    validation_c_plus = sum(val_c_plus) / len(val_c_plus)
    validation_zeta = sum(val_zeta) / len(val_zeta)
    reg_log_equiv = sum(val_reg_log_equiv) / len(val_reg_log_equiv)
    reg_raw = {name: sum(values) / len(values) for name, values in raw_vals.items()}
    reg_scaled = {name: sum(values) / len(values) for name, values in scaled_vals.items()}
    reg_scaled_total = sum(reg_scaled.values())

    lines = [
        "",
        "[best model — validation metrics]",
        f"  best_epoch                       = {best_epoch}",
        f"  s_objective                      = {objective}",
        f"  lambda_scale                     = {lambda_scale:.6f}",
        f"  validation_batches               = {len(val_losses)}",
        f"  non_rev_S [mean/plane]               = {agg_S:.6f}",
        "  non_rev_S per plane              = [" + ", ".join(f"{v:.6f}" for v in per_plane) + "]",
        f"  validation_c_plus                = {validation_c_plus:.6f}",
        f"  validation_zeta                  = {validation_zeta:.6f}",
        f"  validation_loss                  = {validation_loss:.6f}",
        f"  regularization_train_log_equiv   = {reg_log_equiv:.6f}",
        f"  regularization_whole_batch_log_equiv = {whole_batch_reg_log_equiv:.6f}",
        f"  regularization_total_scaled      = {reg_scaled_total:.6f}",
        f"  regularization_whole_batch_total_scaled = {whole_batch_reg_scaled_total:.6f}",
    ]
    for name in sorted(reg_raw):
        lines.append(f"  regularization_{name}_raw         = {reg_raw[name]:.6f}")
        lines.append(f"  regularization_{name}_scaled      = {reg_scaled[name]:.6f}")
    if not reg_raw:
        lines.append("  regularization_active            = none")
    lines.append("")
    with open(os.path.join(run_dir, "about.txt"), "a") as f:
        f.write("\n".join(lines))
    print(f"Appended best-model val metrics → {os.path.join(run_dir, 'about.txt')}")
