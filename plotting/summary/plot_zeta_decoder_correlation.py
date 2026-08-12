"""Compare validation zeta metrics against frozen decoder R^2 for MC Maze sweeps.

This combines the recent finetune sweep result CSVs, resolves each row's
checkpoint path, fills in validation

    zeta_emb  = sum_p C^-_p / sum_p C^+_p
    mean_val_zeta = mean_p C^-_p / C^+_p, computed over the whole validation set

then plots each metric against frozen-decoder R^2 and RMSE.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "model.py").is_file())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_nonrev")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from loss import _batch_rms_normalize, _pair_terms_per_plane
from plotting.mcmaze.plot_mcmaze_plane_spectral_redundancy import (
    build_embedder,
    load_windows_for_run,
)


ROOT = REPO_ROOT
DEFAULT_SWEEPS = [
    ROOT / "mcmaze" / "cca_sweep_constant_lambda",
    ROOT / "mcmaze" / "cca_warmup_finetune_sweep",
    ROOT / "mcmaze" / "dim_train_finetune_sweep",
    ROOT / "mcmaze" / "dim_train_finetune_sweep_mean_plane_zeta",
    ROOT / "mcmaze" / "odd_plane_fraction_finetune_sweep",
]
GROUP_COLUMNS = [
    "sweep",
    "dimension",
    "lambda_start_frac",
    "lambda_block_cca",
    "odd_plane_fraction",
    "antisymmetric_planes",
    "symmetric_planes",
    "n_planes",
    "checkpoint_selection",
]
SWEEP_STYLE = {
    "cca_sweep_constant_lambda": ("tab:blue", "o"),
    "cca_warmup_finetune_sweep": ("tab:orange", "s"),
    "dim_train_finetune_sweep": ("tab:green", "^"),
    "dim_train_finetune_sweep_mean_plane_zeta": ("tab:red", "D"),
    "odd_plane_fraction_finetune_sweep": ("tab:purple", "P"),
}


def _cfg_get(cfg, name: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _cfg_cache_key(cfg) -> tuple:
    context_bins = (
        _cfg_get(cfg, "temporal_context_bins", 0)
        if int(_cfg_get(cfg, "temporal_filters", 0)) > 0
        else 0
    )
    return (
        _cfg_get(cfg, "dataset_name", ""),
        _cfg_get(cfg, "nwb_path", ""),
        int(_cfg_get(cfg, "bin_ms", 5)),
        float(_cfg_get(cfg, "sigma_ms", 50.0)),
        _cfg_get(cfg, "softnorm_method", "none"),
        _cfg_get(cfg, "window_strategy", "trial_aligned"),
        int(_cfg_get(cfg, "window_size", 0)),
        _cfg_get(cfg, "align_field", "move_onset_time"),
        int(_cfg_get(cfg, "pre_ms", 100)),
        int(context_bins),
        _cfg_get(cfg, "split", "dataset"),
        float(_cfg_get(cfg, "val_split", 0.2)),
        int(_cfg_get(cfg, "seed", 0)),
    )


def _mean_axes(cfg) -> tuple[int, ...]:
    axes = _cfg_get(cfg, "F_mean_axis", (0, 2))
    if axes is None:
        return ()
    if isinstance(axes, int):
        axes = (axes,)
    return tuple(int(axis) for axis in axes)


def _center_like_training(F: torch.Tensor, cfg) -> torch.Tensor:
    axes = _mean_axes(cfg)
    if not axes:
        return F
    return F - F.mean(dim=axes, keepdim=True)


def _as_float(value, default=float("nan")) -> float:
    if value is None:
        return default
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _ci95(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) < 2:
        return 0.0
    return 1.96 * float(arr.std(ddof=1)) / math.sqrt(len(arr))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _read_rows(sweep_dirs: list[Path]) -> list[dict]:
    rows = []
    for sweep_dir in sweep_dirs:
        path = sweep_dir / "results.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing sweep results CSV: {path}")
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["sweep"] = sweep_dir.name
                rows.append(row)
    return rows


def _checkpoint_path(row: dict) -> Path:
    run_dir = Path(row["run_dir"]).expanduser()
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    return run_dir / "checkpoints" / "best.pt"


def _validation_zetas_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    data_cache: dict,
    batch_size_override: int | None,
) -> tuple[float, float]:
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    state_dict = saved["model_state_dict"]
    data_key = _cfg_cache_key(cfg)
    if data_key not in data_cache:
        dataset_label, in_channels, _train_ds, val_ds, _sample_rate_hz = load_windows_for_run(cfg)
        if dataset_label != "MC Maze":
            raise ValueError(f"Expected MC Maze checkpoint, got {dataset_label}: {checkpoint_path}")
        data_cache[data_key] = (in_channels, val_ds)
    in_channels, val_ds = data_cache[data_key]

    model = build_embedder(cfg, state_dict, in_channels).to(device)
    model.eval()
    batch_size = int(batch_size_override or _cfg_get(cfg, "batch_size", 64))
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    chunks = []
    with torch.no_grad():
        for (batch,) in loader:
            if batch.shape[0] < 2:
                continue
            chunks.append(model(batch.to(device)).detach().cpu())
    if not chunks:
        raise ValueError(f"No valid validation batches for {checkpoint_path}")

    F = torch.cat(chunks, dim=0)
    F = _center_like_training(F, cfg)
    F_norm = _batch_rms_normalize(F)
    minus_per_plane, plus_per_plane = _pair_terms_per_plane(F_norm)
    plane_zeta = minus_per_plane / (plus_per_plane + 1e-8)
    zeta_emb = minus_per_plane.sum() / (plus_per_plane.sum() + 1e-8)
    return float(zeta_emb.item()), float(plane_zeta.mean().item())


def _fill_zeta_metrics(rows: list[dict], device: torch.device, batch_size: int | None, force: bool) -> list[dict]:
    data_cache = {}
    out = []
    recomputed = 0
    for i, row in enumerate(rows, 1):
        checkpoint_path = _checkpoint_path(row)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint for row {i}: {checkpoint_path}")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        zeta_emb = _as_float(saved.get("val_zeta"))
        zeta_mean = _as_float(row.get("mean_val_zeta"))
        if not np.isfinite(zeta_mean):
            zeta_mean = _as_float(saved.get("val_mean_plane_zeta"))
        source = "checkpoint"

        if force or not np.isfinite(zeta_emb) or not np.isfinite(zeta_mean):
            print(f"[{i}/{len(rows)}] recomputing validation zetas: {checkpoint_path}")
            zeta_emb_recomputed, zeta_mean_recomputed = _validation_zetas_from_checkpoint(
                checkpoint_path,
                device,
                data_cache,
                batch_size,
            )
            if force or not np.isfinite(zeta_emb):
                zeta_emb = zeta_emb_recomputed
            if force or not np.isfinite(zeta_mean):
                zeta_mean = zeta_mean_recomputed
            source = "recomputed" if force else "checkpoint+recomputed_missing"

        enriched = dict(row)
        enriched["checkpoint_path"] = str(checkpoint_path)
        enriched["zeta_emb"] = zeta_emb
        enriched["zeta_mean"] = zeta_mean
        enriched["mean_val_zeta"] = zeta_mean
        enriched["zeta_source"] = source
        enriched["frozen_decoder_r2_mean"] = _as_float(row.get("frozen_decoder_r2_mean"))
        enriched["frozen_decoder_rmse_mean"] = _as_float(row.get("frozen_decoder_rmse_mean"))
        out.append(enriched)
        if "recomputed" in source:
            recomputed += 1
    print(f"Prepared {len(out)} rows; recomputed zeta metrics for {recomputed} rows.")
    return out


def _write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "sweep",
        "dimension",
        "seed",
        "lambda_start_frac",
        "lambda_block_cca",
        "odd_plane_fraction",
        "antisymmetric_planes",
        "symmetric_planes",
        "n_planes",
        "checkpoint_selection",
        "saved_checkpoint_epoch",
        "checkpoint_path",
        "mean_val_zeta",
        "zeta_emb",
        "zeta_mean",
        "zeta_source",
        "frozen_decoder_r2_mean",
        "frozen_decoder_rmse_mean",
        "frozen_decoder_r2_x",
        "frozen_decoder_r2_y",
        "frozen_decoder_rmse_x",
        "frozen_decoder_rmse_y",
        "run_dir",
    ]
    keys = []
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _group_rows(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row.get(col, "") for col in GROUP_COLUMNS)
        groups.setdefault(key, []).append(row)

    out = []
    for key, items in groups.items():
        zeta_emb = [_as_float(item["zeta_emb"]) for item in items]
        zeta_mean = [_as_float(item["zeta_mean"]) for item in items]
        r2 = [_as_float(item["frozen_decoder_r2_mean"]) for item in items]
        rmse = [_as_float(item["frozen_decoder_rmse_mean"]) for item in items]
        group = {col: value for col, value in zip(GROUP_COLUMNS, key)}
        group.update(
            {
                "n": len(items),
                "zeta_emb_mean": float(np.nanmean(zeta_emb)),
                "zeta_emb_ci95": _ci95(zeta_emb),
                "zeta_mean_mean": float(np.nanmean(zeta_mean)),
                "zeta_mean_ci95": _ci95(zeta_mean),
                "frozen_decoder_r2_mean_mean": float(np.nanmean(r2)),
                "frozen_decoder_r2_mean_ci95": _ci95(r2),
                "frozen_decoder_rmse_mean_mean": float(np.nanmean(rmse)),
                "frozen_decoder_rmse_mean_ci95": _ci95(rmse),
            }
        )
        out.append(group)
    return out


def _plot_points(
    rows: list[dict],
    x_key: str,
    y_key: str,
    y_label: str,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for sweep in sorted({row["sweep"] for row in rows}):
        items = [row for row in rows if row["sweep"] == sweep]
        color, marker = SWEEP_STYLE.get(sweep, (None, "o"))
        x = np.asarray([_as_float(row[x_key]) for row in items], dtype=float)
        y = np.asarray([_as_float(row[y_key]) for row in items], dtype=float)
        ax.scatter(x, y, s=38, alpha=0.8, label=sweep, color=color, marker=marker, edgecolors="none")

    x_all = np.asarray([_as_float(row[x_key]) for row in rows], dtype=float)
    y_all = np.asarray([_as_float(row[y_key]) for row in rows], dtype=float)
    ax.set_title(f"{title}\nPearson r={_pearson(x_all, y_all):.3f}, Spearman rho={_spearman(x_all, y_all):.3f}")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_errorbars(
    groups: list[dict],
    x_prefix: str,
    y_prefix: str,
    y_label: str,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for sweep in sorted({row["sweep"] for row in groups}):
        items = [row for row in groups if row["sweep"] == sweep]
        color, marker = SWEEP_STYLE.get(sweep, (None, "o"))
        x = np.asarray([_as_float(row[f"{x_prefix}_mean"]) for row in items], dtype=float)
        y = np.asarray([_as_float(row[f"{y_prefix}_mean"]) for row in items], dtype=float)
        xerr = np.asarray([_as_float(row[f"{x_prefix}_ci95"], 0.0) for row in items], dtype=float)
        yerr = np.asarray([_as_float(row[f"{y_prefix}_ci95"], 0.0) for row in items], dtype=float)
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            ms=6,
            lw=1.1,
            elinewidth=1.0,
            capsize=2.5,
            alpha=0.82,
            color=color,
            label=sweep,
        )

    x_all = np.asarray([_as_float(row[f"{x_prefix}_mean"]) for row in groups], dtype=float)
    y_all = np.asarray([_as_float(row[f"{y_prefix}_mean"]) for row in groups], dtype=float)
    ax.set_title(f"{title}\nGrouped by sweep condition; Pearson r={_pearson(x_all, y_all):.3f}")
    ax.set_xlabel(x_prefix)
    ax.set_ylabel(y_label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_set(
    rows: list[dict],
    groups: list[dict],
    out_dir: Path,
    y_key: str,
    y_prefix: str,
    y_label: str,
    file_label: str,
    title_y: str,
) -> None:
    specs = [
        ("zeta_mean", "zeta_mean", f"Validation mean_val_zeta vs {title_y}"),
        ("zeta_emb", "zeta_emb", f"Validation zeta_emb vs {title_y}"),
    ]
    for x_key, filename_prefix, title in specs:
        _plot_points(
            rows,
            x_key,
            y_key,
            y_label,
            out_dir / f"{filename_prefix}_vs_{file_label}.png",
            title,
        )
        _plot_errorbars(
            groups,
            x_key,
            y_prefix,
            y_label,
            out_dir / f"{filename_prefix}_vs_{file_label}_errorbars.png",
            title,
        )

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    for ax, x_key, label in zip(axes, ["zeta_mean", "zeta_emb"], ["mean_val_zeta", "zeta_emb"]):
        for sweep in sorted({row["sweep"] for row in rows}):
            items = [row for row in rows if row["sweep"] == sweep]
            color, marker = SWEEP_STYLE.get(sweep, (None, "o"))
            x = np.asarray([_as_float(row[x_key]) for row in items], dtype=float)
            y = np.asarray([_as_float(row[y_key]) for row in items], dtype=float)
            ax.scatter(x, y, s=36, alpha=0.8, label=sweep, color=color, marker=marker, edgecolors="none")
        x_all = np.asarray([_as_float(row[x_key]) for row in rows], dtype=float)
        y_all = np.asarray([_as_float(row[y_key]) for row in rows], dtype=float)
        ax.set_title(f"{label}\nr={_pearson(x_all, y_all):.3f}, rho={_spearman(x_all, y_all):.3f}")
        ax.set_xlabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(y_label)
    axes[1].legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"zeta_vs_{file_label}_no_errorbars.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    for ax, x_key, label in zip(axes, ["zeta_mean", "zeta_emb"], ["mean_val_zeta", "zeta_emb"]):
        for sweep in sorted({row["sweep"] for row in groups}):
            items = [row for row in groups if row["sweep"] == sweep]
            color, marker = SWEEP_STYLE.get(sweep, (None, "o"))
            x = np.asarray([_as_float(row[f"{x_key}_mean"]) for row in items], dtype=float)
            y = np.asarray([_as_float(row[f"{y_prefix}_mean"]) for row in items], dtype=float)
            xerr = np.asarray([_as_float(row[f"{x_key}_ci95"], 0.0) for row in items], dtype=float)
            yerr = np.asarray([_as_float(row[f"{y_prefix}_ci95"], 0.0) for row in items], dtype=float)
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, ms=6, lw=1.1, capsize=2.5, alpha=0.82, color=color, label=sweep)
        x_all = np.asarray([_as_float(row[f"{x_key}_mean"]) for row in groups], dtype=float)
        y_all = np.asarray([_as_float(row[f"{y_prefix}_mean"]) for row in groups], dtype=float)
        ax.set_title(f"{label}\ngrouped r={_pearson(x_all, y_all):.3f}")
        ax.set_xlabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(y_label)
    axes[1].legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"zeta_vs_{file_label}_errorbars.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_combined(rows: list[dict], groups: list[dict], out_dir: Path) -> None:
    _plot_metric_set(
        rows,
        groups,
        out_dir,
        y_key="frozen_decoder_r2_mean",
        y_prefix="frozen_decoder_r2_mean",
        y_label="frozen decoder R^2 mean",
        file_label="frozen_decoder_r2",
        title_y="frozen decoder R^2",
    )
    _plot_metric_set(
        rows,
        groups,
        out_dir,
        y_key="frozen_decoder_rmse_mean",
        y_prefix="frozen_decoder_rmse_mean",
        y_label="frozen decoder RMSE mean",
        file_label="frozen_decoder_rmse",
        title_y="frozen decoder RMSE",
    )
    _plot_zeta_metric_relation(rows, groups, out_dir)


def _plot_zeta_metric_relation(rows: list[dict], groups: list[dict], out_dir: Path) -> None:
    rows = [row for row in rows if _as_float(row["zeta_emb"]) >= 0.20]
    groups = [row for row in groups if _as_float(row["zeta_emb_mean"]) >= 0.20]

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    rmse_all = np.asarray([_as_float(row["frozen_decoder_rmse_mean"]) for row in rows], dtype=float)
    finite_rmse = rmse_all[np.isfinite(rmse_all)]
    vmin = float(finite_rmse.min()) if len(finite_rmse) else None
    vmax = float(finite_rmse.max()) if len(finite_rmse) else None
    mappable = None
    for sweep in sorted({row["sweep"] for row in rows}):
        items = [row for row in rows if row["sweep"] == sweep]
        _color, marker = SWEEP_STYLE.get(sweep, (None, "o"))
        x = np.asarray([_as_float(row["zeta_mean"]) for row in items], dtype=float)
        y = np.asarray([_as_float(row["zeta_emb"]) for row in items], dtype=float)
        rmse = np.asarray([_as_float(row["frozen_decoder_rmse_mean"]) for row in items], dtype=float)
        scatter = ax.scatter(
            x,
            y,
            c=rmse,
            cmap="viridis_r",
            vmin=vmin,
            vmax=vmax,
            s=42,
            alpha=0.86,
            label=sweep,
            marker=marker,
            edgecolors="0.2",
            linewidths=0.25,
        )
        mappable = scatter

    x_all = np.asarray([_as_float(row["zeta_mean"]) for row in rows], dtype=float)
    y_all = np.asarray([_as_float(row["zeta_emb"]) for row in rows], dtype=float)
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    if np.any(finite):
        lo = float(min(x_all[finite].min(), y_all[finite].min()))
        hi = float(max(x_all[finite].max(), y_all[finite].max()))
        pad = 0.04 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.35", lw=1.0, ls="--", alpha=0.7)
    ax.set_title(
        "Validation mean_val_zeta vs zeta_emb\n"
        f"Pearson r={_pearson(x_all, y_all):.3f}, Spearman rho={_spearman(x_all, y_all):.3f}"
    )
    ax.set_xlabel("mean_val_zeta")
    ax.set_ylabel("zeta_emb")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=ax, pad=0.02)
        cbar.set_label("frozen decoder RMSE mean")
    ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "zeta_mean_vs_zeta_emb.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for sweep in sorted({row["sweep"] for row in groups}):
        items = [row for row in groups if row["sweep"] == sweep]
        color, marker = SWEEP_STYLE.get(sweep, (None, "o"))
        x = np.asarray([_as_float(row["zeta_mean_mean"]) for row in items], dtype=float)
        y = np.asarray([_as_float(row["zeta_emb_mean"]) for row in items], dtype=float)
        xerr = np.asarray([_as_float(row["zeta_mean_ci95"], 0.0) for row in items], dtype=float)
        yerr = np.asarray([_as_float(row["zeta_emb_ci95"], 0.0) for row in items], dtype=float)
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            ms=6,
            lw=1.1,
            elinewidth=1.0,
            capsize=2.5,
            alpha=0.82,
            color=color,
            label=sweep,
        )

    x_all = np.asarray([_as_float(row["zeta_mean_mean"]) for row in groups], dtype=float)
    y_all = np.asarray([_as_float(row["zeta_emb_mean"]) for row in groups], dtype=float)
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    if np.any(finite):
        lo = float(min(x_all[finite].min(), y_all[finite].min()))
        hi = float(max(x_all[finite].max(), y_all[finite].max()))
        pad = 0.04 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.35", lw=1.0, ls="--", alpha=0.7)
    ax.set_title(
        "Validation mean_val_zeta vs zeta_emb\n"
        f"Grouped by sweep condition; Pearson r={_pearson(x_all, y_all):.3f}"
    )
    ax.set_xlabel("mean_val_zeta")
    ax.set_ylabel("zeta_emb")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "zeta_mean_vs_zeta_emb_errorbars.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _write_summary(rows: list[dict], groups: list[dict], out_path: Path) -> None:
    summary_rows = []
    for scope, data, x_lookup, y_name, y_lookup in [
        ("individual_checkpoints", rows, lambda r, k: _as_float(r[k]), "frozen_decoder_r2_mean", lambda r: _as_float(r["frozen_decoder_r2_mean"])),
        ("grouped_conditions", groups, lambda r, k: _as_float(r[f"{k}_mean"]), "frozen_decoder_r2_mean", lambda r: _as_float(r["frozen_decoder_r2_mean_mean"])),
        ("individual_checkpoints", rows, lambda r, k: _as_float(r[k]), "frozen_decoder_rmse_mean", lambda r: _as_float(r["frozen_decoder_rmse_mean"])),
        ("grouped_conditions", groups, lambda r, k: _as_float(r[f"{k}_mean"]), "frozen_decoder_rmse_mean", lambda r: _as_float(r["frozen_decoder_rmse_mean_mean"])),
    ]:
        y = np.asarray([y_lookup(row) for row in data], dtype=float)
        for metric, metric_label in [("zeta_mean", "mean_val_zeta"), ("zeta_emb", "zeta_emb")]:
            x = np.asarray([x_lookup(row, metric) for row in data], dtype=float)
            summary_rows.append(
                {
                    "scope": scope,
                    "decoder_metric": y_name,
                    "metric": metric_label,
                    "n": int(np.sum(np.isfinite(x) & np.isfinite(y))),
                    "pearson_r": _pearson(x, y),
                    "spearman_rho": _spearman(x, y),
                }
            )
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scope", "decoder_metric", "metric", "n", "pearson_r", "spearman_rho"])
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "mcmaze" / "zeta_decoder_correlation")
    parser.add_argument("--batch-size", type=int, default=0, help="Override checkpoint validation batch size for recomputation.")
    parser.add_argument("--force-recompute", action="store_true", help="Recompute both zeta metrics for every checkpoint.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("sweep_dirs", nargs="*", type=Path, default=DEFAULT_SWEEPS)
    args = parser.parse_args()

    rows = _read_rows([path.expanduser().resolve() for path in args.sweep_dirs])
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    batch_size = args.batch_size if args.batch_size > 0 else None
    print(f"Using device: {device}")
    enriched = _fill_zeta_metrics(rows, device=device, batch_size=batch_size, force=args.force_recompute)
    groups = _group_rows(enriched)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = args.out_dir / "zeta_decoder_correlation_points.csv"
    grouped_csv = args.out_dir / "zeta_decoder_correlation_grouped.csv"
    summary_csv = args.out_dir / "zeta_decoder_correlation_summary.csv"
    _write_csv(enriched, combined_csv)
    _write_csv(groups, grouped_csv)
    _write_summary(enriched, groups, summary_csv)
    _plot_combined(enriched, groups, args.out_dir)

    print(f"Saved point table: {combined_csv}")
    print(f"Saved grouped table: {grouped_csv}")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved plots under: {args.out_dir}")


if __name__ == "__main__":
    main()
