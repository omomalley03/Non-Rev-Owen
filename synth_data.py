"""Data loading helpers for ``main_synth.py`` and synthetic diagnostics."""

from __future__ import annotations

import numpy as np

from config import Config


def _as_knt(windows: np.ndarray, layout: str) -> np.ndarray:
    """Return windows in canonical (K, N, T) layout."""
    if windows.ndim != 3:
        raise ValueError(f"expected a 3D array, got shape {windows.shape}")

    layout = layout.lower()
    if layout == "knt":
        return windows
    if layout == "ktn":
        return np.transpose(windows, (0, 2, 1))
    if layout != "auto":
        raise ValueError("synth_data_layout must be one of: auto, knt, ktn")

    # Real EEG arrays are typically channels x time, e.g. FACED is (K, 32, 2000).
    # The older generated rotations are time x channels. Use the smaller middle
    # axis as channels when the two axes are clearly separated.
    if windows.shape[1] <= windows.shape[2]:
        return windows
    return np.transpose(windows, (0, 2, 1))


def _normalize_channels(windows: np.ndarray, method: str) -> np.ndarray:
    method = method.lower()
    if method in {"none", ""}:
        return windows
    if method != "zscore":
        raise ValueError("synth_normalize must be one of: none, zscore")

    mean = windows.mean(axis=(0, 2), keepdims=True)
    std = windows.std(axis=(0, 2), keepdims=True)
    return (windows - mean) / (std + 1e-6)


def load_synthetic_windows(cfg: Config, data_path: str | None = None) -> np.ndarray:
    """Load an ``.npy`` dataset as float32 windows with shape (K, N, T)."""
    path = data_path or cfg.synth_data_path
    windows = np.load(path)
    windows = _as_knt(windows, getattr(cfg, "synth_data_layout", "auto"))
    windows = np.ascontiguousarray(windows, dtype=np.float32)
    windows = _normalize_channels(windows, getattr(cfg, "synth_normalize", "none"))

    max_trials = getattr(cfg, "synth_max_trials", 0)
    if max_trials and max_trials > 0:
        windows = windows[:max_trials]

    noise_std = getattr(cfg, "synth_noise_std", 0.0)
    if noise_std > 0:
        rng = np.random.default_rng(cfg.seed)
        noise = rng.normal(0.0, noise_std, size=windows.shape).astype(np.float32)
        windows = windows + noise

    return np.ascontiguousarray(windows, dtype=np.float32)
