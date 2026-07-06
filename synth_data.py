"""Data loading helpers for ``main_synth.py`` and synthetic diagnostics."""

from __future__ import annotations

import os

import numpy as np
from scipy import signal

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
    # 


def _parse_steps(spec: str) -> list[str]:
    if spec is None:
        return []
    spec = str(spec).strip().lower()
    if spec in {"", "none"}:
        return []
    return [s.strip() for s in spec.split(",") if s.strip()]


def _parse_bands(spec: str) -> list[tuple[str, float, float]]:
    bands = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, limits = item.split(":", 1)
        else:
            name, limits = item, item
        lo, hi = limits.split("-", 1)
        bands.append((name.strip(), float(lo), float(hi)))
    if not bands:
        raise ValueError("EEG_BANDS must contain at least one band, e.g. mu:8-12")
    return bands


def _bandpass_zero_phase(windows: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    nyquist = fs / 2.0
    if not (0.0 < lo < hi < nyquist):
        raise ValueError(f"invalid EEG band {lo:g}-{hi:g} Hz for fs={fs:g} Hz")
    sos = signal.butter(4, [lo / nyquist, hi / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, windows, axis=-1).astype(np.float32, copy=False)


def _apply_eeg_preprocess(windows: np.ndarray, cfg: Config) -> np.ndarray:
    """Apply the small EEG preprocessing set needed by the best analytic runs."""
    steps = _parse_steps(getattr(cfg, "synth_preprocess", "none"))
    if not steps:
        return windows

    X = np.ascontiguousarray(windows, dtype=np.float32)
    fs = float(getattr(cfg, "eeg_fs", 250.0))
    bands = None

    for step in steps:
        if step == "car":
            X = X - X.mean(axis=1, keepdims=True)
        elif step == "analytic_bandpass":
            if bands is None:
                bands = _parse_bands(getattr(cfg, "eeg_bands", "theta:4-8,alpha:8-13,beta:13-30,gamma:30-45"))
            K, N, T = X.shape
            out = np.empty((K, 2 * N * len(bands), T), dtype=np.float32)
            for band_idx, (_, lo, hi) in enumerate(bands):
                filtered = _bandpass_zero_phase(X, fs=fs, lo=lo, hi=hi)
                analytic = signal.hilbert(filtered, axis=-1)
                start = band_idx * 2 * N
                out[:, start:start + N, :] = filtered.astype(np.float32, copy=False)
                out[:, start + N:start + 2 * N, :] = analytic.imag.astype(np.float32, copy=False)
            X = out
        else:
            raise ValueError(
                f"unknown SYNTH_PREPROCESS step {step!r}; expected comma-separated values from none, car, analytic_bandpass"
            )
        X = np.ascontiguousarray(X, dtype=np.float32)
    return X


def load_synthetic_windows(cfg: Config, data_path: str | None = None) -> np.ndarray:
    """Load an ``.npy`` dataset as float32 windows with shape (K, N, T)."""
    path = data_path or cfg.synth_data_path
    windows = np.load(path)
    windows = _as_knt(windows, getattr(cfg, "synth_data_layout", "auto"))
    windows = np.ascontiguousarray(windows, dtype=np.float32)

    max_trials = getattr(cfg, "synth_max_trials", 0)
    if max_trials and max_trials > 0:
        windows = windows[:max_trials]

    windows = _apply_eeg_preprocess(windows, cfg)
    windows = _normalize_channels(windows, getattr(cfg, "synth_normalize", "none"))

    noise_std = getattr(cfg, "synth_noise_std", 0.0)
    if noise_std > 0:
        rng = np.random.default_rng(cfg.seed)
        noise = rng.normal(0.0, noise_std, size=windows.shape).astype(np.float32)
        windows = windows + noise

    return np.ascontiguousarray(windows, dtype=np.float32)


def load_synthetic_subjects(cfg: Config) -> np.ndarray | None:
    """Load optional per-window subject IDs aligned with ``load_synthetic_windows``."""
    path = getattr(cfg, "synth_subjects_path", "")
    if not path:
        return None

    subjects = np.load(path)
    if subjects.ndim != 1:
        raise ValueError(f"expected 1D subject IDs, got shape {subjects.shape}")

    max_trials = getattr(cfg, "synth_max_trials", 0)
    if max_trials and max_trials > 0:
        subjects = subjects[:max_trials]
    return np.ascontiguousarray(subjects)


def _infer_labels_path(data_path: str) -> str:
    if not data_path.endswith(".npy"):
        return ""
    candidate = data_path[:-4] + "_labels.npy"
    return candidate if os.path.exists(candidate) else ""


def load_synthetic_labels(cfg: Config, data_path: str | None = None) -> np.ndarray | None:
    """Load optional per-window class labels aligned with ``load_synthetic_windows``."""
    path = getattr(cfg, "synth_labels_path", "")
    if not path:
        path = _infer_labels_path(data_path or getattr(cfg, "synth_data_path", ""))
    if not path:
        return None

    labels = np.load(path)
    if labels.ndim != 1:
        raise ValueError(f"expected 1D labels, got shape {labels.shape}")

    max_trials = getattr(cfg, "synth_max_trials", 0)
    if max_trials and max_trials > 0:
        labels = labels[:max_trials]
    return np.ascontiguousarray(labels)
