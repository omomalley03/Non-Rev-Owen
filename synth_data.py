"""Data loading helpers for ``main_synth.py`` and synthetic diagnostics."""

from __future__ import annotations

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
    if method not in {"zscore", "robust_zscore", "robust"}:
        raise ValueError("synth_normalize must be one of: none, zscore, robust_zscore")

    if method in {"robust_zscore", "robust"}:
        median = np.median(windows, axis=(0, 2), keepdims=True)
        mad = np.median(np.abs(windows - median), axis=(0, 2), keepdims=True)
        return (windows - median) / (1.4826 * mad + 1e-6)

    mean = windows.mean(axis=(0, 2), keepdims=True)
    std = windows.std(axis=(0, 2), keepdims=True)
    return (windows - mean) / (std + 1e-6)


def _parse_steps(spec: str) -> list[str]:
    if spec is None:
        return []
    spec = spec.strip().lower()
    if spec in {"", "none"}:
        return []
    return [s.strip() for s in spec.split(",") if s.strip()]


def _parse_bands(spec: str) -> list[tuple[str, float, float]]:
    bands = []
    for item in spec.split(","):
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
        raise ValueError("EEG_BANDS must contain at least one band, e.g. theta:4-8")
    return bands


def _bandpass_zero_phase(windows: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    nyquist = fs / 2.0
    if not (0.0 < lo < hi < nyquist):
        raise ValueError(f"invalid EEG band {lo:g}-{hi:g} Hz for fs={fs:g} Hz")
    sos = signal.butter(4, [lo / nyquist, hi / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, windows, axis=-1).astype(np.float32, copy=False)


def _apply_eeg_preprocess(windows: np.ndarray, cfg: Config) -> np.ndarray:
    """Apply time-equivariant EEG preprocessing to (K, N, T) windows."""
    steps = _parse_steps(getattr(cfg, "synth_preprocess", "none"))
    if not steps:
        return windows

    X = np.ascontiguousarray(windows, dtype=np.float32)
    fs = float(getattr(cfg, "eeg_fs", 250.0))
    bands = None

    for step in steps:
        if step == "car":
            X = X - X.mean(axis=1, keepdims=True)
        elif step in {"temporal_demean", "demean"}:
            X = X - X.mean(axis=-1, keepdims=True)
        elif step == "trial_zscore":
            X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-6)
        elif step == "diff":
            X = np.diff(X, axis=-1, prepend=X[..., :1]).astype(np.float32, copy=False)
        elif step in {"downsample2", "temporal_downsample2"}:
            X = X[..., ::2]
        elif step in {
            "bandpass", "bandpass_concat",
            "bandpower", "bandpower_concat",
            "analytic_bandpass", "analytic_bandpass_concat",
        }:
            if bands is None:
                bands = _parse_bands(getattr(cfg, "eeg_bands", "theta:4-8,alpha:8-13,beta:13-30,gamma:30-45"))
            raw = X if step.endswith("_concat") else None
            K, N, T = X.shape
            channels_per_band = 2 * N if step.startswith("analytic_bandpass") else N
            out_offset = N if raw is not None else 0
            out = np.empty((K, out_offset + channels_per_band * len(bands), T), dtype=np.float32)
            if raw is not None:
                out[:, :N, :] = raw
            for band_idx, (_, lo, hi) in enumerate(bands):
                filtered = _bandpass_zero_phase(X, fs=fs, lo=lo, hi=hi)
                start = out_offset + band_idx * channels_per_band
                if step.startswith("analytic_bandpass"):
                    analytic = signal.hilbert(filtered, axis=-1)
                    out[:, start:start + N, :] = filtered.astype(np.float32, copy=False)
                    out[:, start + N:start + 2 * N, :] = analytic.imag.astype(np.float32, copy=False)
                elif step in {"bandpower", "bandpower_concat"}:
                    analytic = signal.hilbert(filtered, axis=-1)
                    filtered = np.log1p(np.abs(analytic).astype(np.float32) ** 2)
                    out[:, start:start + N, :] = filtered.astype(np.float32, copy=False)
                else:
                    out[:, start:start + N, :] = filtered.astype(np.float32, copy=False)
            X = out
        else:
            raise ValueError(
                "unknown SYNTH_PREPROCESS step "
                f"{step!r}; expected comma-separated values from car, temporal_demean, "
                "trial_zscore, diff, downsample2, bandpass, bandpass_concat, bandpower, "
                "bandpower_concat, analytic_bandpass, analytic_bandpass_concat"
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
