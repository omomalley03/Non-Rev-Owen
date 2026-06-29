import os
import pickle
import warnings

import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset, random_split
from scipy.ndimage import gaussian_filter1d
from nlb_tools.nwb_interface import NWBDataset

# Where NWB→pickle caches live. Override with `export CACHE_DIR=...` (e.g. on the
# HPC) without editing source; defaults to a `cache/` dir next to this file.
_CACHE_DIR = os.environ.get(
    "CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"),
)


# NLB time-related fields stored as timedelta64[ns]; convert to seconds
_TIME_COLS = [
    "start_time", "end_time", "stop_time",
    "target_on_time", "go_cue_time", "move_onset_time",
]


def load_mcmaze(nwb_path: str, bin_ms: int = 5):
    """Load NLB MC_Maze NWB, resample to bin_ms, return spikes + behaviour + trial info.

    Returns
    -------
    spikes_raw : np.ndarray, shape (N, T_total), float32
        Raw spike counts. NaN entries (inter-trial gaps) are filled with 0.
    bin_width_s : float
        Bin duration in seconds.
    trial_info : pd.DataFrame
        Trial table with all NWB fields converted to seconds, plus a
        `reach_angle` column (radians ∈ [-π, π]) computed as the angle of
        the active target's position when available.
    time_index_s : np.ndarray
        Global time index in seconds for mapping trial times → bin indices.
    hand_pos_raw : np.ndarray of shape (2, T_total), float32 — or None
        Resampled hand_pos x/y aligned with `time_index_s`. NaNs filled with 0.
        None if `hand_pos` isn't in the NWB (e.g. spikes-only file).
    """
    ds = NWBDataset(nwb_path)

    # nlb_tools' resample() succeeds at downsampling but raises a pandas
    # freq-set error on data with inter-trial gaps. Suppress and finish the
    # bookkeeping it skipped.
    try:
        ds.resample(bin_ms)
    except ValueError:
        pass
    ds.bin_width = bin_ms

    spikes_df = ds.data["spikes"]                          # (T_total, N)
    spikes_raw = spikes_df.values.T.astype(np.float32)     # (N, T_total)
    spikes_raw = np.nan_to_num(spikes_raw, nan=0.0)

    bin_width_s = bin_ms * 1e-3
    trial_info = ds.trial_info.copy()

    for col in _TIME_COLS:
        if col in trial_info.columns:
            delta = trial_info[col]
            if hasattr(delta, "dt"):
                trial_info[col] = delta.dt.total_seconds().values
            elif np.issubdtype(delta.dtype, np.timedelta64):
                trial_info[col] = delta.values.astype("float64") / 1e9

    time_index_s = spikes_df.index.values
    if np.issubdtype(time_index_s.dtype, np.timedelta64):
        time_index_s = time_index_s.astype("float64") / 1e9
    else:
        time_index_s = time_index_s.astype("float64")

    # Reach angle for visualisation. Two layouts in the wild:
    #   NLB MC_Maze (Jenkins): `target_pos` is a per-row (n_targets, 2) array
    #     and `active_target` is the index of the goal.
    #   Indy/Sabes:            `target_pos` and `cursor_pos` are continuous
    #     signals in ds.data; angle = arctan2(target − cursor at start).
    if "target_pos" in trial_info.columns and "active_target" in trial_info.columns:
        angles = np.empty(len(trial_info), dtype=np.float64)
        for k, (_, row) in enumerate(trial_info.iterrows()):
            tgt_pos = np.asarray(row["target_pos"])    # (n_targets, 2)
            act_idx = int(row["active_target"])
            x, y = tgt_pos[act_idx]
            angles[k] = np.arctan2(y, x)
        trial_info["reach_angle"] = angles
    elif {"target_pos", "cursor_pos"}.issubset(ds.data.columns.get_level_values(0)):
        target_xy = ds.data["target_pos"][["x", "y"]].values
        cursor_xy = ds.data["cursor_pos"][["x", "y"]].values
        starts = trial_info["start_time"].values.astype("float64")
        idx = np.searchsorted(time_index_s, starts).clip(0, len(time_index_s) - 1)
        delta = target_xy[idx] - cursor_xy[idx]
        trial_info["reach_angle"] = np.arctan2(delta[:, 1], delta[:, 0])

    if "hand_pos" in ds.data.columns.get_level_values(0):
        hand_xy = ds.data["hand_pos"][["x", "y"]].values
        hand_xy = np.nan_to_num(hand_xy, nan=0.0)
        hand_pos_raw = hand_xy.T.astype(np.float32)
    else:
        hand_pos_raw = None

    return spikes_raw, bin_width_s, trial_info, time_index_s, hand_pos_raw


def load_mcmaze_cached(nwb_path: str, bin_ms: int = 5):
    """Load MC_Maze data from a pickle cache if available, else fall back to NWB.

    The cache is created by running ``python cache_data.py``. Cache files live
    in ``cache/mcmaze_bin{bin_ms}ms.pkl`` next to this source file.
    """
    cache_file = os.path.join(_CACHE_DIR, f"mcmaze_bin{bin_ms}ms.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            d = pickle.load(f)
        return (d["spikes_raw"], d["bin_width_s"], d["trial_info"],
                d["time_index_s"], d["hand_pos_raw"])

    warnings.warn(
        f"No cache found at {cache_file}. Falling back to NWB "
        f"(run `python cache_data.py` to speed up future loads).",
        stacklevel=2,
    )
    return load_mcmaze(nwb_path, bin_ms)


def gaussian_smooth(X: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Apply Gaussian smoothing independently along the time axis (last axis)."""
    return gaussian_filter1d(X, sigma=sigma_samples, axis=-1)


def soft_normalize(X: np.ndarray, method: str = "churchland") -> np.ndarray:
    """Per-neuron soft normalisation

    X has shape (N, T_total). Returns the same shape with each neuron rescaled
    so their dynamic ranges are comparable

    method:
      'churchland' — divide by (range + 5)
      'max'        — divide by max(max, 0.1)
      None or 'none' — no normalisation (returns X unchanged)
    """
    if method is None or method == "none":
        return X
    if method == "churchland":
        rng = X.max(axis=1) - X.min(axis=1)
        return X / (rng[:, None] + 5.0)
    if method == "max":
        m = np.maximum(X.max(axis=1), 0.1)
        return X / m[:, None]
    raise ValueError(f"unknown softnorm method: {method!r}")


def make_windows(
    X_smooth: np.ndarray,
    trial_info,
    time_index_s: np.ndarray,
    bin_width_s: float,
    strategy: str = "trial_aligned",
    window_size: int = 120,
    align_field: str = "move_onset_time",
    pre_ms: int = 100,
) -> np.ndarray:
    """Segment smoothed spikes into windows of shape (K, N, T) aligned to a per-trial event.

    For each trial, the window starts pre_ms before align_field and is
    window_size bins long. If align_field is missing the trial's start_time
    is used and the entire window is taken from there.
    """
    N = X_smooth.shape[0]
    pre_bins = int(round(pre_ms * 1e-3 / bin_width_s))

    if align_field in trial_info.columns:
        align_times = trial_info[align_field].astype(float).values
    else:
        align_times = trial_info["start_time"].astype(float).values
        pre_bins = 0

    T_total = X_smooth.shape[1]
    windows = []
    for t_align in align_times:
        idx_align = int(np.searchsorted(time_index_s, t_align))
        idx_start = max(0, idx_align - pre_bins)
        idx_end = idx_start + window_size
        chunk = X_smooth[:, idx_start:min(idx_end, T_total)]
        actual_len = chunk.shape[1]

        if actual_len == window_size:
            windows.append(chunk)
        elif actual_len < window_size:
            pad = np.zeros((N, window_size - actual_len), dtype=np.float32)
            windows.append(np.concatenate([chunk, pad], axis=1))
        else:
            windows.append(chunk[:, :window_size])

    return np.stack(windows, axis=0).astype(np.float32)   # (K, N, T)


def train_val_split(
    windows: np.ndarray,
    trial_info=None,
    val_frac: float = 0.1,
    seed: int = 0,
):
    """Split (K, N, T) windows into train/val TensorDatasets.

    If `trial_info` is provided and has a `split` column ('train' / 'val'),
    that split is used directly (NLB datasets supply this).  Otherwise falls
    back to a seeded random split with `val_frac`.

    Returns
    -------
    train_ds, val_ds : torch.utils.data.Subset of a TensorDataset((K, N, T))
        Both have an `.indices` attribute giving original positions in `windows`,
        which align row-for-row with `trial_info.iloc[...]`.
    """
    tensor = torch.from_numpy(windows)
    full_ds = TensorDataset(tensor)

    if trial_info is not None and "split" in trial_info.columns:
        split = trial_info["split"].to_numpy()
        train_pos = np.where(split == "train")[0].tolist()
        val_pos = np.where(split == "val")[0].tolist()
        return Subset(full_ds, train_pos), Subset(full_ds, val_pos)

    K = len(tensor)
    n_val = max(1, int(K * val_frac))
    n_train = K - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(full_ds, [n_train, n_val], generator=generator)
