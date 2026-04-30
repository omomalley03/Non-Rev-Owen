import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
from scipy.ndimage import gaussian_filter1d
from nlb_tools.nwb_interface import NWBDataset


def load_mcmaze(nwb_path: str, bin_ms: int = 5):
    """Load MC_Maze train NWB, resample to bin_ms, return spike array and trial info.

    Returns
    -------
    spikes_raw : np.ndarray, shape (N, T_total), float32
        Raw spike counts. NaN entries (inter-trial gaps in NLB format) are filled with 0.
    bin_width_s : float
        Bin duration in seconds.
    trial_info : pd.DataFrame
        Trial table with start_time and end_time columns.
    """
    ds = NWBDataset(nwb_path)
    ds.resample(bin_ms)

    spikes_df = ds.data["spikes"]           # (T_total, N)
    spikes_raw = spikes_df.values.T.astype(np.float32)  # (N, T_total)
    spikes_raw = np.nan_to_num(spikes_raw, nan=0.0)

    bin_width_s = bin_ms * 1e-3
    trial_info = ds.trial_info
    # Normalise time columns to seconds (stored as timedelta64[ns] in NLB datasets)
    for col in ["start_time", "end_time", "stop_time"]:
        if col in trial_info.columns:
            delta = trial_info[col]
            if hasattr(delta, "dt"):
                trial_info = trial_info.copy()
                trial_info[col] = delta.dt.total_seconds().values
            elif np.issubdtype(delta.dtype, np.timedelta64):
                trial_info = trial_info.copy()
                trial_info[col] = delta.values.astype("float64") / 1e9

    # Store the global time index (in seconds) so we can map trial times → bin indices
    time_index_s = spikes_df.index.values
    if np.issubdtype(time_index_s.dtype, np.timedelta64):
        time_index_s = time_index_s.astype("float64") / 1e9
    else:
        time_index_s = time_index_s.astype("float64")

    return spikes_raw, bin_width_s, trial_info, time_index_s


def gaussian_smooth(X: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Apply Gaussian smoothing independently along the time axis (last axis).

    Parameters
    ----------
    X : (N, T) float array
    sigma_samples : smoothing std in samples
    """
    return gaussian_filter1d(X, sigma=sigma_samples, axis=-1)


def make_windows(
    X_smooth: np.ndarray,
    trial_info,
    time_index_s: np.ndarray,
    bin_width_s: float,
    strategy: str = "trial_aligned",
    window_size: int = 120,
) -> np.ndarray:
    """Segment smoothed spike data into windows of shape (K, N, T).

    strategy='trial_aligned'
        One window per trial; each trial must be exactly window_size bins.
        Trials shorter/longer than window_size are truncated/zero-padded.
    """
    N = X_smooth.shape[0]
    stop_col = "stop_time" if "stop_time" in trial_info.columns else "end_time"

    windows = []
    for _, row in trial_info.iterrows():
        t_start = float(row["start_time"])
        t_end = float(row[stop_col])

        # Map seconds → bin index via the global time index
        idx_start = int(np.searchsorted(time_index_s, t_start))
        idx_end = idx_start + window_size          # all trials are exactly window_size bins

        chunk = X_smooth[:, idx_start:idx_end]    # (N, actual_len)
        actual_len = chunk.shape[1]

        if actual_len == window_size:
            windows.append(chunk)
        elif actual_len < window_size:
            # Zero-pad on the right if trial is shorter than expected
            pad = np.zeros((N, window_size - actual_len), dtype=np.float32)
            windows.append(np.concatenate([chunk, pad], axis=1))
        else:
            # Truncate to window_size
            windows.append(chunk[:, :window_size])

    return np.stack(windows, axis=0).astype(np.float32)   # (K, N, T)


def train_val_split(windows: np.ndarray, val_frac: float = 0.1, seed: int = 0):
    """Split window array into train/val TensorDatasets.

    Returns
    -------
    train_ds, val_ds : TensorDataset of (N, T) float32 tensors
    """
    tensor = torch.from_numpy(windows)          # (K, N, T)
    K = len(tensor)
    n_val = max(1, int(K * val_frac))
    n_train = K - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        TensorDataset(tensor), [n_train, n_val], generator=generator
    )
    return train_ds, val_ds
