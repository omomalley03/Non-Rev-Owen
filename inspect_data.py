"""
One-shot dataset inspection for MC_Maze (DANDI:000129).
Run this before writing data.py to determine C, bin_width, trial lengths,
and an appropriate windowing strategy.
"""
import numpy as np
from nlb_tools.nwb_interface import NWBDataset

NWB_PATH = "/Users/omomalley03/Documents/Dissertation/Data/000129/sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb"

ds = NWBDataset(NWB_PATH)

# --- bin width ---
# NWBDataset stores data in its native resolution; bin by 5 ms (standard for NLB)
BIN_MS = 5
ds.resample(BIN_MS)

# --- spike counts ---
spikes = ds.data["spikes"]          # DataFrame: rows = time bins, cols = units
C = spikes.shape[1]
T_total = spikes.shape[0]
bin_width_s = BIN_MS * 1e-3

print("=" * 60)
print(f"Channels (units):    {C}")
print(f"Bin width:           {BIN_MS} ms  ({bin_width_s} s)")
print(f"Total time bins:     {T_total}  ({T_total * BIN_MS / 1000:.1f} s)")
arr = spikes.values
print(f"NaN bins:            {np.isnan(arr).sum()} of {arr.size}")
arr_valid = arr[~np.isnan(arr)]
print(f"Spike count range:   {int(arr_valid.min())} – {int(arr_valid.max())}")
print(f"Mean spikes/bin:     {arr_valid.mean():.4f}")
print()

# --- trial structure ---
trials = ds.trial_info
print(f"Number of trials:    {len(trials)}")
print(f"Trial columns:       {list(trials.columns)}")
print()

# Compute trial lengths in bins
stop_col = "stop_time" if "stop_time" in trials.columns else ("end_time" if "end_time" in trials.columns else None)
if "start_time" in trials.columns and stop_col:
    delta = trials[stop_col] - trials["start_time"]
    # Convert to float seconds regardless of whether stored as timedelta or float
    if hasattr(delta, "dt"):
        trial_len_s = delta.dt.total_seconds().values
    else:
        trial_len_s = delta.values.astype("float64") / 1e9  # nanoseconds → seconds
    trial_len_bins = (trial_len_s / bin_width_s).astype(int)
    print(f"Trial length (bins): min={trial_len_bins.min()}, "
          f"max={trial_len_bins.max()}, "
          f"median={int(np.median(trial_len_bins))}, "
          f"mean={trial_len_bins.mean():.1f}")
else:
    trial_len_bins = None
    print("Could not find start_time/end_time columns for trial lengths.")

print()

# --- move_onset offset from trial start ---
onset_col = None
for col in ["move_onset_time", "go_cue_time", "move_onset", "move_onset_time"]:
    if col in trials.columns:
        onset_col = col
        break

if onset_col and "start_time" in trials.columns:
    onset_offset_s = (trials[onset_col] - trials["start_time"]).values
    onset_offset_bins = (onset_offset_s / bin_width_s).astype(int)
    print(f"Onset column used:   '{onset_col}'")
    print(f"Onset offset (bins): min={onset_offset_bins.min()}, "
          f"max={onset_offset_bins.max()}, "
          f"median={int(np.median(onset_offset_bins))}")
    pre_bins  = int(np.median(onset_offset_bins))
    post_bins = int(np.median(trial_len_bins - onset_offset_bins)) if trial_len_bins is not None else 90
    suggested_T = pre_bins + post_bins
    print(f"\nSuggested epoch: {pre_bins} bins pre-onset + {post_bins} bins post-onset")
    print(f"  => window_size T = {suggested_T} bins  ({suggested_T * BIN_MS} ms)")
elif trial_len_bins is not None:
    suggested_T = int(np.median(trial_len_bins))
    print(f"No onset column found; using full trial as epoch.")
    print(f"  => window_size T = {suggested_T} bins  ({suggested_T * BIN_MS} ms)")
else:
    suggested_T = 140
    print(f"No onset or trial length info; defaulting to T=140 (700 ms)")

print()

# --- windowing recommendation ---
if trial_len_bins is not None:
    cv = trial_len_bins.std() / trial_len_bins.mean()
    print(f"Trial length CV (std/mean): {cv:.3f}")
    if cv > 0.1:
        print("=> Trials are variable length  →  use TRIAL-ALIGNED epochs (strategy='trial_aligned')")
    else:
        print("=> Trials are roughly uniform  →  SLIDING WINDOWS also viable (strategy='sliding')")

print()
print("=" * 60)
print(f"Summary for config.py:")
print(f"  C              = {C}")
print(f"  bin_width_ms   = {BIN_MS}")
print(f"  window_size    = {suggested_T}  # bins")
print(f"  stride         = {max(1, suggested_T // 2)}  # 50% overlap (if using sliding windows)")
print(f"  window_strategy = 'trial_aligned'  # recommended")
print("=" * 60)
