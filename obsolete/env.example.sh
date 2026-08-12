# Copy this to a machine-specific file such as env.mac.sh, env.gpu.sh, or
# env.hpc.sh. Those files are ignored by git.

# Base directory for data, cache, and run outputs on this machine.
export RUNS_BASE="$HOME/POC_MLP"

# Optional explicit overrides. If omitted, these default under RUNS_BASE.
export CACHE_DIR="$RUNS_BASE/cache"
export SYNTH_DATA_PATH="$RUNS_BASE/rotations_mixed_freqs.npy"
# export NWB_PATH="$RUNS_BASE/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

mkdir -p "$RUNS_BASE/runs" "$RUNS_BASE/synth_runs" "$CACHE_DIR"
