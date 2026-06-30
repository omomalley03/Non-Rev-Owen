# FACED settings for main_synth.py.
#
# Usage:
#   source faced_setup.sh
#   /home/omo26/micromamba/envs/nonrev/bin/python -u main_synth.py
#
# Or, from an activated nonrev environment:
#   source faced_setup.sh
#   python -u main_synth.py
#
# To override a value for a quick run, source first and then export it, e.g.:
#   source faced_setup.sh
#   export EPOCHS=1 SYNTH_MAX_TRIALS=2048

export RUNS_BASE="/home/omo26/Non-Rev-Owen"
export DATASET_NAME="faced"
export CACHE_DIR="$RUNS_BASE/cache"
export MPLCONFIGDIR="/tmp/matplotlib_nonrev"

# FACED source.
export FACED_PROCESSED_DIR="/home/omo26/data_owen/FACED/processed"
export FACED_NPY="$FACED_PROCESSED_DIR/faced_data.npy"
export FACED_LABELS_NPY="$FACED_PROCESSED_DIR/faced_labels.npy"
export FACED_IDS_NPY="$FACED_PROCESSED_DIR/faced_ids.npy"

# main_synth.py data settings.
export SYNTH_DATA_PATH="$FACED_NPY"
export SYNTH_DATA_LAYOUT="auto"
export SYNTH_NORMALIZE="zscore"
export SYNTH_PREPROCESS="car,analytic_bandpass"
export EEG_FS="250"
export EEG_BANDS="theta:4-8,alpha:8-13,beta:13-30,gamma:30-45"
export SYNTH_NOISE_STD="0"
export SYNTH_MAX_TRIALS="2048"
export SYNTH_VIZ_MAX_TRIALS="64"
export SYNTH_VIZ_MAX_TIMEPOINTS="400"

# Stage 1: deliberately use all windows for both train and validation. This is
# for checking whether the model/objective can fit any FACED structure at all.
export SYNTH_SPLIT="train_eq_val"
export VAL_SPLIT="0.1"

# Model settings. These are the best FACED settings found in the 2026-06-30
# sweep so far. Four-band zero-phase analytic bandpass preprocessing adds a
# Hilbert quadrature channel, similar to complex wavelet coefficients, without
# adding absolute time features. The 2048-trial cap keeps the verified run
# within memory while the full dataset path remains available by override.
export D="2"
export HIDDEN_DIM="128"
export DEPTH="1"
export DROPOUT="0.2"
export TEMPORAL_FILTERS="0"
export TEMPORAL_KERNEL_SIZE="61"

# Training settings.
export BATCH_SIZE="128"
export EPOCHS="8"
export LR="1e-4"
export WEIGHT_DECAY="1e-4"
export LAMBDA_XP="0.0"
export LAMBDA_BT="0.0"
export LAMBDA_PLANE_BT="0.0"
export LAMBDA_BLOCK_CCA="0.0"
export LAMBDA_START_FRAC="1.0"
export S_OBJECTIVE="mean"
export S_SOFTMIN_TAU="0.05"
export T_0="10"
export T_MULT="2"

mkdir -p "$RUNS_BASE/$DATASET_NAME/runs" "$RUNS_BASE/$DATASET_NAME/synth_runs" "$CACHE_DIR" "$MPLCONFIGDIR"
