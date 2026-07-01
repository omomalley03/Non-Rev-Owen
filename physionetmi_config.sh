# PhysioNetMI settings for main_synth.py.
#
# Usage:
#   source physionetmi_config.sh
#   /home/omo26/micromamba/envs/nonrev/bin/python -u main_synth.py
#
# To override a value for a quick run, source first and then export it, e.g.:
#   source physionetmi_config.sh
#   export EPOCHS=1 SYNTH_VIZ_MAX_TRIALS=16

export RUNS_BASE="/home/omo26/Non-Rev-Owen"
export DATASET_NAME="physionetmi"
export CACHE_DIR="$RUNS_BASE/cache"
export MPLCONFIGDIR="/tmp/matplotlib_nonrev"

# LMDB source and converted .npy outputs.
export PHYSIONETMI_PROCESSED_DIR="/home/omo26/data_owen/PhysioNetMI/processed"
export PHYSIONETMI_NPY="$CACHE_DIR/physionetmi_train_val.npy"
export PHYSIONETMI_LABELS_NPY="$CACHE_DIR/physionetmi_train_val_labels.npy"
export PHYSIONETMI_SUBJECTS_NPY="$CACHE_DIR/physionetmi_train_val_subjects.npy"

# main_synth.py data settings.
export SYNTH_DATA_PATH="$PHYSIONETMI_NPY"
export SYNTH_DATA_LAYOUT="knt"
export SYNTH_NORMALIZE="zscore"
export SYNTH_PREPROCESS="none"
export EEG_FS="160"
export EEG_BANDS="mu:8-12,beta:13-30"
export SYNTH_NOISE_STD="0"
export SYNTH_MAX_TRIALS="2048"
export SYNTH_SPLIT="train_eq_val"
export SYNTH_VIZ_MAX_TRIALS="64"
export SYNTH_VIZ_MAX_TIMEPOINTS="200"
export VAL_SPLIT="0.1"

# Model settings.
# Best residual-branch architecture from the 2026-07-01 sweep. This uses the
# CoCoT-style per-channel multi-kernel ResidualBranch front-end ported into
# model.py; wider 8/16-filter variants raised raw S but damaged the 2D geometry.
export D="2"
export HIDDEN_DIM="128"
export DEPTH="1"
export DROPOUT="0.2"
export TEMPORAL_FRONTEND="residual"
export TEMPORAL_FILTERS="4"
export RESIDUAL_KERNELS="7,15,31,61"
export TEMPORAL_KERNEL_SIZE="61"

# Training settings.
export BATCH_SIZE="128"
export EPOCHS="50"
export LR="5e-4"
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
