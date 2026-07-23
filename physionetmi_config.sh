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
export SYNTH_LABELS_PATH="$PHYSIONETMI_LABELS_NPY"
export SYNTH_DATA_LAYOUT="knt"
export SYNTH_NORMALIZE="zscore"
export SYNTH_PREPROCESS="none"
export SYNTH_NOISE_STD="0"

# Subject-aware train/val split:
#   1. Use all cached PhysioNet trials before splitting.
#   2. Randomly sample SYNTH_SUBJECT_COUNT participants with SEED.
#   3. Randomly split only those participants' trials into 90% train / 10% val.
#
# The cache has 89 participants. Most have 90 trials; subject 88 has 114.
# A 23-participant subset gives 2094 trials for SEED=0, close to the old
# 2048-trial experiments, but without train=val leakage.
export SEED="0"
export SYNTH_MAX_TRIALS="10000"
export SYNTH_SPLIT="subject_random"
export SYNTH_SUBJECTS_PATH="$PHYSIONETMI_SUBJECTS_NPY"
export SYNTH_SUBJECT_COUNT="89"
export SYNTH_SUBJECT_IDS=""   # optional explicit comma-separated IDs; overrides count
export SYNTH_HOLDOUT_SUBJECT_COUNT="5"  # excluded from embedding train/val and used for decoder test
export SYNTH_HOLDOUT_SUBJECT_IDS=""     # optional explicit comma-separated held-out IDs
export SYNTH_VIZ_MAX_TRIALS="1"
export SYNTH_VIZ_MAX_TIMEPOINTS="1000"
export SYNTH_VIZ_PARTICIPANT_MODE="top_zeta"  # top_zeta or random
export SYNTH_VIZ_PARTICIPANT_COUNT="8"
export VAL_SPLIT="0.2"

# Model settings.
export D="16"
export HIDDEN_DIM="256"
export DEPTH="2"
export DROPOUT="0.2"
export TEMPORAL_FRONTEND="multiscale_symmetric"
export ANTISYMMETRIC_PLANES=-1
export TEMPORAL_FILTERS="4"
export MULTISCALE_SYMMETRIC_CONV_LAYERS="1"
export RESIDUAL_KERNELS="7,15,31,61"
export TEMPORAL_KERNEL_SIZE="61"

# Training settings.
export BATCH_SIZE="128"
export EPOCHS="50"
export LR="1e-3"
export WEIGHT_DECAY="1e-4"
export LAMBDA_XP="0.0"
export LAMBDA_BT="0.0"
export LAMBDA_PLANE_BT="0.0"
export LAMBDA_BLOCK_CCA="1.0"
export LAMBDA_START_FRAC="1.0"
export S_OBJECTIVE="mean"
export S_SOFTMIN_TAU="0.05"
export T_0="10"
export T_MULT="2"

mkdir -p "$RUNS_BASE/$DATASET_NAME/runs" "$RUNS_BASE/$DATASET_NAME/synth_runs" "$CACHE_DIR" "$MPLCONFIGDIR"
