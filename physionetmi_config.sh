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
export PHYSIONETMI_CONTEXT_BINS="30"
export PHYSIONETMI_NPY="$CACHE_DIR/physionetmi_all_context30_noedge.npy"
export PHYSIONETMI_LABELS_NPY="$CACHE_DIR/physionetmi_all_context30_noedge_labels.npy"
export PHYSIONETMI_SUBJECTS_NPY="$CACHE_DIR/physionetmi_all_context30_noedge_subjects.npy"
export PHYSIONETMI_KEYS_TXT="$CACHE_DIR/physionetmi_all_context30_noedge_keys.txt"

# main_synth.py data settings.
export SYNTH_DATA_PATH="$PHYSIONETMI_NPY"
export SYNTH_LABELS_PATH="$PHYSIONETMI_LABELS_NPY"
export SYNTH_DATA_LAYOUT="knt"
export SYNTH_PRECOMPUTED_CONTEXT_BINS="$PHYSIONETMI_CONTEXT_BINS"
export SYNTH_NORMALIZE="zscore"
export SYNTH_PREPROCESS="none"
export EEG_FS="160"
export SYNTH_NOISE_STD="0"

# Subject-random split draws from all exported PhysioNetMI subjects in the
# all-splits no-edge context cache. Set SYNTH_HOLDOUT_SUBJECT_COUNT to reserve
# subjects for decoder-only held-out evaluation.
export SEED="0"
export SYNTH_MAX_TRIALS="0"
export SYNTH_SPLIT="subject_random"
export SYNTH_SUBJECTS_PATH="$PHYSIONETMI_SUBJECTS_NPY"
export SYNTH_SUBJECT_COUNT="0"
export SYNTH_SUBJECT_IDS=""   # optional explicit comma-separated IDs; overrides count
export SYNTH_HOLDOUT_SUBJECT_COUNT="10"  # subject_random only: excluded from embedding train/val and used for decoder test
export SYNTH_HOLDOUT_SUBJECT_IDS=""     # optional explicit comma-separated held-out IDs
export SYNTH_VIZ_MAX_TRIALS="6"
export SYNTH_VIZ_MAX_TIMEPOINTS="1000"
export SYNTH_VIZ_PARTICIPANT_MODE="top_zeta"  # top_zeta or random
export SYNTH_VIZ_PARTICIPANT_COUNT="8"
export VAL_SPLIT="0.2"

# Model settings.
export D="128"
export HIDDEN_DIM="256"
export DEPTH="2"
export DROPOUT="0.2"
export TEMPORAL_FRONTEND="mixed_parity"
export ANTISYMMETRIC_PLANES=-1
export TEMPORAL_CONTEXT_BINS="$PHYSIONETMI_CONTEXT_BINS"
export TEMPORAL_FILTERS="4"
export MULTISCALE_SYMMETRIC_CONV_LAYERS="1"
export RESIDUAL_KERNELS="7,15,31,61"
export TEMPORAL_KERNEL_SIZE="61"

# Training settings.
export BATCH_SIZE="128"
export EPOCHS="75"
export LR="3e-4"
export WEIGHT_DECAY="1e-4"
export LAMBDA_XP="0.0"
export LAMBDA_BT="0.0"
export LAMBDA_PLANE_BT="0.0"
export LAMBDA_BLOCK_CCA="2.0"
export LAMBDA_START_FRAC="1.0"
export VAL_CHECKPOINT_METRIC="mean_plane_zeta"  # zeta, mean_plane_zeta, or s
export VAL_CHECKPOINTS="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
export S_OBJECTIVE="mean"
export S_SOFTMIN_TAU="0.05"
export T_0="50"
export T_MULT="1"

mkdir -p "$RUNS_BASE/$DATASET_NAME/runs" "$RUNS_BASE/$DATASET_NAME/synth_runs" "$CACHE_DIR" "$MPLCONFIGDIR"
