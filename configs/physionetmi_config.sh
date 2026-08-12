# PhysioNetMI settings for main_synth.py.
#
# Usage:
#   source configs/physionetmi_config.sh
#   /home/omo26/micromamba/envs/nonrev/bin/python -u main_synth.py
#
# To override a value for a quick run, source first and then export it, e.g.:
#   source configs/physionetmi_config.sh
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
export SYNTH_NORMALIZE="zscore"  # zscore or train_zscore
export SYNTH_PREPROCESS="none"
export EEG_FS="160"
export SYNTH_NOISE_STD="0"

# Split examples:
#   Hold out test subjects only, but mix remaining participants between train/val:
#     SYNTH_SPLIT=subject_random
#     SYNTH_HOLDOUT_SUBJECT_COUNT=10
#     VAL_SPLIT=0.2
#   Hold out distinct validation and test participants:
#     SYNTH_SPLIT=subject_holdout
#     SYNTH_VAL_SUBJECT_COUNT=19
#     SYNTH_HOLDOUT_SUBJECT_COUNT=20
# In subject_holdout, validation/test splits include all loaded trials from
# their selected subjects. SYNTH_MAX_TRIALS or cache construction can still cap
# which trials are loaded in the first place.
export SEED="0"
export SYNTH_MAX_TRIALS="0" # 0 means no trial cap (this is what we used)
export SYNTH_SPLIT="subject_random"  # old figure regime: train/val are random trials across selected subjects
export SYNTH_SUBJECTS_PATH="$PHYSIONETMI_SUBJECTS_NPY"
export SYNTH_SUBJECT_COUNT="0"
export SYNTH_SUBJECT_IDS="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109"
export SYNTH_VAL_SUBJECT_COUNT="0"  # subject_holdout only; keep unset for subject_random
export SYNTH_VAL_SUBJECT_IDS=""
export SYNTH_HOLDOUT_SUBJECT_COUNT="10"  # v2 sweep: hold out test participants, mix remaining participants in train/val
export SYNTH_HOLDOUT_SUBJECT_IDS=""
export SYNTH_VIZ_MAX_TRIALS="6"
export SYNTH_VIZ_MAX_TIMEPOINTS="1000"
export SYNTH_VIZ_PARTICIPANT_MODE="top_zeta"  # top_zeta or random
export SYNTH_VIZ_PARTICIPANT_COUNT="8"
export SYNTH_VIZ_PARTICIPANT_IDS=""  # optional explicit IDs for participant-condition plots
export SYNTH_VIZ_PLANE_INDICES=""     # optional explicit planes for participant-condition plots
export VAL_SPLIT="0.2" # subject_random trial fraction; subject_holdout fallback subject fraction when val count/IDs are unset

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
