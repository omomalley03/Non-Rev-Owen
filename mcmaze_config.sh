# MC Maze settings for main.py.
#
# Usage:
#   source mcmaze_config.sh
#   /home/omo26/micromamba/envs/nonrev/bin/python -u main.py
#
# Or, from an activated nonrev environment:
#   source mcmaze_config.sh
#   python -u main.py
#
# To override a value for a quick run, source first and then export it, e.g.:
#   source mcmaze_config.sh
#   export EPOCHS=1 SPLIT=random

export RUNS_BASE="/home/omo26/Non-Rev-Owen"
export DATASET_NAME="mcmaze"
export CACHE_DIR="$RUNS_BASE/cache"
export MPLCONFIGDIR="/tmp/matplotlib_nonrev"

# main.py / NWB data settings.
export NWB_PATH="/home/omo26/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"
export BIN_MS="10"
export SIGMA_MS="10"
export SOFTNORM_METHOD="churchland"
export ALIGN_FIELD="move_onset_time"
export PRE_MS="200"
export POST_MS="140" # THIS IS OBSOLETE
export WINDOW_SIZE="90"
export WINDOW_STRATEGY="trial_aligned"
export VAL_SPLIT="0.1"
export SPLIT="dataset"

# Model settings.
# export D="2"
# export HIDDEN_DIM="128"
# export DEPTH="2"
# export DROPOUT="0.2"
# export TEMPORAL_FILTERS="4"
# export TEMPORAL_KERNEL_SIZE="31"

export D="128"
export HIDDEN_DIM="256"
export DEPTH="3" # depth=2 ==> one hidden layer. depth=1 ==> linear 
export DROPOUT="0.2"
export TEMPORAL_FRONTEND="symmetric_multiscale"
export TEMPORAL_FILTERS="4" # setting this to 0 makes it an MLP embedder
export MULTISCALE_SYMMETRIC_CONV_LAYERS="1"
export RESIDUAL_KERNELS="7,15,31,61" # ignore name "residual" -- that is left over from CoCoT naming 
export TEMPORAL_KERNEL_SIZE="61" # THIS IS OBSOLOTE / OLD TEMPORAL KERNEL FILTERING

# Training settings.
export BATCH_SIZE="64"
export EPOCHS="200"
export LR="1e-3"
export WEIGHT_DECAY="1e-4"
export LAMBDA_XP="0.0" # PENALISE NONREV IN CROSS-PLANES (e.g., dim 0 and dim 3 are cross-plane)
export LAMBDA_BT="0.0" 
export LAMBDA_PLANE_BT="0.0" # BARLOW-TWINS MASKING OUT DIMS THAT FORM PLANES
export LAMBDA_BLOCK_CCA="1.0" # THIS IS THE CROSS-PLANE REGULARISATION
export LAMBDA_START_FRAC="1.0" # Use this to linearly step regularisation scaling
export S_OBJECTIVE="mean" # have also tried sum, softmin across planes... USE MEAN
export S_SOFTMIN_TAU="0.05" # obsolete if using mean ^
export VAL_S_CHECKPOINTS="0.1,0.2,0.3,0.4,0.5,0.6" 
export T_0="10" # learn rate scheduler hyperparams 
export T_MULT="2"

mkdir -p "$RUNS_BASE/$DATASET_NAME/runs" "$RUNS_BASE/$DATASET_NAME/synth_runs" "$CACHE_DIR" "$MPLCONFIGDIR"
