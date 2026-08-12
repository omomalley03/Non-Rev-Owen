#!/bin/bash
#
# Creates a Python venv on top of the mlmi2 conda environment for POC_MLP.
# Inherits torch, numpy, matplotlib etc. from mlmi2, and adds the packages
# this repo needs that aren't guaranteed there (see requirements.txt).
#
# Run once on a login node before submitting any SLURM jobs:
#   bash setup_env.sh
#
# The venv is created at ~/poc_venv and is reused by slurm_synth.sh.

set -e

MLMI2_PYTHON=/rds/project/rds-xyBFuSj0hm0/MLMI2.M2025/miniconda3/envs/mlmi2/bin/python
VENV_DIR="$HOME/poc_venv"

echo "Using base Python: $MLMI2_PYTHON"
echo "Creating venv at:  $VENV_DIR"

# --system-site-packages: inherit torch, numpy, matplotlib, etc. from mlmi2
$MLMI2_PYTHON -m venv --system-site-packages "$VENV_DIR"

echo "Installing additional packages (scipy, nlb-tools, pynwb, einops, tqdm, matplotlib, seaborn)..."
"$VENV_DIR/bin/pip" install --quiet scipy nlb-tools pynwb einops tqdm matplotlib seaborn

echo ""
echo "Done. Venv ready at $VENV_DIR"
echo "Python: $VENV_DIR/bin/python"
echo ""
"$VENV_DIR/bin/python" -c "
import torch, numpy, scipy, nlb_tools, pynwb, einops, seaborn
print(f'torch {torch.__version__}, numpy {numpy.__version__}, scipy {scipy.__version__}')
print(f'pynwb {pynwb.__version__}, einops {einops.__version__}, seaborn {seaborn.__version__} — all OK')
print(f'CUDA available: {torch.cuda.is_available()}')
"
