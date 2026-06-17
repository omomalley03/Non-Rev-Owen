"""Central definition of where run outputs are saved and loaded.

Every script (main.py, main_synth.py, visualize*.py, star.py, probe_*.py) reads
RUNS_DIR / SYNTH_RUNS_DIR from here, so the whole pipeline can be pointed at an
external drive in one place.

Runs live on the external drive under RUNS_BASE, which must contain the
``runs/`` and ``synth_runs/`` subdirectories. To change location, edit
_DEFAULT_RUNS_BASE below, or (preferred, survives a drive rename) export the env
var before running anything:

    export RUNS_BASE="/Volumes/ADATA HD710/POC_MLP"

The env var, when set, always wins over the hard-coded default.
"""

import os

# ── EDIT THIS ────────────────────────────────────────────────────────────────
# Folder on the external drive that contains the `runs/` and `synth_runs/`
# subdirectories. Volume names are case-sensitive and may contain spaces.
# _DEFAULT_RUNS_BASE = "/Volumes/ADATA HD710/POC_MLP"
_DEFAULT_RUNS_BASE = "/Users/omomalley03/Documents/Dissertation/POC_MLP"
# ─────────────────────────────────────────────────────────────────────────────

RUNS_BASE = os.environ.get("RUNS_BASE", _DEFAULT_RUNS_BASE)

RUNS_DIR = os.path.join(RUNS_BASE, "runs")
SYNTH_RUNS_DIR = os.path.join(RUNS_BASE, "synth_runs")
# 