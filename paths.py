"""Central definition of where data, caches, and run outputs live.

Every script (main.py, main_synth.py, visualize*.py, star.py, probe_*.py) reads
RUNS_DIR / SYNTH_RUNS_DIR from here, so the whole pipeline can be pointed at an
external drive, scratch directory, or local repo in one place.

Set RUNS_BASE in a per-machine env file to move all machine-specific paths
without editing source:

    export RUNS_BASE="$HOME/POC_MLP"

Derived defaults:
    runs/            -> $RUNS_BASE/runs
    synth_runs/      -> $RUNS_BASE/synth_runs
    cache/           -> $RUNS_BASE/cache

CACHE_DIR can still be set directly when the cache should live somewhere else.
"""

import os

# If RUNS_BASE is not exported, keep everything under the repo directory.
_DEFAULT_RUNS_BASE = os.path.dirname(os.path.abspath(__file__))

RUNS_BASE = os.environ.get("RUNS_BASE", _DEFAULT_RUNS_BASE)

RUNS_DIR = os.path.join(RUNS_BASE, "runs")
SYNTH_RUNS_DIR = os.path.join(RUNS_BASE, "synth_runs")
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(RUNS_BASE, "cache"))
