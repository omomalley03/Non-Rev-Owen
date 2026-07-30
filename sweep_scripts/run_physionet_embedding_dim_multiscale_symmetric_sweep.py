"""Run PhysioNetMI embedder dimensions with multiscale-symmetric filters.

This intentionally sources ``physionetmi_config.sh`` for every run so the
baseline data, model, and training parameters stay in one place.  The only
overrides are:

  - D: swept over powers of two from 32 to 2048
  - EPOCHS: fixed at 50
  - TEMPORAL_FRONTEND: multiscale_symmetric
  - ANTISYMMETRIC_PLANES: 0
  - SYNTH_SKIP_DIAGNOSTIC_PLOTS: 1

Usage:
    micromamba run -n nonrev python -u sweep_scripts/run_physionet_embedding_dim_multiscale_symmetric_sweep.py
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIMS = (32, 64, 128, 256, 512, 1024, 2048)
DEFAULT_LOG_DIR = REPO_ROOT / "physionetmi" / "embedding_dim_multiscale_symmetric_sweep" / "logs"


def parse_dims(spec: str) -> list[int]:
    dims = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if not dims:
        raise ValueError("--dims must contain at least one dimension")
    bad = [d for d in dims if d <= 0 or d % 2 != 0]
    if bad:
        raise ValueError(f"embedding dimensions must be positive and even, got {bad}")
    return dims


def run_one(dim: int, args: argparse.Namespace) -> None:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"d{dim}.log"
    config_path = args.config.resolve()
    main_path = REPO_ROOT / "main_synth.py"
    python_exe = args.python or sys.executable

    command = "\n".join(
        [
            f". {shlex.quote(str(config_path))}",
            f"export D={dim}",
            "export EPOCHS=50",
            'export TEMPORAL_FRONTEND="multiscale_symmetric"',
            "export ANTISYMMETRIC_PLANES=0",
            "export SYNTH_SKIP_DIAGNOSTIC_PLOTS=1",
            f"{shlex.quote(str(python_exe))} -u {shlex.quote(str(main_path))}",
        ]
    )

    print(f"Running D={dim}; log={log_path}", flush=True)
    if args.dry_run:
        print(command)
        return

    with log_path.open("w") as log_file:
        subprocess.run(
            ["/bin/sh", "-lc", command],
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dims",
        default=",".join(str(d) for d in DEFAULT_DIMS),
        help="Comma-separated embedding dimensions to run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "physionetmi_config.sh",
        help="Config shell file to source before applying sweep overrides.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory for per-dimension stdout/stderr logs.",
    )
    parser.add_argument(
        "--python",
        default="",
        help="Python executable for main_synth.py. Defaults to this interpreter.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for dim in parse_dims(args.dims):
        run_one(dim, args)


if __name__ == "__main__":
    main()
