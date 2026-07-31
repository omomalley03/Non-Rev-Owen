#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Copy generated figure files from this repo to another machine or local folder.

Usage:
  scripts/copy_figures_to_personal.sh [options] DEST

DEST examples:
  user@personal-host:/Users/user/NonRevFigures/
  user@personal-host:~/NonRevFigures/
  /tmp/NonRevFigures/

Options:
  --dry-run      Show what would be copied without copying.
  --with-data    Also copy .csv and .npz sidecar files.
  -h, --help     Show this help.

Notes:
  This is a push script when run over SSH on the remote/training machine. That
  only works if your personal machine is reachable by SSH from this machine.

  If your personal machine is not running an SSH server, run rsync from your
  personal machine instead, pulling from this machine:

    rsync -avz --prune-empty-dirs \
      --include='*/' --include='*.png' --include='*.pdf' --include='*.svg' \
      --include='*.jpg' --include='*.jpeg' --exclude='*' \
      YOUR_REMOTE:/home/omo26/Non-Rev-Owen/ ~/NonRevFigures/
EOF
}

dry_run=()
with_data=0
dest=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=(--dry-run)
            shift
            ;;
        --with-data)
            with_data=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "$dest" ]]; then
                echo "Expected one DEST, got extra argument: $1" >&2
                usage >&2
                exit 2
            fi
            dest="$1"
            shift
            ;;
    esac
done

if [[ -z "$dest" ]]; then
    usage >&2
    exit 2
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required but was not found on PATH." >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

candidate_dirs=(
    "runs"
    "mcmaze/runs"
    "physionetmi/runs"
    "physionetmi/condition_accuracy_sweep_6subj/runs"
    "physionetmi/condition_accuracy_sweep_12subj/runs"
    "physionetmi/condition_accuracy_sweep_30subj/runs"
    "physionetmi/full_frozen_temporal_conv_sweep_89subj/runs"
    "physionetmi/multiscale_symmetric_sweep/runs"
    "physionetmi/multiscale_symmetric_sweep_23subj/runs"
    "synth_runs"
    "faced/runs"
)

sources=()
cd "$repo_root"
for dir in "${candidate_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
        sources+=("./$dir/")
    fi
done

if [[ ${#sources[@]} -eq 0 ]]; then
    echo "No run directories found under $repo_root." >&2
    exit 1
fi

include_args=(
    --include='*/'
    --include='*.png'
    --include='*.PNG'
    --include='*.pdf'
    --include='*.PDF'
    --include='*.svg'
    --include='*.SVG'
    --include='*.jpg'
    --include='*.JPG'
    --include='*.jpeg'
    --include='*.JPEG'
)

if [[ "$with_data" -eq 1 ]]; then
    include_args+=(
        --include='*.csv'
        --include='*.CSV'
        --include='*.npz'
        --include='*.NPZ'
    )
fi

echo "Copying figures from $repo_root to $dest"
rsync -avz --prune-empty-dirs --relative \
    "${dry_run[@]}" \
    "${include_args[@]}" \
    --exclude='*' \
    "${sources[@]}" \
    "$dest"
