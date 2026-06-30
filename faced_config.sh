# Compatibility wrapper. The editable FACED setup lives in faced_setup.sh.
#
# Usage:
#   source faced_config.sh
#   python -u main_synth.py

_FACED_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)"
source "$_FACED_CONFIG_DIR/faced_setup.sh"
unset _FACED_CONFIG_DIR
