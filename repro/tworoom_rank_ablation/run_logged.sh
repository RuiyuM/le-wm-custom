#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF' >&2
Usage:
  repro/tworoom_rank_ablation/run_logged.sh <label> <command> [args...]
EOF
  exit 1
fi

LABEL="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)"

export STABLEWM_HOME="${STABLEWM_HOME:-/workspace/stablewm}"
export PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
export PYTHONUNBUFFERED=1

LOG_DIR="${STABLEWM_HOME}/logs/tworoom_rank_ablation"
mkdir -p "${LOG_DIR}"

LOG_PATH="${LOG_DIR}/${LABEL}.log"
PID_PATH="${LOG_DIR}/${LABEL}.pid"
CMD_PATH="${LOG_DIR}/${LABEL}.cmd"

printf '%s\n' "$*" > "${CMD_PATH}"
touch "${LOG_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "label=${LABEL}"
echo "pid=$$"
echo "repo_root=${REPO_ROOT}"
echo "stablewm_home=${STABLEWM_HOME}"
echo "python_bin=${PYTHON_BIN}"
printf 'command='
printf '%q ' "$@"
printf '\n'

echo "$$" > "${PID_PATH}"

cd "${REPO_ROOT}"
exec stdbuf -oL -eL "$@"
