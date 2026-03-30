#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  cat <<'EOF' >&2
Usage:
  repro/tworoom_rank_ablation/evaluate_tworoom_run.sh <run_name> [output_relpath]
EOF
  exit 1
fi

RUN_NAME="$1"
OUTPUT_REL="${2:-${RUN_NAME}/eval_results.txt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)"

export STABLEWM_HOME="${STABLEWM_HOME:-/workspace/stablewm}"
export PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
export EVAL_DEVICE="${EVAL_DEVICE:-0}"

cd "${REPO_ROOT}"
exec env CUDA_VISIBLE_DEVICES="${EVAL_DEVICE}" \
  "${PYTHON_BIN}" "${REPO_ROOT}/eval.py" \
  --config-name=tworoom \
  policy="${RUN_NAME}" \
  cache_dir="${STABLEWM_HOME}" \
  output.filename="${OUTPUT_REL}"
