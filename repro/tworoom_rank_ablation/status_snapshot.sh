#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)"

export STABLEWM_HOME="${STABLEWM_HOME:-/workspace/stablewm}"
LOG_DIR="${STABLEWM_HOME}/logs/tworoom_rank_ablation"

printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'repo_root=%s\n' "${REPO_ROOT}"
printf 'stablewm_home=%s\n' "${STABLEWM_HOME}"

echo
echo "[gpu]"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader

echo
echo "[runs]"
shopt -s nullglob
for pid_path in "${LOG_DIR}"/*.pid; do
  label="$(basename "${pid_path}" .pid)"
  pid="$(cat "${pid_path}")"
  log_path="${LOG_DIR}/${label}.log"
  state="dead"
  if ps -p "${pid}" > /dev/null 2>&1; then
    state="alive"
  fi

  progress="$(tr '\r' '\n' < "${log_path}" | rg 'Epoch|global_step|Copied warmup checkpoint|latent_dim=|num_latents_used=' | tail -n 1 || true)"
  failure="$(tr '\r' '\n' < "${log_path}" | rg 'CUDA out of memory|Traceback|RuntimeError|error:' | tail -n 1 || true)"

  ckpt_epoch=""
  run_dir="${STABLEWM_HOME}/${label}"
  if [[ -d "${run_dir}" ]]; then
    ckpt_epoch="$(
      find "${run_dir}" -maxdepth 1 -type f -name '*_epoch_*_object.ckpt' \
        | sed -E 's/.*_epoch_([0-9]+)_object\.ckpt/\1/' \
        | sort -n \
        | tail -n 1
    )"
  fi

  printf '%s | pid=%s | state=%s' "${label}" "${pid}" "${state}"
  if [[ -n "${ckpt_epoch}" ]]; then
    printf ' | latest_ckpt_epoch=%s' "${ckpt_epoch}"
  fi
  if [[ -n "${progress}" ]]; then
    printf ' | progress=%s' "${progress}"
  fi
  if [[ -n "${failure}" ]]; then
    printf ' | failure=%s' "${failure}"
  fi
  printf '\n'
done
