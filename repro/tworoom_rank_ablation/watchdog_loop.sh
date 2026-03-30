#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STABLEWM_HOME="${STABLEWM_HOME:-/workspace/stablewm}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
LOG_DIR="${STABLEWM_HOME}/logs/tworoom_rank_ablation"
STATE_PATH="${LOG_DIR}/pipeline_state.json"
WATCHDOG_LOG="${LOG_DIR}/watchdog.log"
WATCHER_CMD="${PYTHON_BIN} ${REPO_ROOT}/repro/tworoom_rank_ablation/watch_pipeline.py"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-600}"
STATE_STALE_SECONDS="${STATE_STALE_SECONDS:-300}"
GPU_IDLE_UTIL_MAX="${GPU_IDLE_UTIL_MAX:-25}"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${WATCHDOG_LOG}"
}

watcher_pid() {
  ps -eo pid,args --no-headers | awk -v py="${PYTHON_BIN}" -v script="${REPO_ROOT}/repro/tworoom_rank_ablation/watch_pipeline.py" '$2==py && $3==script {print $1; exit}'
}

start_watcher() {
  if [[ -f /workspace/.secrets/github_env.sh ]]; then
    # shellcheck disable=SC1091
    source /workspace/.secrets/github_env.sh
  fi
  (
    cd "${REPO_ROOT}"
    export STABLEWM_HOME
    export PYTHON_BIN
    export PYTHONUNBUFFERED=1
    export GPU_MAX_JOBS="${GPU_MAX_JOBS:-10}"
    export GPU_LAUNCH_UTIL_MAX="${GPU_LAUNCH_UTIL_MAX:-90}"
    export GPU_MEMORY_HEADROOM_MIB="${GPU_MEMORY_HEADROOM_MIB:-6144}"
    export RETRY_COOLDOWN_SECONDS="${RETRY_COOLDOWN_SECONDS:-180}"
    nohup bash -lc "${WATCHER_CMD}" >> "${LOG_DIR}/watch_pipeline_console.log" 2>&1 &
  )
  sleep 2
  log "started watcher pid=$(watcher_pid || true)"
}

ensure_watcher() {
  local pid
  pid="$(watcher_pid || true)"
  if [[ -z "${pid}" ]]; then
    log "watcher missing; restarting"
    start_watcher
    return
  fi
  log "watcher alive pid=${pid}"
}

check_state_and_gpu() {
  local gpu_line util ready running failed state_ts state_age
  gpu_line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,power.draw --format=csv,noheader,nounits | head -n1)"
  util="$(cut -d',' -f1 <<< "${gpu_line}" | tr -d ' ')"

  if [[ -f "${STATE_PATH}" ]]; then
    read -r ready running failed state_ts state_age < <(
      "${PYTHON_BIN}" - <<'PY'
import json, time
from pathlib import Path
p = Path("/workspace/stablewm/logs/tworoom_rank_ablation/pipeline_state.json")
data = json.loads(p.read_text())
jobs = data["jobs"]
ready = sum(1 for v in jobs.values() if v["status"] == "pending" and all(jobs[d]["status"] == "completed" for d in v["deps"]))
running = sum(1 for v in jobs.values() if v["status"] == "running")
failed = sum(1 for v in jobs.values() if v["status"] == "failed")
ts = p.stat().st_mtime
print(ready, running, failed, int(ts), int(time.time() - ts))
PY
    )
    log "gpu=${gpu_line} ready=${ready} running=${running} failed=${failed} state_age_s=${state_age}"
    if (( state_age > STATE_STALE_SECONDS )); then
      log "state file stale; restarting watcher"
      pkill -f "${REPO_ROOT}/repro/tworoom_rank_ablation/watch_pipeline.py" || true
      sleep 2
      start_watcher
      return
    fi
    if (( ready > 0 && util < GPU_IDLE_UTIL_MAX )); then
      log "gpu idle while ready jobs exist; watcher should launch soon"
    fi
  else
    log "missing pipeline_state.json; restarting watcher"
    pkill -f "${REPO_ROOT}/repro/tworoom_rank_ablation/watch_pipeline.py" || true
    sleep 2
    start_watcher
  fi
}

while true; do
  ensure_watcher
  check_state_and_gpu
  sleep "${CHECK_INTERVAL_SECONDS}"
done
