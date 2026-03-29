#!/usr/bin/env bash
set -euo pipefail

WM121="/people/cs/r/rxm210041/.conda/envs/worldmodel121"
REPO="/data/rxm210041/le-wm-custom"
export STABLEWM_HOME="/data/rxm210041/stablewm"
export LD_LIBRARY_PATH="$WM121/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$WM121/lib/python3.10/site-packages/nvidia/cusparse/lib:$WM121/lib/python3.10/site-packages/nvidia/cublas/lib:$WM121/lib/python3.10/site-packages/nvidia/cudnn/lib:$WM121/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$WM121/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$WM121/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:$WM121/lib/python3.10/site-packages/nvidia/curand/lib:$WM121/lib/python3.10/site-packages/nvidia/cusolver/lib:$WM121/lib/python3.10/site-packages/nvidia/nccl/lib:$WM121/lib/python3.10/site-packages/nvidia/nvtx/lib:${LD_LIBRARY_PATH:-}"

BASELINE_PID="${1:?baseline pid required}"
PID_20="${2:?20pct train pid required}"
PID_15="${3:?15pct train pid required}"
PID_10="${4:?10pct train pid required}"
PID_05="${5:?5pct train pid required}"

BASELINE_RESULT="$STABLEWM_HOME/tworoom/tworoom_results_official_full_50ep.txt"
SUMMARY_OUT="$STABLEWM_HOME/tworoom_subset_summary.txt"

wait_for_result() {
  local pid="$1"
  local result="$2"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
  if [[ ! -f "$result" ]]; then
    echo "Expected result missing: $result" >&2
    return 1
  fi
}

wait_and_eval() {
  local pid="$1"
  local gpu="$2"
  local run_name="$3"
  local result_rel="$4"

  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done

  if ! compgen -G "$STABLEWM_HOME/$run_name/*_object.ckpt" > /dev/null; then
    echo "No object checkpoint found for $run_name" >&2
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$WM121/bin/python" "$REPO/eval.py" \
    --config-name=tworoom \
    policy="$run_name" \
    cache_dir="$STABLEWM_HOME" \
    output.filename="$result_rel"
}

wait_for_result "$BASELINE_PID" "$BASELINE_RESULT" &
BASELINE_WATCH_PID=$!

wait_and_eval "$PID_20" 0 "tworoom_pct20" "tworoom_pct20/eval_results.txt" &
EVAL_20_PID=$!
wait_and_eval "$PID_15" 1 "tworoom_pct15" "tworoom_pct15/eval_results.txt" &
EVAL_15_PID=$!
wait_and_eval "$PID_10" 2 "tworoom_pct10" "tworoom_pct10/eval_results.txt" &
EVAL_10_PID=$!
wait_and_eval "$PID_05" 3 "tworoom_pct05" "tworoom_pct05/eval_results.txt" &
EVAL_05_PID=$!

wait "$BASELINE_WATCH_PID" "$EVAL_20_PID" "$EVAL_15_PID" "$EVAL_10_PID" "$EVAL_05_PID"

"$WM121/bin/python" "$REPO/scripts/summarize_tworoom_results.py" \
  --baseline "$BASELINE_RESULT" \
  --result 5pct "$STABLEWM_HOME/tworoom_pct05/eval_results.txt" \
  --result 10pct "$STABLEWM_HOME/tworoom_pct10/eval_results.txt" \
  --result 15pct "$STABLEWM_HOME/tworoom_pct15/eval_results.txt" \
  --result 20pct "$STABLEWM_HOME/tworoom_pct20/eval_results.txt" \
  | tee "$SUMMARY_OUT"
