#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_ENV_PREFIX="/people/cs/r/rxm210041/.conda/envs/worldmodel121"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "${DEFAULT_ENV_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${DEFAULT_ENV_PREFIX}/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

export STABLEWM_HOME="${STABLEWM_HOME:-/data/rxm210041/stablewm}"
SEED_DIR="${SEED_DIR:-seed42}"
FIT_DEVICE="${FIT_DEVICE:-auto}"

if [[ -d "${DEFAULT_ENV_PREFIX}/lib/python3.10/site-packages/nvidia" ]]; then
  CUDA_LIB_ROOT="${DEFAULT_ENV_PREFIX}/lib/python3.10/site-packages/nvidia"
  for subdir in \
    nvjitlink/lib \
    cusparse/lib \
    cublas/lib \
    cudnn/lib \
    cuda_runtime/lib \
    cuda_nvrtc/lib \
    cuda_cupti/lib \
    curand/lib \
    cusolver/lib \
    nccl/lib \
    nvtx/lib
  do
    if [[ -d "${CUDA_LIB_ROOT}/${subdir}" ]]; then
      export LD_LIBRARY_PATH="${CUDA_LIB_ROOT}/${subdir}:${LD_LIBRARY_PATH:-}"
    fi
  done
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/run_tworoom_rank_ablation.sh warmup <budget> <device> [max_epochs] [run_name]
  scripts/run_tworoom_rank_ablation.sh fit-pca <budget> <warmup_run> [output_path] [max_batches]
  scripts/run_tworoom_rank_ablation.sh make-random <budget> [output_path] [random_seed]
  scripts/run_tworoom_rank_ablation.sh branch <budget> <variant> <device> <warmup_run> [max_epochs] [run_name]

Arguments:
  budget   One of: pct05 pct10 pct15 pct20
  device   GPU id like 0/1/2/3, or 'cpu'
  variant  One of: full, pca-r4, pca-r8, pca-r16, pca-r32, random-r16

Environment overrides:
  PYTHON_BIN     Python interpreter to use
  STABLEWM_HOME  Output/cache root
  SEED_DIR       Subset bundle under repro/tworoom_subset_indices (default: seed42)
  FIT_DEVICE     Device for fit_sigreg_subspace.py (default: auto)
  FIT_EXTRA_ARGS Extra args forwarded to fit_sigreg_subspace.py
  TRAIN_EXTRA_ARGS Extra Hydra overrides appended to train.py
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

validate_budget() {
  case "$1" in
    pct05|pct10|pct15|pct20) ;;
    *) die "budget must be one of pct05|pct10|pct15|pct20, got '$1'" ;;
  esac
}

subset_indices_path() {
  local budget="$1"
  printf '%s/repro/tworoom_subset_indices/%s/%s_episode_indices.npy\n' \
    "$REPO_ROOT" "$SEED_DIR" "$budget"
}

default_warmup_run() {
  local budget="$1"
  printf 'tworoom_rank_%s_%s_warmup\n' "$SEED_DIR" "$budget"
}

default_pca_artifact() {
  local budget="$1"
  printf '%s/tworoom_rank_subspaces/%s/%s_warmup_pca.pt\n' \
    "$STABLEWM_HOME" "$SEED_DIR" "$budget"
}

default_random_artifact() {
  local budget="$1"
  local seed="${2:-0}"
  printf '%s/tworoom_rank_subspaces/%s/%s_warmup_random_seed%s.pt\n' \
    "$STABLEWM_HOME" "$SEED_DIR" "$budget" "$seed"
}

run_train() {
  local device="$1"
  shift

  if [[ "$device" == "cpu" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/train.py" \
      trainer.accelerator=cpu \
      trainer.devices=1 \
      trainer.precision=32-true \
      num_workers=0 \
      "$@"
    return
  fi

  CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" "$REPO_ROOT/train.py" "$@"
}

run_warmup() {
  local budget="$1"
  local device="$2"
  local max_epochs="${3:-10}"
  local run_name="${4:-$(default_warmup_run "$budget")}"
  local indices_path
  indices_path="$(subset_indices_path "$budget")"
  require_file "$indices_path"
  local extra_args=()
  if [[ -n "${TRAIN_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "${TRAIN_EXTRA_ARGS}"
  fi

  run_train "$device" \
    data=tworoom \
    +launcher=local \
    wandb.enabled=False \
    subspace=off \
    trainer.max_epochs="$max_epochs" \
    subdir="$run_name" \
    +subset.indices_file="$indices_path" \
    "${extra_args[@]}"
}

run_fit_pca() {
  local budget="$1"
  local warmup_run="$2"
  local output_path="${3:-$(default_pca_artifact "$budget")}"
  local max_batches="${4:-100}"
  local indices_path
  indices_path="$(subset_indices_path "$budget")"
  require_file "$indices_path"

  local warmup_dir="${STABLEWM_HOME}/${warmup_run}"
  local config_path="${warmup_dir}/config.yaml"
  local ckpt_path="${warmup_dir}/lewm_weights.ckpt"
  require_file "$config_path"
  require_file "$ckpt_path"
  local extra_args=()
  if [[ -n "${FIT_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "${FIT_EXTRA_ARGS}"
  fi

  "$PYTHON_BIN" "$REPO_ROOT/scripts/fit_sigreg_subspace.py" \
    --config-path "$config_path" \
    --checkpoint-path "$ckpt_path" \
    --checkpoint-kind weights \
    --output-path "$output_path" \
    --subset-indices-file "$indices_path" \
    --max-batches "$max_batches" \
    --device "$FIT_DEVICE" \
    "${extra_args[@]}"
}

run_make_random() {
  local budget="$1"
  local output_path="${2:-$(default_random_artifact "$budget" 0)}"
  local random_seed="${3:-0}"
  local pca_artifact
  pca_artifact="$(default_pca_artifact "$budget")"
  require_file "$pca_artifact"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/make_random_subspace.py" \
    --reference-artifact "$pca_artifact" \
    --output-path "$output_path" \
    --seed "$random_seed"
}

parse_variant() {
  local variant="$1"
  VARIANT_MODE=""
  VARIANT_RANK=""
  case "$variant" in
    full)
      VARIANT_MODE="full"
      ;;
    pca-r*)
      VARIANT_MODE="pca"
      VARIANT_RANK="${variant#pca-r}"
      ;;
    random-r*)
      VARIANT_MODE="random"
      VARIANT_RANK="${variant#random-r}"
      ;;
    *)
      die "variant must be full|pca-rN|random-rN, got '$variant'"
      ;;
  esac
}

run_branch() {
  local budget="$1"
  local variant="$2"
  local device="$3"
  local warmup_run="$4"
  local max_epochs="${5:-20}"
  local run_name="${6:-tworoom_rank_${SEED_DIR}_${budget}_${variant}_e${max_epochs}}"
  local indices_path
  indices_path="$(subset_indices_path "$budget")"
  require_file "$indices_path"

  local warmup_ckpt="${STABLEWM_HOME}/${warmup_run}/lewm_weights.ckpt"
  require_file "$warmup_ckpt"

  parse_variant "$variant"

  local train_args=(
    data=tworoom
    +launcher=local
    wandb.enabled=False
    trainer.max_epochs="$max_epochs"
    subdir="$run_name"
    resume_ckpt_path="$warmup_ckpt"
    +subset.indices_file="$indices_path"
  )
  if [[ -n "${TRAIN_EXTRA_ARGS:-}" ]]; then
    local extra_args=()
    read -r -a extra_args <<< "${TRAIN_EXTRA_ARGS}"
    train_args+=("${extra_args[@]}")
  fi

  if [[ "$VARIANT_MODE" == "full" ]]; then
    train_args+=(subspace=off)
  else
    local basis_path
    if [[ "$VARIANT_MODE" == "pca" ]]; then
      basis_path="$(default_pca_artifact "$budget")"
    else
      basis_path="$(default_random_artifact "$budget" 0)"
    fi
    require_file "$basis_path"
    train_args+=(
      subspace=pca_fixed
      "subspace.rank=${VARIANT_RANK}"
      "subspace.basis_path=${basis_path}"
    )
  fi

  run_train "$device" "${train_args[@]}"
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local cmd="$1"
  shift

  case "$cmd" in
    warmup)
      [[ $# -ge 2 ]] || die "warmup requires: <budget> <device> [max_epochs] [run_name]"
      validate_budget "$1"
      run_warmup "$@"
      ;;
    fit-pca)
      [[ $# -ge 2 ]] || die "fit-pca requires: <budget> <warmup_run> [output_path] [max_batches]"
      validate_budget "$1"
      run_fit_pca "$@"
      ;;
    make-random)
      [[ $# -ge 1 ]] || die "make-random requires: <budget> [output_path] [random_seed]"
      validate_budget "$1"
      run_make_random "$@"
      ;;
    branch)
      [[ $# -ge 4 ]] || die "branch requires: <budget> <variant> <device> <warmup_run> [max_epochs] [run_name]"
      validate_budget "$1"
      run_branch "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage
      die "unknown command: $cmd"
      ;;
  esac
}

main "$@"
