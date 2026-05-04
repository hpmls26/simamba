#!/usr/bin/env bash
set -euo pipefail

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
TASK="${TASK:-mod_sum}"
MODULUS="${MODULUS:-16}"
SEQ_LEN="${SEQ_LEN:-128}"
EVAL_SEQ_LENS="${EVAL_SEQ_LENS:-128 256}"
STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_BATCHES="${EVAL_BATCHES:-32}"
LOG_EVERY="${LOG_EVERY:-25}"
LR="${LR:-1e-3}"
DTYPE="${DTYPE:-fp32}"
DEVICE="${DEVICE:-cuda}"

D_MODEL="${D_MODEL:-64}"
N_LAYER="${N_LAYER:-2}"
D_STATE="${D_STATE:-32}"
HEADDIM="${HEADDIM:-32}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"

WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-simamba}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-state_tracking_${TASK}_${STAMP}}"
PARALLEL="${PARALLEL:-0}"

mkdir -p run_logs outputs

COMMON_ARGS=(
  --task "${TASK}"
  --modulus "${MODULUS}"
  --seq-len "${SEQ_LEN}"
  --eval-seq-lens ${EVAL_SEQ_LENS}
  --batch-size "${BATCH_SIZE}"
  --steps "${STEPS}"
  --eval-every "${EVAL_EVERY}"
  --eval-batches "${EVAL_BATCHES}"
  --log-every "${LOG_EVERY}"
  --lr "${LR}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --d-model "${D_MODEL}"
  --n-layer "${N_LAYER}"
  --d-state "${D_STATE}"
  --headdim "${HEADDIM}"
  --chunk-size "${CHUNK_SIZE}"
)

WANDB_ARGS=()
if [[ "${WANDB}" == "1" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" && -f scripts/run_train_simamba_130m.sh ]]; then
    WANDB_API_KEY="$(
      sed -n -E "s/^DEFAULT_WANDB_API_KEY=['\"]?([^'\"]+)['\"]?$/\1/p" scripts/run_train_simamba_130m.sh | head -1
    )"
    export WANDB_API_KEY
  fi
  WANDB_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT}" --wandb-group "${WANDB_GROUP}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
  fi
fi

run_one() {
  local name="$1"
  shift
  local out_dir="outputs/${name}_${STAMP}"
  local log_file="run_logs/${name}_${STAMP}.log"
  echo "starting ${name}: ${log_file}"
  PYTHONUNBUFFERED=1 .venv/bin/python scripts/train_state_tracking.py \
    "${COMMON_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    --wandb-name "${name}_${STAMP}" \
    --output-dir "${out_dir}" \
    "$@" > "${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${log_file}.pid"
  if [[ "${PARALLEL}" != "1" ]]; then
    wait "${pid}"
  fi
}

run_one "state_${TASK}_simamba_simpson" \
  --model-layer Simamba \
  --simamba-backend triton \
  --simamba-discretization simpson \
  --simamba-a-max 4 \
  --simamba-outproj-norm

run_one "state_${TASK}_simamba_trapezoid" \
  --model-layer Simamba \
  --simamba-backend reference \
  --simamba-discretization trapezoid \
  --simamba-a-max 4 \
  --simamba-outproj-norm

run_one "state_${TASK}_mamba2" \
  --model-layer Mamba2

if [[ "${PARALLEL}" == "1" ]]; then
  wait
fi

echo "done: ${WANDB_GROUP}"
