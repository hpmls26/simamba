#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data/slimpajama_500m_50m}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train.bin}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/val.bin}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
GROUP="${WANDB_GROUP:-simamba_localconv_50m_${STAMP}}"

if [[ ! -f "${TRAIN_DATA}" || ! -f "${VAL_DATA}" ]]; then
  echo "Dataset is not ready: ${DATA_DIR}" >&2
  exit 1
fi

# Reuse the local training wrapper's W&B key without duplicating it here.
if [[ -z "${WANDB_API_KEY:-}" && -f scripts/run_train_simamba_130m.sh ]]; then
  WANDB_API_KEY="$(
    sed -n "s/^DEFAULT_WANDB_API_KEY=//p" scripts/run_train_simamba_130m.sh \
      | tail -n 1 \
      | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"
  )"
  if [[ -n "${WANDB_API_KEY}" ]]; then
    export WANDB_API_KEY
  fi
fi

mkdir -p outputs run_logs

base_args=(
  --train-data "${TRAIN_DATA}"
  --val-data "${VAL_DATA}"
  --d-model "${D_MODEL:-160}"
  --n-layer "${N_LAYER:-8}"
  --vocab-size 50280
  --seq-len "${SEQ_LEN:-128}"
  --global-batch-size "${GLOBAL_BATCH_SIZE:-64}"
  --micro-batch-size "${MICRO_BATCH_SIZE:-8}"
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --grad-clip "${GRAD_CLIP:-1.0}"
  --grad-scale-init 8192
  --skip-nonfinite-steps
  --max-nonfinite-skips 1
  --max-steps "${MAX_STEPS:-6104}"
  --warmup-steps "${WARMUP_STEPS:-500}"
  --lr "${LR:-2e-4}"
  --min-lr "${MIN_LR:-2e-5}"
  --eval-every "${EVAL_EVERY:-500}"
  --eval-iters "${EVAL_ITERS:-200}"
  --save-every "${SAVE_EVERY:-2000}"
  --keep-milestones 2
  --dtype "${DTYPE:-fp32}"
  --log-every "${LOG_EVERY:-10}"
  --train-sampling epoch
  --eval-sampling fixed
  --eval-seed "${EVAL_SEED:-20261504}"
  --seed "${SEED:-20260504}"
  --wandb
  --wandb-project "${WANDB_PROJECT:-simamba}"
  --wandb-entity "${WANDB_ENTITY:-ssb2234-columbia}"
  --wandb-group "${GROUP}"
  --model-layer Simamba
  --simamba-d-state "${D_STATE:-64}"
  --simamba-headdim "${HEADDIM:-32}"
  --simamba-chunk-size 16
  --simamba-recompute-chunk-size 16
  --simamba-a-max 4
  --simamba-d-conv "${SIMAMBA_D_CONV:-4}"
  --simamba-outproj-norm
)

launch() {
  local key="$1"
  shift
  local run_name="${key}_${STAMP}"
  local output_dir="outputs/${run_name}"
  local log_file="run_logs/${run_name}.log"
  mkdir -p "${output_dir}"
  setsid nohup .venv/bin/python scripts/train_simamba_lm.py \
    "${base_args[@]}" \
    --output-dir "${output_dir}" \
    --wandb-name "${run_name}" \
    "$@" \
    > "${log_file}" 2>&1 < /dev/null &
  local pid="$!"
  echo "${pid}" > "run_logs/${run_name}.pid"
  echo "${pid}" > "${output_dir}.pid"
  echo "run=${run_name} pid=${pid} log=${log_file}"
}

launch disc10m_simamba_localconv_simpson_50m \
  --simamba-backend triton \
  --simamba-discretization simpson

launch disc10m_simamba_localconv_simpson_lowctrl_50m \
  --simamba-backend triton \
  --simamba-discretization simpson \
  --simamba-control-logit-offset -4.0

launch disc10m_simamba_localconv_trapezoid_50m \
  --simamba-backend reference \
  --simamba-discretization trapezoid

echo "group=${GROUP}"
echo "stamp=${STAMP}"
