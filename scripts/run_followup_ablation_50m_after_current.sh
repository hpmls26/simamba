#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data/slimpajama_500m_50m}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train.bin}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/val.bin}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
GROUP="${WANDB_GROUP:-simamba_followup_ablation_50m_${STAMP}}"

WAIT_PIDS="${WAIT_PIDS:-297020 297021 297022 299206}"
MAX_STEPS="${MAX_STEPS:-6104}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
LR="${LR:-2e-4}"
MIN_LR="${MIN_LR:-2e-5}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_ITERS="${EVAL_ITERS:-200}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
LOG_EVERY="${LOG_EVERY:-10}"
SEED="${SEED:-20260504}"
EVAL_SEED="${EVAL_SEED:-20261504}"

D_MODEL="${D_MODEL:-160}"
N_LAYER="${N_LAYER:-8}"
D_STATE="${D_STATE:-64}"
HEADDIM="${HEADDIM:-32}"
SEQ_LEN="${SEQ_LEN:-128}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
DTYPE="${DTYPE:-fp32}"

WANDB_PROJECT="${WANDB_PROJECT:-simamba}"
WANDB_ENTITY="${WANDB_ENTITY:-ssb2234-columbia}"

if [[ ! -f "${TRAIN_DATA}" || ! -f "${VAL_DATA}" ]]; then
  echo "Dataset is not ready: ${DATA_DIR}" >&2
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" && -f scripts/run_train_simamba_130m.sh ]]; then
  WANDB_API_KEY="$(
    sed -n -E "s/^DEFAULT_WANDB_API_KEY=['\"]?([^'\"]+)['\"]?$/\1/p" scripts/run_train_simamba_130m.sh | head -1
  )"
  export WANDB_API_KEY
fi

mkdir -p outputs run_logs

is_training_pid() {
  local pid="$1"
  ps -p "${pid}" -o args= 2>/dev/null | grep -q "scripts/train_simamba_lm.py"
}

echo "[followup] waiting for current training pids: ${WAIT_PIDS}"
while true; do
  alive=()
  for pid in ${WAIT_PIDS}; do
    if is_training_pid "${pid}"; then
      alive+=("${pid}")
    fi
  done
  if [[ "${#alive[@]}" -eq 0 ]]; then
    break
  fi
  echo "[followup] still running: ${alive[*]}"
  sleep 60
done

base_args=(
  --train-data "${TRAIN_DATA}"
  --val-data "${VAL_DATA}"
  --d-model "${D_MODEL}"
  --n-layer "${N_LAYER}"
  --vocab-size 50280
  --seq-len "${SEQ_LEN}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --weight-decay 0.1
  --grad-clip 1.0
  --grad-scale-init 8192
  --skip-nonfinite-steps
  --max-nonfinite-skips 1
  --max-steps "${MAX_STEPS}"
  --warmup-steps "${WARMUP_STEPS}"
  --lr "${LR}"
  --min-lr "${MIN_LR}"
  --eval-every "${EVAL_EVERY}"
  --eval-iters "${EVAL_ITERS}"
  --save-every "${SAVE_EVERY}"
  --keep-milestones 2
  --dtype "${DTYPE}"
  --log-every "${LOG_EVERY}"
  --train-sampling epoch
  --eval-sampling fixed
  --eval-seed "${EVAL_SEED}"
  --seed "${SEED}"
  --wandb
  --wandb-project "${WANDB_PROJECT}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-group "${GROUP}"
)

launch() {
  local key="$1"
  shift
  local output_dir="outputs/${key}_${STAMP}"
  local log_file="run_logs/${key}_${STAMP}.log"
  mkdir -p "${output_dir}"
  echo "[followup] launch ${key}"
  setsid nohup .venv/bin/python scripts/train_simamba_lm.py \
    "${base_args[@]}" \
    --output-dir "${output_dir}" \
    --wandb-name "${key}_${STAMP}" \
    "$@" \
    > "${log_file}" 2>&1 < /dev/null &
  local pid="$!"
  echo "${pid}" > "run_logs/${key}_${STAMP}.pid"
  echo "[followup] ${key} pid=${pid} log=${log_file}"
}

launch disc10m_ablate_trapezoid_vec_50m \
  --model-layer Simamba \
  --simamba-backend reference \
  --simamba-discretization trapezoid \
  --simamba-d-state "${D_STATE}" \
  --simamba-headdim "${HEADDIM}" \
  --simamba-chunk-size 16 \
  --simamba-recompute-chunk-size 16 \
  --simamba-a-max 4 \
  --simamba-outproj-norm

launch disc10m_ablate_simpson_default_50m \
  --model-layer Simamba \
  --simamba-backend triton \
  --simamba-discretization simpson \
  --simamba-d-state "${D_STATE}" \
  --simamba-headdim "${HEADDIM}" \
  --simamba-chunk-size 16 \
  --simamba-recompute-chunk-size 16 \
  --simamba-a-max 4 \
  --simamba-outproj-norm

launch disc10m_ablate_simpson_lowctrl_50m \
  --model-layer Simamba \
  --simamba-backend triton \
  --simamba-discretization simpson \
  --simamba-control-logit-offset -4.0 \
  --simamba-d-state "${D_STATE}" \
  --simamba-headdim "${HEADDIM}" \
  --simamba-chunk-size 16 \
  --simamba-recompute-chunk-size 16 \
  --simamba-a-max 4 \
  --simamba-outproj-norm

launch disc10m_ablate_mamba2_dconv1_50m \
  --model-layer Mamba2 \
  --mamba2-d-state "${D_STATE}" \
  --mamba2-d-conv 1 \
  --mamba2-headdim "${HEADDIM}" \
  --mamba2-chunk-size 16

echo "[followup] group=${GROUP}"
echo "[followup] stamp=${STAMP}"
