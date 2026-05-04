#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data/slimpajama_500m_50m}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train.bin}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/val.bin}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
GROUP="${WANDB_GROUP:-simamba_10m_continue_plus_trap_500m_${STAMP}}"

D_MODEL="${D_MODEL:-160}"
N_LAYER="${N_LAYER:-8}"
D_STATE="${D_STATE:-64}"
HEADDIM="${HEADDIM:-32}"
SEQ_LEN="${SEQ_LEN:-128}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
CONTINUE_MAX_STEPS="${CONTINUE_MAX_STEPS:-122070}"
TRAP_MAX_STEPS="${TRAP_MAX_STEPS:-61035}"
CONTINUE_LR="${CONTINUE_LR:-2e-5}"
CONTINUE_MIN_LR="${CONTINUE_MIN_LR:-2e-5}"
TRAP_LR="${TRAP_LR:-2e-4}"
TRAP_MIN_LR="${TRAP_MIN_LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
DTYPE="${DTYPE:-fp32}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_ITERS="${EVAL_ITERS:-500}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
LOG_EVERY="${LOG_EVERY:-10}"
SEED="${SEED:-20260502}"
EVAL_SEED="${EVAL_SEED:-20261502}"

WANDB_PROJECT="${WANDB_PROJECT:-simamba}"
WANDB_ENTITY="${WANDB_ENTITY:-ssb2234-columbia}"

if [[ ! -f "${TRAIN_DATA}" || ! -f "${VAL_DATA}" || ! -f "${DATA_DIR}/meta.json" ]]; then
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

base_args=(
  --train-data "${TRAIN_DATA}"
  --val-data "${VAL_DATA}"
  --d-model "${D_MODEL}"
  --n-layer "${N_LAYER}"
  --vocab-size 50280
  --seq-len "${SEQ_LEN}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --weight-decay "${WEIGHT_DECAY}"
  --grad-clip 1.0
  --grad-scale-init 8192
  --skip-nonfinite-steps
  --max-nonfinite-skips 1
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

launch_continue() {
  local key="$1"
  local output_dir="$2"
  local resume_path="${output_dir}/latest/trainer.pt"
  shift 2
  if [[ ! -f "${resume_path}" ]]; then
    echo "Missing resume checkpoint: ${resume_path}" >&2
    exit 1
  fi
  local log_file="run_logs/${key}_continue_500m_${STAMP}.log"
  echo "[launch] ${key} resume=${resume_path}"
  setsid nohup .venv/bin/python scripts/train_simamba_lm.py \
    "${base_args[@]}" \
    --max-steps "${CONTINUE_MAX_STEPS}" \
    --warmup-steps 0 \
    --lr "${CONTINUE_LR}" \
    --min-lr "${CONTINUE_MIN_LR}" \
    --resume "${resume_path}" \
    --output-dir "${output_dir}" \
    --wandb-name "${key}_continue_500m_${STAMP}" \
    "$@" \
    > "${log_file}" 2>&1 < /dev/null &
  local pid="$!"
  echo "${pid}" > "run_logs/${key}_continue_500m_${STAMP}.pid"
  echo "  pid=${pid} log=${log_file}"
}

launch_new_trap() {
  local key="disc10m_simamba_trapezoid_vec"
  local output_dir="outputs/${key}_500m_${STAMP}"
  local log_file="run_logs/${key}_500m_${STAMP}.log"
  echo "[launch] ${key} output=${output_dir}"
  mkdir -p "${output_dir}"
  setsid nohup .venv/bin/python scripts/train_simamba_lm.py \
    "${base_args[@]}" \
    --max-steps "${TRAP_MAX_STEPS}" \
    --warmup-steps 1000 \
    --lr "${TRAP_LR}" \
    --min-lr "${TRAP_MIN_LR}" \
    --output-dir "${output_dir}" \
    --wandb-name "${key}_500m_${STAMP}" \
    --model-layer Simamba \
    --simamba-backend reference \
    --simamba-discretization trapezoid \
    --simamba-d-state "${D_STATE}" \
    --simamba-headdim "${HEADDIM}" \
    --simamba-chunk-size 16 \
    --simamba-recompute-chunk-size 16 \
    --simamba-a-max 4 \
    --simamba-outproj-norm \
    > "${log_file}" 2>&1 < /dev/null &
  local pid="$!"
  echo "${pid}" > "run_logs/${key}_500m_${STAMP}.pid"
  echo "  pid=${pid} log=${log_file}"
}

launch_continue \
  disc10m_mamba2_fp32 \
  outputs/disc10m_mamba2_fp32_500m_20260502_185217 \
  --model-layer Mamba2 \
  --mamba2-d-state "${D_STATE}" \
  --mamba2-headdim "${HEADDIM}" \
  --mamba2-chunk-size 16

launch_continue \
  disc10m_simamba_fp32 \
  outputs/disc10m_simamba_fp32_500m_20260502_190333 \
  --model-layer Simamba \
  --simamba-backend triton \
  --simamba-d-state "${D_STATE}" \
  --simamba-headdim "${HEADDIM}" \
  --simamba-chunk-size 16 \
  --simamba-recompute-chunk-size 16 \
  --simamba-a-max 4 \
  --simamba-outproj-norm

launch_continue \
  disc10m_simamba_midpoint_fp32 \
  outputs/disc10m_simamba_midpoint_fp32_500m_20260502_190333 \
  --model-layer Simamba \
  --simamba-backend triton \
  --simamba-d-state "${D_STATE}" \
  --simamba-headdim "${HEADDIM}" \
  --simamba-chunk-size 16 \
  --simamba-recompute-chunk-size 16 \
  --simamba-a-max 4 \
  --simamba-use-midpoint-control \
  --simamba-outproj-norm

launch_new_trap

echo "[launch] group=${GROUP}"
echo "[launch] stamp=${STAMP}"
