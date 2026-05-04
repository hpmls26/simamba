#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data/slimpajama_500m_50m}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train.bin}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/val.bin}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
GROUP="${WANDB_GROUP:-simamba_10m_variants_500m_${STAMP}}"

D_MODEL="${D_MODEL:-160}"
N_LAYER="${N_LAYER:-8}"
D_STATE="${D_STATE:-64}"
HEADDIM="${HEADDIM:-32}"
SEQ_LEN="${SEQ_LEN:-128}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-61035}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
LR="${LR:-2e-4}"
MIN_LR="${MIN_LR:-2e-5}"
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
  echo "Expected train.bin, val.bin, and meta.json." >&2
  exit 1
fi

# Reuse the W&B key from the existing training wrapper without duplicating it here.
if [[ -z "${WANDB_API_KEY:-}" && -f scripts/run_train_simamba_130m.sh ]]; then
  WANDB_API_KEY="$(
    sed -n "s/^DEFAULT_WANDB_API_KEY=['\"]\\{0,1\\}//p" scripts/run_train_simamba_130m.sh \
      | tail -n 1 \
      | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"
  )"
  if [[ -n "${WANDB_API_KEY}" ]]; then
    export WANDB_API_KEY
  fi
fi

mkdir -p outputs run_logs

common_args=(
  --train-data "${TRAIN_DATA}"
  --val-data "${VAL_DATA}"
  --d-model "${D_MODEL}"
  --n-layer "${N_LAYER}"
  --vocab-size 50280
  --seq-len "${SEQ_LEN}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --max-steps "${MAX_STEPS}"
  --warmup-steps "${WARMUP_STEPS}"
  --lr "${LR}"
  --min-lr "${MIN_LR}"
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

launch() {
  local model_key="$1"
  shift
  local run_name="disc10m_${model_key}_500m_${STAMP}"
  local output_dir="outputs/${run_name}"
  local log_file="run_logs/${run_name}.log"

  echo "[launch] ${model_key}"
  echo "  output=${output_dir}"
  echo "  log=${log_file}"
  mkdir -p "${output_dir}"

  setsid nohup .venv/bin/python scripts/train_simamba_lm.py \
    "${common_args[@]}" \
    --output-dir "${output_dir}" \
    --wandb-name "${run_name}" \
    "$@" \
    > "${log_file}" 2>&1 < /dev/null &

  local pid="$!"
  echo "${pid}" > "${output_dir}.pid"
  echo "  pid=${pid}"
}

launch simamba \
  --model-layer Simamba \
  --simamba-backend triton \
  --simamba-d-state "${D_STATE}" \
  --simamba-headdim "${HEADDIM}" \
  --simamba-chunk-size 16 \
  --simamba-recompute-chunk-size 16 \
  --simamba-a-max 4 \
  --simamba-outproj-norm

launch simamba_midpoint \
  --model-layer Simamba \
  --simamba-backend triton \
  --simamba-d-state "${D_STATE}" \
  --simamba-headdim "${HEADDIM}" \
  --simamba-chunk-size 16 \
  --simamba-recompute-chunk-size 16 \
  --simamba-a-max 4 \
  --simamba-use-midpoint-control \
  --simamba-outproj-norm

launch mamba2_fp32 \
  --model-layer Mamba2 \
  --mamba2-d-state "${D_STATE}" \
  --mamba2-headdim "${HEADDIM}" \
  --mamba2-chunk-size 16

echo "[launch] group=${GROUP}"
echo "[launch] stamp=${STAMP}"
