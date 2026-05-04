#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data/slimpajama_500m_50m}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train.bin}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/val.bin}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
GROUP="${WANDB_GROUP:-simamba_annealed_50m_${STAMP}}"
RUN_NAME="${RUN_NAME:-disc10m_simamba_localconv_annealed_midpoint_50m_${STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-run_logs/${RUN_NAME}.log}"

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

mkdir -p outputs run_logs "${OUTPUT_DIR}"

setsid nohup .venv/bin/python scripts/train_simamba_lm.py \
  --train-data "${TRAIN_DATA}" \
  --val-data "${VAL_DATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --d-model "${D_MODEL:-160}" \
  --n-layer "${N_LAYER:-8}" \
  --vocab-size 50280 \
  --seq-len "${SEQ_LEN:-128}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-64}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-8}" \
  --weight-decay "${WEIGHT_DECAY:-0.1}" \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  --grad-scale-init 8192 \
  --skip-nonfinite-steps \
  --max-nonfinite-skips "${MAX_NONFINITE_SKIPS:-1}" \
  --max-steps "${MAX_STEPS:-6104}" \
  --warmup-steps "${WARMUP_STEPS:-500}" \
  --lr "${LR:-2e-4}" \
  --min-lr "${MIN_LR:-2e-5}" \
  --eval-every "${EVAL_EVERY:-500}" \
  --eval-iters "${EVAL_ITERS:-200}" \
  --save-every "${SAVE_EVERY:-2000}" \
  --keep-milestones "${KEEP_MILESTONES:-2}" \
  --dtype "${DTYPE:-fp32}" \
  --log-every "${LOG_EVERY:-10}" \
  --train-sampling epoch \
  --eval-sampling fixed \
  --eval-seed "${EVAL_SEED:-20261504}" \
  --seed "${SEED:-20260504}" \
  --wandb \
  --wandb-project "${WANDB_PROJECT:-simamba}" \
  --wandb-entity "${WANDB_ENTITY:-ssb2234-columbia}" \
  --wandb-group "${GROUP}" \
  --wandb-name "${RUN_NAME}" \
  --model-layer Simamba \
  --simamba-backend triton \
  --simamba-discretization simpson \
  --simamba-d-state "${D_STATE:-64}" \
  --simamba-headdim "${HEADDIM:-32}" \
  --simamba-chunk-size "${CHUNK_SIZE:-16}" \
  --simamba-recompute-chunk-size "${RECOMPUTE_CHUNK_SIZE:-16}" \
  --simamba-a-max "${SIMAMBA_A_MAX:-4}" \
  --simamba-d-conv "${SIMAMBA_D_CONV:-4}" \
  --simamba-outproj-norm \
  --simamba-use-midpoint-control \
  --simamba-control-logit-offset "${CONTROL_LOGIT_OFFSET:-0.0}" \
  --simamba-midpoint-logit-offset "${MIDPOINT_LOGIT_OFFSET:-0.0}" \
  --simamba-correction-anneal-min "${ANNEAL_MIN:-0.0}" \
  --simamba-correction-anneal-max "${ANNEAL_MAX:-1.0}" \
  --simamba-correction-anneal-start "${ANNEAL_START:-0}" \
  --simamba-correction-anneal-steps "${ANNEAL_STEPS:-2000}" \
  --simamba-correction-anneal-schedule "${ANNEAL_SCHEDULE:-cosine}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

pid="$!"
echo "${pid}" > "run_logs/${RUN_NAME}.pid"
echo "${pid}" > "${OUTPUT_DIR}.pid"
echo "run=${RUN_NAME}"
echo "pid=${pid}"
echo "log=${LOG_FILE}"
echo "output_dir=${OUTPUT_DIR}"
echo "group=${GROUP}"
