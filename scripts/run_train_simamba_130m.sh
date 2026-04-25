#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --job-name=Simamba130MTrain
#SBATCH --output=/insomnia001/home/ssb2234/logs/%x-%j.out
#SBATCH --error=/insomnia001/home/ssb2234/logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --time=1-00:00
#SBATCH --mem-per-cpu=8G

set -euo pipefail

# Minimal launcher for Simamba LM pretraining on a single node.
#
# Usage:
WANDB_API_KEY=''
WANDB_ENTITY='ssb2234-columbia'
WANDB_PROJECT='simamba'
#   bash scripts/run_train_simamba_130m.sh /path/train.bin /path/val.bin /path/output_dir
#
# Optional environment overrides:
#   GPUS=1
#   SEQ_LEN=2048
#   GLOBAL_BATCH_SIZE=32
#   MICRO_BATCH_SIZE=4
#   MAX_STEPS=10000
#   SAVE_EVERY=1000
#   EVAL_EVERY=500
#   KEEP_MILESTONES=1
#   DTYPE=bf16
#   COMPILE=0

TRAIN_DATA="${1:-}"
VAL_DATA="${2:-}"
OUTPUT_DIR="${3:-}"

if [[ -z "${TRAIN_DATA}" || -z "${OUTPUT_DIR}" ]]; then
  echo "usage: bash scripts/run_train_simamba_130m.sh TRAIN_BIN [VAL_BIN] OUTPUT_DIR" >&2
  exit 1
fi

# Support both:
#   bash script train.bin val.bin outdir
#   bash script train.bin outdir
if [[ -n "${VAL_DATA}" && "${VAL_DATA}" != /* && ! -f "${VAL_DATA}" ]]; then
  OUTPUT_DIR="${VAL_DATA}"
  VAL_DATA=""
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set. Export it in the shell before running." >&2
  exit 1
fi

GPUS="${GPUS:-1}"
SEQ_LEN="${SEQ_LEN:-2048}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
KEEP_MILESTONES="${KEEP_MILESTONES:-1}"
DTYPE="${DTYPE:-bf16}"
COMPILE="${COMPILE:-0}"

mkdir -p "${OUTPUT_DIR}"

export WANDB_ENTITY="${WANDB_ENTITY:-ssb2234-columbia}"
export WANDB_PROJECT="${WANDB_PROJECT:-simamba}"
export PYTHONUNBUFFERED=1

ARGS=(
  --train-data "${TRAIN_DATA}"
  --output-dir "${OUTPUT_DIR}"
  --d-model 768
  --n-layer 24
  --vocab-size 50280
  --seq-len "${SEQ_LEN}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --max-steps "${MAX_STEPS}"
  --save-every "${SAVE_EVERY}"
  --eval-every "${EVAL_EVERY}"
  --keep-milestones "${KEEP_MILESTONES}"
  --dtype "${DTYPE}"
  --simamba-backend triton
  --wandb
  --wandb-project "${WANDB_PROJECT}"
)

if [[ -n "${VAL_DATA}" ]]; then
  ARGS+=(--val-data "${VAL_DATA}")
fi

if [[ "${COMPILE}" == "1" ]]; then
  ARGS+=(--compile)
fi

echo "Launching Simamba 130M-ish pretraining"
echo "  train_data=${TRAIN_DATA}"
echo "  val_data=${VAL_DATA:-<none>}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  gpus=${GPUS}"
echo "  seq_len=${SEQ_LEN}"
echo "  global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "  micro_batch_size=${MICRO_BATCH_SIZE}"
echo "  max_steps=${MAX_STEPS}"
echo "  wandb_entity=${WANDB_ENTITY}"
echo "  wandb_project=${WANDB_PROJECT}"

if [[ "${GPUS}" == "1" ]]; then
  .venv/bin/python scripts/train_simamba_lm.py "${ARGS[@]}"
else
  .venv/bin/torchrun --nproc_per_node="${GPUS}" scripts/train_simamba_lm.py "${ARGS[@]}"
fi
