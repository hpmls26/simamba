#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --job-name=SlimPajamaPrepTrain
#SBATCH --output=/insomnia001/home/ssb2234/logs/%x-%j.out
#SBATCH --error=/insomnia001/home/ssb2234/logs/%x-%j.err
#SBATCH -c 2
#SBATCH --time=0-05:00
#SBATCH --mem-per-cpu=4G

set -euo pipefail

# Prepare a bounded SlimPajama subset sized for training on limited local disk.
# By default this writes about 5.0 GB total across train.bin and val.bin because
# the prep script stores uint16 tokens (2 bytes per token).
#
# Usage:
#   bash scripts/prepare_slimpajama_train.sh /path/output_dir
#
# Optional environment overrides:
#   TRAIN_TOKENS=2450000000
#   VAL_TOKENS=50000000
#   DATASET_NAME=MBZUAI-LLM/SlimPajama-627B-DC
#   DATASET_CONFIG=default
#   TOKENIZER=EleutherAI/gpt-neox-20b
#   SHUFFLE_BUFFER=10000
#   ALLOWED_SOURCES="RedPajamaWikipedia RedPajamaArXiv"

DEFAULT_OUT_DIR="/insomnia001/home/ssb2234/slimpajama_train_5g"
OUT_DIR="${1:-${DEFAULT_OUT_DIR}}"
TRAIN_TOKENS="${TRAIN_TOKENS:-2450000000}"
VAL_TOKENS="${VAL_TOKENS:-50000000}"
DATASET_NAME="${DATASET_NAME:-MBZUAI-LLM/SlimPajama-627B-DC}"
DATASET_CONFIG="${DATASET_CONFIG:-default}"
TOKENIZER="${TOKENIZER:-EleutherAI/gpt-neox-20b}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-10000}"
ALLOWED_SOURCES="${ALLOWED_SOURCES:-}"

if [[ -z "${OUT_DIR}" ]]; then
  echo "usage: bash scripts/prepare_slimpajama_train.sh OUTPUT_DIR" >&2
  exit 1
fi

echo "[prepare-train] output_dir=${OUT_DIR}"
echo "[prepare-train] train_tokens=${TRAIN_TOKENS}"
echo "[prepare-train] val_tokens=${VAL_TOKENS}"
echo "[prepare-train] dataset=${DATASET_NAME}"
echo "[prepare-train] config=${DATASET_CONFIG}"
echo "[prepare-train] tokenizer=${TOKENIZER}"
echo "[prepare-train] shuffle_buffer=${SHUFFLE_BUFFER}"
if [[ -n "${ALLOWED_SOURCES}" ]]; then
  echo "[prepare-train] allowed_sources=${ALLOWED_SOURCES}"
else
  echo "[prepare-train] allowed_sources=<all>"
fi

ARGS=(
  --output-dir "${OUT_DIR}"
  --train-tokens "${TRAIN_TOKENS}"
  --val-tokens "${VAL_TOKENS}"
  --dataset-name "${DATASET_NAME}"
  --dataset-config "${DATASET_CONFIG}"
  --tokenizer "${TOKENIZER}"
  --shuffle-buffer "${SHUFFLE_BUFFER}"
)

if [[ -n "${ALLOWED_SOURCES}" ]]; then
  read -r -a allowed_sources_array <<< "${ALLOWED_SOURCES}"
  ARGS+=(--allowed-sources "${allowed_sources_array[@]}")
fi

.venv/bin/python scripts/prepare_slimpajama.py "${ARGS[@]}"
