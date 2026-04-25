#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --job-name=SimambaSmokeCompare
#SBATCH --output=/insomnia001/home/ssb2234/logs/%x-%j.out
#SBATCH --error=/insomnia001/home/ssb2234/logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --time=0-06:00
#SBATCH --mem-per-cpu=8G

set -euo pipefail

# Run three short smoke tests on the same sampled SlimPajama subset.
# These run sequentially on one GPU; on a single A6000 that is the only sensible way
# to get comparable traces without destroying throughput via GPU contention.
#
# Usage:
#   export WANDB_API_KEY='...'
#   export WANDB_ENTITY='ssb2234-columbia'
#   export WANDB_PROJECT='simamba'
#   bash scripts/run_compare_smoke_a6000.sh /path/data_dir /path/output_root

DEFAULT_DATA_DIR="/insomnia001/home/ssb2234/slimpajama_smoke"
DEFAULT_OUT_ROOT="/insomnia001/home/ssb2234/simamba_compare_smoke"
DATA_DIR="${1:-${DEFAULT_DATA_DIR}}"
OUT_ROOT="${2:-${DEFAULT_OUT_ROOT}}"

if [[ -z "${DATA_DIR}" || -z "${OUT_ROOT}" ]]; then
  echo "usage: bash scripts/run_compare_smoke_a6000.sh DATA_DIR OUTPUT_ROOT" >&2
  exit 1
fi

TRAIN_BIN="${DATA_DIR}/train.bin"
VAL_BIN="${DATA_DIR}/val.bin"

if [[ ! -f "${TRAIN_BIN}" || ! -f "${VAL_BIN}" ]]; then
  echo "Expected ${TRAIN_BIN} and ${VAL_BIN}" >&2
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
GROUP="slimpajama_smoke_${STAMP}"

COMMON_ARGS=(
  --train-data "${TRAIN_BIN}"
  --val-data "${VAL_BIN}"
  --seq-len 1024
  --global-batch-size 8
  --micro-batch-size 2
  --max-steps 200
  --save-every 100
  --eval-every 100
  --eval-iters 10
  --keep-milestones 1
  --dtype bf16
  --wandb
  --wandb-project "${WANDB_PROJECT:-simamba}"
  --wandb-entity "${WANDB_ENTITY:-ssb2234-columbia}"
  --wandb-group "${GROUP}"
)

mkdir -p "${OUT_ROOT}"

echo "data_dir=${DATA_DIR}"
echo "output_root=${OUT_ROOT}"
echo "W&B group: ${GROUP}"

.venv/bin/python scripts/train_simamba_lm.py \
  "${COMMON_ARGS[@]}" \
  --model-layer Simamba \
  --simamba-backend triton \
  --output-dir "${OUT_ROOT}/simamba_smoke" \
  --wandb-name "simamba_smoke_${STAMP}"

.venv/bin/python scripts/train_simamba_lm.py \
  "${COMMON_ARGS[@]}" \
  --model-layer Mamba2 \
  --output-dir "${OUT_ROOT}/mamba2_smoke" \
  --wandb-name "mamba2_smoke_${STAMP}"

.venv/bin/python scripts/train_simamba_lm.py \
  "${COMMON_ARGS[@]}" \
  --model-layer Mamba3 \
  --mamba3-chunk-size 64 \
  --output-dir "${OUT_ROOT}/mamba3_smoke" \
  --wandb-name "mamba3_smoke_${STAMP}"
