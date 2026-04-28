#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --partition=burst
#SBATCH --job-name=SimambaSmokeCompare
#SBATCH --output=/insomnia001/home/ssb2234/logs/%x-%j.out
#SBATCH --error=/insomnia001/home/ssb2234/logs/%x-%j.err
#SBATCH --gres=gpu:4
#SBATCH -c 1
#SBATCH --time=0-06:00
#SBATCH --mem-per-cpu=4G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected virtualenv python at ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Run three short smoke tests on the same sampled SlimPajama subset.
# On this node we default to 4x A6000 and run each smoke test with DDP across
# all requested GPUs so the comparison finishes quickly on the available hardware.
#
# Usage:
WANDB_API_KEY='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'
WANDB_ENTITY='ssb2234-columbia'
WANDB_PROJECT='simamba'
#   bash scripts/run_compare_smoke_a6000.sh /path/data_dir /path/output_root
# Submission:
#   sbatch scripts/run_compare_smoke_a6000.sh
#   sbatch --partition=burst --gres=gpu:A6000:4 scripts/run_compare_smoke_a6000.sh  # explicit override

DEFAULT_DATA_DIR="/insomnia001/home/ssb2234/slimpajama_smoke"
DEFAULT_OUT_ROOT="/insomnia001/home/ssb2234/simamba_compare_smoke"
DATA_DIR="${1:-${DEFAULT_DATA_DIR}}"
OUT_ROOT="${2:-${DEFAULT_OUT_ROOT}}"
GPUS="${GPUS:-4}"

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
echo "gpus=${GPUS}"
echo "W&B group: ${GROUP}"

"${PYTHON_BIN}" - <<'PY'
import os
import sys
import torch

try:
    import wandb  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        "W&B support is enabled for this smoke run, but 'wandb' is not installed in the repo venv. "
        f"Install it with: {sys.executable} -m pip install wandb "
        "or sync the optional 'train' dependencies declared in pyproject.toml."
    ) from exc

print(f"torch={torch.__version__}")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"cuda.device_count={torch.cuda.device_count()}")
print(f"cuda.is_available={torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA preflight failed: PyTorch can see device entries but cannot initialize CUDA. "
        "Verify the job was started in a fresh GPU allocation and that the active Python environment has a working CUDA-enabled torch build."
    )
PY

run_train() {
  if [[ "${GPUS}" == "1" ]]; then
    "${PYTHON_BIN}" scripts/train_simamba_lm.py "$@"
  else
    "${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node="${GPUS}" scripts/train_simamba_lm.py "$@"
  fi
}

run_train \
  "${COMMON_ARGS[@]}" \
  --model-layer Simamba \
  --simamba-backend triton \
  --output-dir "${OUT_ROOT}/simamba_smoke" \
  --wandb-name "simamba_smoke_${STAMP}"

run_train \
  "${COMMON_ARGS[@]}" \
  --model-layer Mamba2 \
  --output-dir "${OUT_ROOT}/mamba2_smoke" \
  --wandb-name "mamba2_smoke_${STAMP}"

run_train \
  "${COMMON_ARGS[@]}" \
  --model-layer Mamba3 \
  --mamba3-chunk-size 64 \
  --output-dir "${OUT_ROOT}/mamba3_smoke" \
  --wandb-name "mamba3_smoke_${STAMP}"
