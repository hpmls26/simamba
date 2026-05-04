#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --partition=burst
#SBATCH --job-name=SimambaSmokeCompare
#SBATCH --output=/insomnia001/home/ssb2234/logs/%x-%j.out
#SBATCH --error=/insomnia001/home/ssb2234/logs/%x-%j.err
#SBATCH --export=ALL
#SBATCH --gres=gpu:4
#SBATCH -c 1
#SBATCH --time=0-06:00
#SBATCH --mem-per-cpu=4G

set -euo pipefail

resolve_repo_root() {
  local script_dir candidate job_command script_path
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  for candidate in \
    "${SLURM_SUBMIT_DIR:-}"
  do
    if [[ -n "${candidate}" && -x "${candidate}/.venv/bin/python" && -f "${candidate}/pyproject.toml" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    job_command="$(
      scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | sed -n 's/.* Command=\([^[:space:]]*\).*/\1/p' | head -n 1 || true
    )"
    if [[ -n "${job_command}" ]]; then
      if [[ "${job_command}" = /* ]]; then
        script_path="${job_command}"
      else
        script_path="${SLURM_SUBMIT_DIR:-${PWD}}/${job_command}"
      fi
      candidate="$(cd "$(dirname "${script_path}")/.." && pwd)"
      if [[ -x "${candidate}/.venv/bin/python" && -f "${candidate}/pyproject.toml" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    fi
  fi

  candidate="$(cd "${script_dir}/.." && pwd)"
  if [[ -x "${candidate}/.venv/bin/python" && -f "${candidate}/pyproject.toml" ]]; then
    printf '%s\n' "${candidate}"
    return 0
  fi

  return 1
}

REPO_ROOT="$(resolve_repo_root)" || {
  echo "Unable to locate repo root from SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}' or BASH_SOURCE='${BASH_SOURCE[0]}'." >&2
  exit 1
}
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
if [[ -z "${GPUS:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
    GPUS="${#visible_devices[@]}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d '[:space:]')"
  else
    GPUS=4
  fi
fi
EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-}"

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

if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  for gpu_name in "${gpu_names[@]}"; do
    if [[ "${gpu_name}" != *"${EXPECTED_GPU_NAME}"* ]]; then
      echo "Expected GPUs matching '${EXPECTED_GPU_NAME}', but found '${gpu_name}'." >&2
      exit 1
    fi
  done
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

STAMP="$(date +%Y%m%d_%H%M%S)"
GROUP="slimpajama_smoke_${STAMP}"

export WANDB_ENTITY="${WANDB_ENTITY:-ssb2234-columbia}"
export WANDB_PROJECT="${WANDB_PROJECT:-simamba}"
WANDB_ROOT="${WANDB_ROOT:-${OUT_ROOT}/.wandb}"
export WANDB_DIR="${WANDB_DIR:-${WANDB_ROOT}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_ROOT}/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_ROOT}/config}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-auto}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

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
  --dtype auto
  --wandb
  --wandb-project "${WANDB_PROJECT}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-group "${GROUP}"
  --wandb-console "${WANDB_CONSOLE}"
)

mkdir -p "${OUT_ROOT}"

echo "data_dir=${DATA_DIR}"
echo "output_root=${OUT_ROOT}"
echo "gpus=${GPUS}"
echo "expected_gpu=${EXPECTED_GPU_NAME}"
echo "W&B group: ${GROUP}"
echo "W&B dir: ${WANDB_DIR}"
echo "W&B console: ${WANDB_CONSOLE}"

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
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${GPUS}" scripts/train_simamba_lm.py "$@"
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
