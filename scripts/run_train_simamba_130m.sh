#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --job-name=Simamba130MTrain
#SBATCH --output=/insomnia001/home/ssb2234/logs/%x-%j.out
#SBATCH --error=/insomnia001/home/ssb2234/logs/%x-%j.err
#SBATCH --export=ALL
#SBATCH --gres=gpu:A6000:4
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --time=0-12:00
#SBATCH --mem-per-cpu=8G

set -euo pipefail

# Slurm batch shells are not guaranteed to source the user's interactive shell
# startup files. Load ~/.bashrc explicitly so CUDA-related env stays consistent
# with manual launches. The cluster's /etc/bashrc and profile.d fragments are
# not strict-mode-safe, so temporarily relax strict shell options while sourcing.
if [[ -f "${HOME}/.bashrc" ]]; then
  had_errexit=0
  had_nounset=0
  had_pipefail=0
  case $- in
    *e*)
      had_errexit=1
      set +e
      ;;
  esac
  case $- in
    *u*)
      had_nounset=1
      set +u
      ;;
  esac
  if set -o | grep -q '^pipefail[[:space:]]*on$'; then
    had_pipefail=1
    set +o pipefail
  fi
  # shellcheck disable=SC1090
  . "${HOME}/.bashrc"
  if (( had_pipefail )); then
    set -o pipefail
  fi
  if (( had_errexit )); then
    set -e
  fi
  if (( had_nounset )); then
    set -u
  fi
fi

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
mkdir -p /insomnia001/home/ssb2234/logs

log_phase() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*"
}

print_usage() {
  cat >&2 <<'EOF'
usage: bash scripts/run_train_simamba_130m.sh TRAIN_BIN [VAL_BIN] OUTPUT_DIR [options]

options:
  --bucket NAME                  Override the GCS bucket used for restore/export.
  --gcs-export 0|1               Enable or disable async checkpoint export.
  --gcs-restore-if-missing 0|1   Attempt GCS restore when no local latest checkpoint exists.
  --resume PATH                  Resume from a specific trainer.pt checkpoint.
  --gpus N                       Override the visible GPU count expected by the launcher.
  --expected-gpu-name NAME       Override the GPU model substring check.
  --wandb-project NAME           Override the W&B project name.
  --wandb-entity NAME            Override the W&B entity.
  --wandb-name NAME              Override the W&B run name.
  --wandb-group NAME             Override the W&B run group.
  --wandb-id ID                  Resume a specific W&B run ID.
  --wandb-resume POLICY          W&B resume policy: allow, must, never, or auto.
  -h, --help                     Show this help text.
EOF
}

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Single-node Simamba pretraining launcher tuned to mirror the known-good
# 4x burst GPU smoke job while keeping checkpoint exports off the training
# critical path.
#
# Optional environment overrides:
#   GPUS=4
#   EXPECTED_GPU_NAME=A6000
#   SEQ_LEN=2048
#   GLOBAL_BATCH_SIZE=32
#   MICRO_BATCH_SIZE=1
#   SIMAMBA_CHUNK_SIZE=64
#   SIMAMBA_RECOMPUTE_CHUNK_SIZE=512  # Defaulted automatically for MICRO_BATCH_SIZE=1
#   LR=2e-5
#   MIN_LR=1e-5
#   WARMUP_STEPS=2000
#   WEIGHT_DECAY=0.1
#   BETA1=0.9
#   BETA2=0.98
#   GRAD_CLIP=1.0
#   SIMAMBA_DT_LIMIT_MIN=0.001
#   SIMAMBA_DT_LIMIT_MAX=0.1
#   SIMAMBA_A_MAX=auto
#   SIMAMBA_OUTPROJ_NORM=1
#   MAX_STEPS=10000
#   SAVE_EVERY=25
#   EVAL_EVERY=500
#   KEEP_MILESTONES=4
#   DTYPE=auto
#   COMPILE=0
#   RESUME_CHECKPOINT=/path/to/trainer.pt
#   GCS_EXPORT=1
#   GCS_PREFIX=experiments
#   GCS_RUN_PREFIX=my-run-name
#   GCS_EXPORT_POLL_SECS=15
#   GCS_PROJECT=my-project
#   GCS_RESTORE_IF_MISSING=1

DEFAULT_WANDB_API_KEY='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'
DEFAULT_WANDB_ENTITY='ssb2234-columbia'
DEFAULT_WANDB_PROJECT='simamba'
DEFAULT_GCS_BUCKET='hpml-model-checkpoints'
DEFAULT_GCS_CREDENTIALS="${REPO_ROOT}/scripts/gcs-key.json"

POSITIONAL_ARGS=()
CLI_GCS_BUCKET=""
CLI_GCS_EXPORT=""
CLI_GCS_RESTORE_IF_MISSING=""
CLI_RESUME_CHECKPOINT=""
CLI_GPUS=""
CLI_EXPECTED_GPU_NAME=""
CLI_WANDB_PROJECT=""
CLI_WANDB_ENTITY=""
CLI_WANDB_NAME=""
CLI_WANDB_GROUP=""
CLI_WANDB_ID=""
CLI_WANDB_RESUME=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --bucket)
      CLI_GCS_BUCKET="${2:?missing value for --bucket}"
      shift 2
      ;;
    --gcs-export)
      CLI_GCS_EXPORT="${2:?missing value for --gcs-export}"
      shift 2
      ;;
    --gcs-restore-if-missing)
      CLI_GCS_RESTORE_IF_MISSING="${2:?missing value for --gcs-restore-if-missing}"
      shift 2
      ;;
    --resume)
      CLI_RESUME_CHECKPOINT="${2:?missing value for --resume}"
      shift 2
      ;;
    --gpus)
      CLI_GPUS="${2:?missing value for --gpus}"
      shift 2
      ;;
    --expected-gpu-name)
      CLI_EXPECTED_GPU_NAME="${2:?missing value for --expected-gpu-name}"
      shift 2
      ;;
    --wandb-project)
      CLI_WANDB_PROJECT="${2:?missing value for --wandb-project}"
      shift 2
      ;;
    --wandb-entity)
      CLI_WANDB_ENTITY="${2:?missing value for --wandb-entity}"
      shift 2
      ;;
    --wandb-name)
      CLI_WANDB_NAME="${2:?missing value for --wandb-name}"
      shift 2
      ;;
    --wandb-group)
      CLI_WANDB_GROUP="${2:?missing value for --wandb-group}"
      shift 2
      ;;
    --wandb-id)
      CLI_WANDB_ID="${2:?missing value for --wandb-id}"
      shift 2
      ;;
    --wandb-resume)
      CLI_WANDB_RESUME="${2:?missing value for --wandb-resume}"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --)
      shift
      while [[ "$#" -gt 0 ]]; do
        POSITIONAL_ARGS+=("$1")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      print_usage
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"

case "$#" in
  2)
    TRAIN_DATA="$1"
    VAL_DATA=""
    OUTPUT_DIR="$2"
    ;;
  3)
    TRAIN_DATA="$1"
    VAL_DATA="$2"
    OUTPUT_DIR="$3"
    ;;
  *)
    print_usage
    exit 1
    ;;
esac

if [[ ! -f "${TRAIN_DATA}" ]]; then
  echo "Train data file not found: ${TRAIN_DATA}" >&2
  exit 1
fi

if [[ -n "${VAL_DATA}" && ! -f "${VAL_DATA}" ]]; then
  echo "Validation data file not found: ${VAL_DATA}" >&2
  exit 1
fi

if [[ -f "${DEFAULT_GCS_CREDENTIALS}" ]]; then
  export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-${DEFAULT_GCS_CREDENTIALS}}"
fi

export WANDB_API_KEY="${WANDB_API_KEY:-${DEFAULT_WANDB_API_KEY}}"
export WANDB_ENTITY="${CLI_WANDB_ENTITY:-${WANDB_ENTITY:-${DEFAULT_WANDB_ENTITY}}}"
export WANDB_PROJECT="${CLI_WANDB_PROJECT:-${WANDB_PROJECT:-${DEFAULT_WANDB_PROJECT}}}"
export GCS_BUCKET="${CLI_GCS_BUCKET:-${GCS_BUCKET:-${DEFAULT_GCS_BUCKET}}}"

if [[ -z "${WANDB_API_KEY}" ]]; then
  echo "WANDB_API_KEY is not set. Export it before running or submit with sbatch --export=ALL." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

detect_allocated_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
    echo "${#visible_devices[@]}"
    return
  fi
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    if [[ "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
      echo "${SLURM_GPUS_ON_NODE}"
      return
    fi
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d '[:space:]'
    return
  fi
  echo ""
}

query_visible_gpu_inventory() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  local query_output status
  query_output="$(nvidia-smi --query-gpu=name,gpu_bus_id --format=csv,noheader 2>&1)"
  status=$?
  if [[ ${status} -ne 0 || "${query_output}" == *"Unable to determine the device handle"* || "${query_output}" == *"Unknown Error"* ]]; then
    echo "${query_output}" >&2
    echo "GPU preflight failed before Python startup: nvidia-smi reported an unhealthy or inaccessible GPU in the current allocation." >&2
    echo "Release this job and request a fresh allocation before rerunning training." >&2
    exit 1
  fi
  printf '%s\n' "${query_output}"
}

validate_gpu_inventory() {
  local expected_name="$1"
  local expected_gpus="$2"
  local gpu_inventory
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; skipping GPU model validation." >&2
    return
  fi

  mapfile -t gpu_inventory < <(query_visible_gpu_inventory)
  if [[ "${#gpu_inventory[@]}" -eq 0 ]]; then
    echo "Unable to detect visible GPU names with nvidia-smi." >&2
    exit 1
  fi
  if [[ "${#gpu_inventory[@]}" -ne "${expected_gpus}" ]]; then
    echo "GPU mismatch: launcher expects ${expected_gpus} visible GPUs, but nvidia-smi reports ${#gpu_inventory[@]} healthy GPUs in this allocation." >&2
    printf 'nvidia-smi visible GPUs:\n%s\n' "${gpu_inventory[*]}" >&2
    exit 1
  fi

  if [[ -z "${expected_name}" ]]; then
    return
  fi

  local gpu_name
  for gpu_name in "${gpu_inventory[@]}"; do
    if [[ "${gpu_name}" != *"${expected_name}"* ]]; then
      echo "Expected GPUs matching '${expected_name}', but found '${gpu_name}'." >&2
      exit 1
    fi
  done
}

ALLOCATED_GPUS="$(detect_allocated_gpus)"
GPUS="${CLI_GPUS:-${GPUS:-${ALLOCATED_GPUS:-4}}}"
EXPECTED_GPU_NAME="${CLI_EXPECTED_GPU_NAME:-${EXPECTED_GPU_NAME:-}}"

if [[ -n "${ALLOCATED_GPUS}" && "${ALLOCATED_GPUS}" != "${GPUS}" ]]; then
  echo "GPU mismatch: launcher is configured for GPUS=${GPUS}, but the allocation exposes ${ALLOCATED_GPUS} GPUs." >&2
  echo "Submit with matching sbatch GPU args or override GPUS explicitly only when the allocation matches." >&2
  exit 1
fi

validate_gpu_inventory "${EXPECTED_GPU_NAME}" "${GPUS}"

FIRST_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
if [[ "${FIRST_GPU_NAME}" == *"V100"* ]]; then
  DEFAULT_SEQ_LEN=512
  DEFAULT_GLOBAL_BATCH_SIZE=4
else
  DEFAULT_SEQ_LEN=2048
  DEFAULT_GLOBAL_BATCH_SIZE=32
fi

SEQ_LEN="${SEQ_LEN:-${DEFAULT_SEQ_LEN}}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${DEFAULT_GLOBAL_BATCH_SIZE}}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-10000}"
LR="${LR:-2e-5}"
MIN_LR="${MIN_LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.98}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SAVE_EVERY="${SAVE_EVERY:-25}"
EVAL_EVERY="${EVAL_EVERY:-500}"
KEEP_MILESTONES="${KEEP_MILESTONES:-4}"
LOG_EVERY="${LOG_EVERY:-1}"
DTYPE="${DTYPE:-auto}"
SIMAMBA_CHUNK_SIZE="${SIMAMBA_CHUNK_SIZE:-64}"
SIMAMBA_DT_LIMIT_MIN="${SIMAMBA_DT_LIMIT_MIN:-0.001}"
SIMAMBA_DT_LIMIT_MAX="${SIMAMBA_DT_LIMIT_MAX:-0.1}"
SIMAMBA_A_MAX="${SIMAMBA_A_MAX:-auto}"
SIMAMBA_OUTPROJ_NORM="${SIMAMBA_OUTPROJ_NORM:-1}"
if [[ -n "${SIMAMBA_RECOMPUTE_CHUNK_SIZE:-}" ]]; then
  SIMAMBA_RECOMPUTE_CHUNK_SIZE="${SIMAMBA_RECOMPUTE_CHUNK_SIZE}"
elif [[ "${MICRO_BATCH_SIZE}" -eq 1 ]]; then
  SIMAMBA_RECOMPUTE_CHUNK_SIZE=512
else
  SIMAMBA_RECOMPUTE_CHUNK_SIZE="${SIMAMBA_CHUNK_SIZE}"
fi
COMPILE="${COMPILE:-0}"
RESUME_CHECKPOINT="${CLI_RESUME_CHECKPOINT:-${RESUME_CHECKPOINT:-}}"
GCS_EXPORT="${CLI_GCS_EXPORT:-${GCS_EXPORT:-1}}"
GCS_RUN_PREFIX="${GCS_RUN_PREFIX:-$(basename "${OUTPUT_DIR}")}"
GCS_EXPORT_POLL_SECS="${GCS_EXPORT_POLL_SECS:-15}"
GCS_RESTORE_IF_MISSING="${CLI_GCS_RESTORE_IF_MISSING:-${GCS_RESTORE_IF_MISSING:-1}}"
export GPUS EXPECTED_GPU_NAME GCS_BUCKET GCS_EXPORT GCS_RUN_PREFIX GCS_EXPORT_POLL_SECS GCS_RESTORE_IF_MISSING

if [[ "${MICRO_BATCH_SIZE}" -lt 1 ]]; then
  echo "MICRO_BATCH_SIZE must be >= 1, got ${MICRO_BATCH_SIZE}." >&2
  exit 1
fi

if [[ "${SIMAMBA_CHUNK_SIZE}" -lt 1 || "${SIMAMBA_RECOMPUTE_CHUNK_SIZE}" -lt 1 ]]; then
  echo "SIMAMBA_CHUNK_SIZE and SIMAMBA_RECOMPUTE_CHUNK_SIZE must both be >= 1." >&2
  exit 1
fi

if [[ "${SIMAMBA_OUTPROJ_NORM}" != "0" && "${SIMAMBA_OUTPROJ_NORM}" != "1" ]]; then
  echo "SIMAMBA_OUTPROJ_NORM must be 0 or 1, got ${SIMAMBA_OUTPROJ_NORM}." >&2
  exit 1
fi

if [[ "${GLOBAL_BATCH_SIZE}" -lt 1 ]]; then
  echo "GLOBAL_BATCH_SIZE must be >= 1, got ${GLOBAL_BATCH_SIZE}." >&2
  exit 1
fi

if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * GPUS) != 0 )); then
  echo "Invalid batch shape: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by MICRO_BATCH_SIZE * GPUS = $(( MICRO_BATCH_SIZE * GPUS ))." >&2
  exit 1
fi

GRAD_ACCUM_STEPS=$(( GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * GPUS) ))

if [[ "${SEQ_LEN}" -ge 2048 && "${MICRO_BATCH_SIZE}" -gt 1 ]]; then
  echo "Warning: Simamba backward stores large per-token state histories; SEQ_LEN=${SEQ_LEN} with MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} is likely to OOM on 48 GiB A6000 GPUs." >&2
  echo "Warning: The launcher default MICRO_BATCH_SIZE=1 is intentional; raise it only after validating memory on your target allocation." >&2
fi

if [[ -z "${RESUME_CHECKPOINT}" ]]; then
  if [[ -f "${OUTPUT_DIR}/latest/trainer.pt" ]]; then
    RESUME_CHECKPOINT="${OUTPUT_DIR}/latest/trainer.pt"
  elif [[ "${GCS_RESTORE_IF_MISSING}" == "1" && "${GCS_EXPORT}" == "1" ]]; then
    log_phase "Local latest checkpoint missing; attempting synchronous restore from GCS."
    RESTORE_JSON="$("${PYTHON_BIN}" scripts/fetch_latest_checkpoint_from_gcs.py --bucket "${GCS_BUCKET}" --output-dir "${OUTPUT_DIR}")"
    echo "${RESTORE_JSON}"
    RESTORE_STATUS="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["status"])' "${RESTORE_JSON}")"
    if [[ "${RESTORE_STATUS}" == "restored" ]]; then
      RESUME_CHECKPOINT="${OUTPUT_DIR}/latest/trainer.pt"
      log_phase "Resuming from restored GCS milestone archive; optimizer state is not included in remote step_* exports."
    fi
  fi
fi

if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${GPUS}" -gt 0 ]]; then
  THREADS_PER_RANK=$(( SLURM_CPUS_PER_TASK / GPUS ))
  if [[ "${THREADS_PER_RANK}" -lt 1 ]]; then
    THREADS_PER_RANK=1
  fi
else
  THREADS_PER_RANK=1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS_PER_RANK}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"

WANDB_ROOT="${WANDB_ROOT:-${OUTPUT_DIR}/.wandb}"
export WANDB_DIR="${WANDB_DIR:-${WANDB_ROOT}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_ROOT}/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_ROOT}/config}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-auto}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

WANDB_NAME="${CLI_WANDB_NAME:-${WANDB_NAME:-simamba_130m_$(date +%Y%m%d_%H%M%S)}}"
WANDB_GROUP="${CLI_WANDB_GROUP:-${WANDB_GROUP:-simamba_130m}}"

if [[ "${GCS_EXPORT}" == "1" ]]; then
  if [[ -z "${GCS_BUCKET:-}" ]]; then
    echo "GCS_EXPORT=1 requires GCS_BUCKET to be set." >&2
    exit 1
  fi
fi

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
  --lr "${LR}"
  --min-lr "${MIN_LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --betas "${BETA1}" "${BETA2}"
  --grad-clip "${GRAD_CLIP}"
  --warmup-steps "${WARMUP_STEPS}"
  --log-every "${LOG_EVERY}"
  --save-every "${SAVE_EVERY}"
  --eval-every "${EVAL_EVERY}"
  --keep-milestones "${KEEP_MILESTONES}"
  --dtype "${DTYPE}"
  --simamba-backend triton
  --simamba-chunk-size "${SIMAMBA_CHUNK_SIZE}"
  --simamba-recompute-chunk-size "${SIMAMBA_RECOMPUTE_CHUNK_SIZE}"
  --simamba-dt-limit "${SIMAMBA_DT_LIMIT_MIN}" "${SIMAMBA_DT_LIMIT_MAX}"
  --wandb
  --wandb-project "${WANDB_PROJECT}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-name "${WANDB_NAME}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-console "${WANDB_CONSOLE}"
)

if [[ "${SIMAMBA_A_MAX}" != "auto" ]]; then
  ARGS+=(--simamba-a-max "${SIMAMBA_A_MAX}")
fi

if [[ "${SIMAMBA_OUTPROJ_NORM}" == "1" ]]; then
  ARGS+=(--simamba-outproj-norm)
fi

if [[ -n "${CLI_WANDB_ID}" ]]; then
  ARGS+=(--wandb-id "${CLI_WANDB_ID}")
fi

if [[ -n "${CLI_WANDB_RESUME}" ]]; then
  ARGS+=(--wandb-resume "${CLI_WANDB_RESUME}")
fi

if [[ -n "${VAL_DATA}" ]]; then
  ARGS+=(--val-data "${VAL_DATA}")
fi

if [[ "${COMPILE}" == "1" ]]; then
  ARGS+=(--compile)
fi

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi

log_phase "Launching Simamba 130M-ish pretraining"
echo "  train_data=${TRAIN_DATA}"
echo "  val_data=${VAL_DATA:-<none>}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  gpus=${GPUS}"
echo "  expected_gpu=${EXPECTED_GPU_NAME}"
echo "  seq_len=${SEQ_LEN}"
echo "  global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "  micro_batch_size=${MICRO_BATCH_SIZE}"
echo "  grad_accum_steps=${GRAD_ACCUM_STEPS}"
echo "  simamba_chunk_size=${SIMAMBA_CHUNK_SIZE}"
echo "  simamba_recompute_chunk_size=${SIMAMBA_RECOMPUTE_CHUNK_SIZE}"
echo "  simamba_dt_limit=(${SIMAMBA_DT_LIMIT_MIN}, ${SIMAMBA_DT_LIMIT_MAX})"
echo "  simamba_a_max=${SIMAMBA_A_MAX}"
echo "  simamba_outproj_norm=${SIMAMBA_OUTPROJ_NORM}"
echo "  max_steps=${MAX_STEPS}"
echo "  lr=${LR}"
echo "  min_lr=${MIN_LR}"
echo "  warmup_steps=${WARMUP_STEPS}"
echo "  weight_decay=${WEIGHT_DECAY}"
echo "  betas=(${BETA1}, ${BETA2})"
echo "  grad_clip=${GRAD_CLIP}"
echo "  log_every=${LOG_EVERY}"
echo "  save_every=${SAVE_EVERY}"
echo "  keep_milestones=${KEEP_MILESTONES}"
echo "  resume_checkpoint=${RESUME_CHECKPOINT:-<none>}"
echo "  gcs_restore_if_missing=${GCS_RESTORE_IF_MISSING}"
echo "  wandb_entity=${WANDB_ENTITY}"
echo "  wandb_project=${WANDB_PROJECT}"
echo "  wandb_dir=${WANDB_DIR}"
echo "  wandb_console=${WANDB_CONSOLE}"
echo "  gcs_export=${GCS_EXPORT}"
if [[ "${GCS_EXPORT}" == "1" ]]; then
  echo "  gcs_bucket=${GCS_BUCKET}"
  echo "  gcs_prefix=${GCS_PREFIX:-<none>}"
  echo "  gcs_run_prefix=${GCS_RUN_PREFIX}"
fi

if [[ "${SIMAMBA_RECOMPUTE_CHUNK_SIZE}" -lt "${SEQ_LEN}" ]]; then
  echo "Note: SIMAMBA_RECOMPUTE_CHUNK_SIZE=${SIMAMBA_RECOMPUTE_CHUNK_SIZE} is below SEQ_LEN=${SEQ_LEN}; this mainly matters only if a fallback/stateful Simamba path is used." >&2
fi

log_phase "Python environment preflight begin"
"${PYTHON_BIN}" - <<'PY'
import importlib
import os
import sys
import torch

required_modules = ["wandb"]
if os.environ.get("GCS_EXPORT") == "1":
    required_modules.append("google.cloud.storage")

for module_name in required_modules:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Required module '{module_name}' is missing from the repo venv. "
            f"Install the repo train extras with: {sys.executable} -m pip install -e '.[train]'"
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

expected_gpus = int(os.environ.get("GPUS", "1"))
if torch.cuda.device_count() != expected_gpus:
    raise SystemExit(
        f"Expected {expected_gpus} visible GPUs for this launch, but torch sees {torch.cuda.device_count()}."
    )
PY
log_phase "Python environment preflight end"

EXPORT_PID=""
EXPORT_STOP_FILE="${OUTPUT_DIR}/.gcs_export/stop"
EXPORT_LOG="${OUTPUT_DIR}/checkpoint_export.log"
rm -f "${EXPORT_STOP_FILE}"

if [[ "${GCS_EXPORT}" == "1" ]]; then
  export GCS_RUN_PREFIX
  export GCS_EXPORT_POLL_SECS
  export GCS_STOP_FILE="${EXPORT_STOP_FILE}"
  log_phase "Checkpoint export helper start"
  "${PYTHON_BIN}" scripts/export_checkpoints_async.py "${OUTPUT_DIR}" >> "${EXPORT_LOG}" 2>&1 &
  EXPORT_PID="$!"
  log_phase "Checkpoint export helper running pid=${EXPORT_PID}"
fi

TRAIN_PID=""
forward_training_stop() {
  local reason="$1"
  if [[ -n "${TRAIN_PID}" ]]; then
    log_phase "Launcher received ${reason}; forwarding SIGTERM to training pid=${TRAIN_PID}"
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
  fi
}

trap 'forward_training_stop HUP' HUP
trap 'forward_training_stop INT' INT
trap 'forward_training_stop TERM' TERM

set +e
if [[ "${GPUS}" == "1" ]]; then
  log_phase "Training process start mode=single_gpu"
  "${PYTHON_BIN}" scripts/train_simamba_lm.py "${ARGS[@]}" &
else
  log_phase "Training process start mode=ddp nproc_per_node=${GPUS}"
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${GPUS}" scripts/train_simamba_lm.py "${ARGS[@]}" &
fi
TRAIN_PID=$!
wait "${TRAIN_PID}"
TRAIN_RC=$?
TRAIN_PID=""
set -e
trap - HUP INT TERM
log_phase "Training process end rc=${TRAIN_RC}"

EXPORT_RC=0
if [[ -n "${EXPORT_PID}" ]]; then
  touch "${EXPORT_STOP_FILE}"
  set +e
  log_phase "Waiting for checkpoint export helper pid=${EXPORT_PID}"
  wait "${EXPORT_PID}"
  EXPORT_RC=$?
  set -e
  log_phase "Checkpoint export helper end rc=${EXPORT_RC}"
fi

if [[ "${TRAIN_RC}" -ne 0 ]]; then
  echo "Training failed with exit code ${TRAIN_RC}." >&2
  exit "${TRAIN_RC}"
fi

if [[ "${EXPORT_RC}" -ne 0 ]]; then
  echo "Checkpoint export failed with exit code ${EXPORT_RC}. See ${EXPORT_LOG}." >&2
  exit "${EXPORT_RC}"
fi
