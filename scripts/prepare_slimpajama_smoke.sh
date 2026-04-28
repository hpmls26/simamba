#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --job-name=SlimPajamaPrepSmoke
#SBATCH --output=/insomnia001/home/ssb2234/logs/%x-%j.out
#SBATCH --error=/insomnia001/home/ssb2234/logs/%x-%j.err
#SBATCH -c 2
#SBATCH --time=0-02:00
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
mkdir -p /insomnia001/home/ssb2234/logs

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

for module_name in ("datasets", "zstandard"):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Required module '{module_name}' is missing from the repo venv. "
            f"Install it with: {sys.executable} -m pip install -e '.[train]' "
            "or minimally: "
            f"{sys.executable} -m pip install -U datasets zstandard"
        ) from exc

print(f"python={sys.executable}")
PY

# Prepare a small bounded SlimPajama subset suitable for smoke tests.
# 
# Usage:
#   bash scripts/prepare_slimpajama_smoke.sh /path/output_dir

DEFAULT_OUT_DIR="/insomnia001/home/ssb2234/slimpajama_smoke"
OUT_DIR="${1:-${DEFAULT_OUT_DIR}}"
if [[ -z "${OUT_DIR}" ]]; then
  echo "usage: bash scripts/prepare_slimpajama_smoke.sh OUTPUT_DIR" >&2
  exit 1
fi

echo "[prepare-smoke] output_dir=${OUT_DIR}"
echo "[prepare-smoke] repo_root=${REPO_ROOT}"
echo "[prepare-smoke] python=${PYTHON_BIN}"

"${PYTHON_BIN}" scripts/prepare_slimpajama.py \
  --output-dir "${OUT_DIR}" \
  --train-tokens 10000000 \
  --val-tokens 1000000
