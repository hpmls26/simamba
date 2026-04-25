#!/usr/bin/env bash
#SBATCH --account=edu
#SBATCH --job-name=SlimPajamaPrepSmoke
#SBATCH --output=/insomnia001/home/ssb2234/logs/%x-%j.out
#SBATCH --error=/insomnia001/home/ssb2234/logs/%x-%j.err
#SBATCH -c 2
#SBATCH --time=0-02:00
#SBATCH --mem-per-cpu=4G

set -euo pipefail

# Prepare a small bounded SlimPajama sample suitable for smoke tests.
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

.venv/bin/python scripts/prepare_slimpajama_sample.py \
  --output-dir "${OUT_DIR}" \
  --train-tokens 10000000 \
  --val-tokens 1000000
