#!/usr/bin/env bash
set -euo pipefail

# Prepare a small bounded SlimPajama sample suitable for smoke tests.
#
# Usage:
#   bash scripts/prepare_slimpajama_smoke.sh /path/output_dir

OUT_DIR="${1:-}"
if [[ -z "${OUT_DIR}" ]]; then
  echo "usage: bash scripts/prepare_slimpajama_smoke.sh OUTPUT_DIR" >&2
  exit 1
fi

.venv/bin/python scripts/prepare_slimpajama_sample.py \
  --output-dir "${OUT_DIR}" \
  --train-tokens 10000000 \
  --val-tokens 1000000
