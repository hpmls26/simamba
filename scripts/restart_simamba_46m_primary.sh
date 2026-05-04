#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Reuse the W&B key from the existing training wrapper without duplicating it here.
if [[ -z "${WANDB_API_KEY:-}" && -f scripts/run_train_simamba_130m.sh ]]; then
  WANDB_API_KEY="$(
    sed -n "s/^DEFAULT_WANDB_API_KEY=['\"]\\{0,1\\}//p" scripts/run_train_simamba_130m.sh \
      | tail -n 1 \
      | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"
  )"
  if [[ -n "${WANDB_API_KEY}" ]]; then
    export WANDB_API_KEY
  fi
fi

exec .venv/bin/python scripts/train_simamba_lm.py \
  --resume outputs/simamba_2day_50m_seq128_100m_amax4_lr5e5_scalerfix_20260502_050811/latest/trainer.pt \
  --train-data data/slimpajama_100m_10m/train.bin \
  --val-data data/slimpajama_100m_10m/val.bin \
  --output-dir outputs/simamba_2day_50m_seq128_100m_amax4_lr5e5_scalerfix_20260502_050811 \
  --model-layer Simamba \
  --simamba-backend triton \
  --d-model 512 \
  --n-layer 12 \
  --vocab-size 50280 \
  --simamba-d-state 128 \
  --simamba-headdim 64 \
  --simamba-chunk-size 16 \
  --simamba-recompute-chunk-size 16 \
  --simamba-outproj-norm \
  --seq-len 128 \
  --global-batch-size 8 \
  --micro-batch-size 1 \
  --max-steps 97600 \
  --warmup-steps 1200 \
  --lr 5e-5 \
  --min-lr 5e-6 \
  --weight-decay 0.0 \
  --grad-clip 1.0 \
  --grad-scale-init 8192 \
  --skip-nonfinite-steps \
  --max-nonfinite-skips 1 \
  --eval-every 1000 \
  --eval-iters 100 \
  --save-every 5000 \
  --keep-milestones 3 \
  --dtype auto \
  --log-every 1 \
  --train-sampling random \
  --eval-sampling random \
  --wandb \
  --wandb-project simamba \
  --wandb-entity ssb2234-columbia \
  --wandb-group simamba_v100_2day_amax4
