#!/usr/bin/env bash

source "$(dirname -- "$0")/common.sh"

CHECKPOINT="${1:?Usage: evaluate_rollout.sh CHECKPOINT LABEL DRAW}"
LABEL="${2:?Usage: evaluate_rollout.sh CHECKPOINT LABEL DRAW}"
DRAW="${3:?Usage: evaluate_rollout.sh CHECKPOINT LABEL DRAW}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rollout/raw}"
mkdir -p "${OUTPUT_ROOT}"

exec "${PYTHON_BIN}" scripts/evaluate_skx_x5_stage2_rollout_arg.py \
  --checkpoint "${CHECKPOINT}" --label "${LABEL}" \
  --output "${OUTPUT_ROOT}/${LABEL}_draw${DRAW}.csv" \
  --state ema --device "${DEVICE}" --ode-steps "${ODE_STEPS:-10}" \
  --batch-size "${BATCH_SIZE:-64}" --draw-index "${DRAW}" \
  --seed "${EVAL_SEED:-20260812}" --range-start-ns 1.0 --range-end-ns 6.0
