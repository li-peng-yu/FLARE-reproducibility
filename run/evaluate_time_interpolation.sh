#!/usr/bin/env bash

source "$(dirname -- "$0")/common.sh"

CHECKPOINT="${1:?Usage: evaluate_time_interpolation.sh CHECKPOINT LABEL}"
LABEL="${2:?Usage: evaluate_time_interpolation.sh CHECKPOINT LABEL}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/time_interpolation}"
mkdir -p "${OUTPUT_ROOT}"

exec "${PYTHON_BIN}" scripts/evaluate_x5_single_segment.py \
  --checkpoint "${CHECKPOINT}" --label "${LABEL}" \
  --output "${OUTPUT_ROOT}/${LABEL}.csv" --state ema --device "${DEVICE}" \
  --ode-steps "${ODE_STEPS:-10}" --method "${METHOD:-heun}" \
  --batch-size "${BATCH_SIZE:-32}" --draws 4 \
  --durations-ns 1.0 1.5 2.0 2.5 3.0 4.0 5.0 \
  --max-cases-per-duration "${MAX_CASES_PER_DURATION:-256}" \
  --seed "${EVAL_SEED:-20260814}"
