#!/usr/bin/env bash

source "$(dirname -- "$0")/common.sh"

CHECKPOINT="${1:?Usage: evaluate_single_segment.sh CHECKPOINT LABEL}"
LABEL="${2:?Usage: evaluate_single_segment.sh CHECKPOINT LABEL}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/single_segment}"
mkdir -p "${OUTPUT_ROOT}"
EXTRA_ARGS=()
[[ -n "${METHOD:-}" ]] && EXTRA_ARGS+=(--method "${METHOD}")

exec "${PYTHON_BIN}" scripts/evaluate_x5_single_segment.py \
  --checkpoint "${CHECKPOINT}" --label "${LABEL}" \
  --output "${OUTPUT_ROOT}/${LABEL}_ode${ODE_STEPS:-10}.csv" \
  --state ema --device "${DEVICE}" --ode-steps "${ODE_STEPS:-10}" \
  --batch-size "${BATCH_SIZE:-32}" --draws "${DRAWS:-4}" \
  --seed "${EVAL_SEED:-20260813}" --max-cases "${MAX_CASES:-0}" \
  "${EXTRA_ARGS[@]}"
