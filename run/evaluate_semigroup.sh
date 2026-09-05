#!/usr/bin/env bash

source "$(dirname -- "$0")/common.sh"

CHECKPOINT="${1:?Usage: evaluate_semigroup.sh CHECKPOINT LABEL}"
LABEL="${2:?Usage: evaluate_semigroup.sh CHECKPOINT LABEL}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/semigroup_handoff}"
mkdir -p "${OUTPUT_ROOT}/${LABEL}"

exec "${PYTHON_BIN}" scripts/evaluate_x5_semigroup_handoff.py \
  --checkpoint "${CHECKPOINT}" --label "${LABEL}" \
  --output-dir "${OUTPUT_ROOT}/${LABEL}" --state ema --device "${DEVICE}" \
  --ode-steps "${ODE_STEPS:-10}" --batch-size "${BATCH_SIZE:-32}" \
  --max-semigroup "${MAX_SEMIGROUP:-512}" --max-handoff "${MAX_HANDOFF:-736}" \
  --draws "${DRAWS:-4}" --seed "${EVAL_SEED:-20260813}"
