#!/usr/bin/env bash

source "$(dirname -- "$0")/common.sh"

exec "${PYTHON_BIN}" scripts/render_x5_paper_qualitative.py \
  --root "${OUTPUT_ROOT:-outputs}" \
  --output-dir "${QUALITATIVE_OUTPUT:-outputs/qualitative}" \
  --stage1 "${FLARE_CHECKPOINT_ROOT}/flare/core_seed78.pt" \
  --mixed "${FLARE_CHECKPOINT_ROOT}/flare/stage2_mixed_seed78.pt" \
  --cartesian "${FLARE_CHECKPOINT_ROOT}/flare/cartesian_seed78.pt" \
  --fno "${FLARE_CHECKPOINT_ROOT}/flare/fno_seed78.pt" \
  --device "${DEVICE}" \
  --seed "${EVAL_SEED:-20260822}"
