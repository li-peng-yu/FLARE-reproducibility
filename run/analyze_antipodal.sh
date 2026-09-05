#!/usr/bin/env bash

source "$(dirname -- "$0")/common.sh"

exec "${PYTHON_BIN}" scripts/analyze_x5_antipodal_statistics.py \
  --condition-root "${CONDITION_ROOT:-outputs/distribution/flare/conditions}" \
  --output-dir "${OUTPUT_ROOT:-outputs/analysis/antipodal_statistics}" \
  --bootstrap "${BOOTSTRAP:-10000}" \
  --seed "${EVAL_SEED:-208240900}"
