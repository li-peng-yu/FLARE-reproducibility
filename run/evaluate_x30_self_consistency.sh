#!/usr/bin/env bash

source "$(dirname -- "$0")/common.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/x30_self_consistency_input}"
mkdir -p "${OUTPUT_ROOT}"

"${PYTHON_BIN}" scripts/evaluate_skx_x5_same_condition_distribution.py \
  --checkpoint "${CHECKPOINT:-${FLARE_CHECKPOINT_ROOT}/flare/core_seed78.pt}" \
  --stats "${STATS:-configs/stats/dataset_stats_both.json}" \
  --evaluation-config "${EVAL_CONFIG:-configs/data/x30_evaluation.yaml}" \
  --output-root "${OUTPUT_ROOT}" \
  --state ema \
  --device "${DEVICE}" \
  --repeat-dataset skx_bt_165base_x30_sharedrelax_20260813 \
  --repeats-per-group 30 \
  --segments 1 2 \
  --num-model "${NUM_MODEL:-5}" \
  --batch-size "${BATCH_SIZE:-5}" \
  --ode-steps "${ODE_STEPS:-10}" \
  --bootstrap "${BOOTSTRAP:-5000}" \
  --seed "${EVAL_SEED:-208180700}"

exec "${PYTHON_BIN}" scripts/analyze_x30_mumax_self_score.py \
  --root "${OUTPUT_ROOT}" \
  --output "${ANALYSIS_OUTPUT:-outputs/analysis/distribution_sensitivity_observables/x30_mumax_self_score}" \
  --bootstrap "${ANALYSIS_BOOTSTRAP:-10000}" \
  --seed "${ANALYSIS_SEED:-20260822}"
