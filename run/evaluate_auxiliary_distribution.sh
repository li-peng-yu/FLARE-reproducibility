#!/usr/bin/env bash

source "$(dirname -- "$0")/common.sh"

METHOD="${1:?Usage: evaluate_distribution.sh METHOD}"
RESULT_ROOT="${RESULT_ROOT:-outputs/auxiliary_distribution}"
SEGMENTS="${SEGMENTS:-1 2}"
# Separate saved-frame single-segment audit used by metric diagnostics.
ROLLOUT_TARGET_ROOT= to run a saved-frame audit instead.
ROLLOUT_TARGET_ROOT=""
DURATION_NS="${DURATION_NS-}"
TARGET_ARGS=()
if [[ -n "${ROLLOUT_TARGET_ROOT}" ]]; then
  TARGET_ARGS=(--rollout-target-root "${ROLLOUT_TARGET_ROOT}")
elif [[ -n "${DURATION_NS}" ]]; then
  TARGET_ARGS=(--duration-ns "${DURATION_NS}")
fi
GROUP_ARGS=()
if (( ${MAX_GROUPS:-0} > 0 )); then
  GROUP_ARGS=(--max-groups "${MAX_GROUPS}")
fi
mkdir -p "${RESULT_ROOT}"

if [[ "${METHOD}" == flare ]]; then
  exec "${PYTHON_BIN}" scripts/evaluate_skx_x5_same_condition_distribution.py \
    --checkpoint "${CHECKPOINT:-${FLARE_CHECKPOINT_ROOT}/flare/core_seed78.pt}" \
    --stats "${STATS:-configs/stats/dataset_stats_both.json}" \
    --output-root "${RESULT_ROOT}/flare" \
    --state ema \
    --device "${DEVICE}" \
    --segments ${SEGMENTS} \
    "${TARGET_ARGS[@]}" \
    "${GROUP_ARGS[@]}" \
    --num-model "${NUM_MODEL:-5}" \
    --score-num-model "${SCORE_NUM_MODEL:-5}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --ode-steps "${ODE_STEPS:-10}" \
    --bootstrap "${BOOTSTRAP:-5000}" \
    --seed "${EVAL_SEED:-208160700}"
fi

case "${METHOD}" in
  poseidon_t|cno_fm|dpot_ti|mpp_avit_ti) PRECISION_DEFAULT=fp32 ;;
  pdearena_unet|le_pde) PRECISION_DEFAULT=bf16 ;;
  *) echo "Unknown method: ${METHOD}" >&2; exit 2 ;;
esac

exec "${NATIVE_PYTHON}" external_baselines/evaluate_x5_native_same_condition_distribution.py \
  --method "${METHOD}" \
  --config configs/base/x5.yaml \
  --checkpoint "${CHECKPOINT:-${FLARE_CHECKPOINT_ROOT}/baselines/${METHOD}.pt}" \
  --output-root "${RESULT_ROOT}/${METHOD}" \
  --precision "${PRECISION:-${PRECISION_DEFAULT}}" \
  --device "${DEVICE}" \
  --segments ${SEGMENTS} \
  "${TARGET_ARGS[@]}" \
  "${GROUP_ARGS[@]}" \
  --bootstrap "${BOOTSTRAP:-5000}" \
  --seed "${EVAL_SEED:-208160700}"
