#!/usr/bin/env bash

source "$(dirname -- "$0")/common.sh"

MODE="${1:?Usage: neuralmag.sh MODE [SHARD_INDEX]}"
CONFIG="${CONFIG:-configs/base/x5.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/neuralmag_x5}"
CHECKPOINT="${CHECKPOINT:-${FLARE_CHECKPOINT_ROOT}/baselines/neuralmag_x5.pt}"
mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/NeuralMAG:${PYTHONPATH}"

case "${MODE}" in
  validate)
    exec "${PYTHON_BIN}" external_baselines/neuralmag_x5.py validate-demag \
      --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
      --output "${OUTPUT_ROOT}/demag_validation.json" --samples "${SAMPLES:-2}"
    ;;
  finetune)
    # Paper-eligible NeuralMAG uses
    # the exact verified 50,000-step recipe below. The former 6,000-step pilot
    # and its shorter/default optimization recipe must not be used.
    exec "${PYTHON_BIN}" external_baselines/neuralmag_x5.py finetune-demag \
      --config "${CONFIG}" \
      --checkpoint "${FINETUNE_INPUT:-${FLARE_CHECKPOINT_ROOT}/pretrained/neuralmag/author_model.pt}" \
      --output "${CHECKPOINT}" \
      --steps "${FINETUNE_STEPS:-50000}" \
      --batch-size "${FINETUNE_BATCH_SIZE:-4}" \
      --lr "${FINETUNE_LR:-3e-5}" \
      --weight-decay "${FINETUNE_WEIGHT_DECAY:-1e-5}" \
      --physical-weight "${FINETUNE_PHYSICAL_WEIGHT:-10.0}" \
      --state-noise-std "${FINETUNE_STATE_NOISE_STD:-0.1}" \
      --validation-samples "${FINETUNE_VALIDATION_SAMPLES:-5}"
    ;;
  score)
    SHARD_INDEX="${2:?Usage: neuralmag.sh score SHARD_INDEX}"
    EXTRA_ARGS=()
    (( ${MAX_GROUPS:-0} > 0 )) && EXTRA_ARGS+=(--max-groups "${MAX_GROUPS}")
    [[ "${EXACT_DEMAG:-0}" == 1 ]] && EXTRA_ARGS+=(--exact-demag)
    [[ "${NO_THERMAL:-0}" == 1 ]] && EXTRA_ARGS+=(--no-thermal)
    [[ "${SAMPLES_ONLY:-0}" == 1 ]] && EXTRA_ARGS+=(--samples-only)
    # Saved-frame endpoints are used for the auxiliary comparison.
    ROLLOUT_TARGET_ROOT="${ROLLOUT_TARGET_ROOT-}"
    if [[ -n "${ROLLOUT_TARGET_ROOT}" ]]; then
      EXTRA_ARGS+=(--rollout-target-root "${ROLLOUT_TARGET_ROOT}")
    elif [[ -n "${DURATION_NS-}" ]]; then
      EXTRA_ARGS+=(--duration-ns "${DURATION_NS}")
    fi
    exec "${PYTHON_BIN}" external_baselines/neuralmag_x5.py score-shard \
      --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
      --output-root "${SCORE_OUTPUT:-outputs/auxiliary_distribution/neuralmag_x5}" \
      --shard-index "${SHARD_INDEX}" --shard-count "${SHARD_COUNT:-11}" \
      --segments ${SEGMENTS:-2} \
      --sot-scale "${SOT_SCALE:-1.0}" --dmi-scale "${DMI_SCALE:-1.0}" \
      --dt-s "${DT_S:-1e-13}" \
      --draws-per-anchor "${DRAWS_PER_ANCHOR:-1}" \
      --draw-batch-size "${DRAW_BATCH_SIZE:-0}" \
      --bootstrap "${BOOTSTRAP:-5000}" \
      "${EXTRA_ARGS[@]}"
    ;;
  merge)
    exec "${PYTHON_BIN}" external_baselines/neuralmag_x5.py merge \
      --output-root "${SCORE_OUTPUT:-outputs/auxiliary_distribution/neuralmag_x5}" \
      --checkpoint "${CHECKPOINT}" \
      --shard-count "${SHARD_COUNT:-11}" \
      --expected-conditions "${EXPECTED_CONDITIONS:-33}" \
      --bootstrap "${BOOTSTRAP:-5000}" \
      --seed "${EVAL_SEED:-208160830}"
    ;;
  benchmark)
    exec "${PYTHON_BIN}" external_baselines/neuralmag_x5.py benchmark \
      --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
      --output "${TIMING_OUTPUT:-${OUTPUT_ROOT}/neuralmag_x5_5ns_timing.json}" \
      --horizon-ns "${HORIZON_NS:-5}" \
      --rollout-target-root "${ROLLOUT_TARGET_ROOT:-outputs/matched_5ns_rollout_targets}" \
      --batch-size "${BATCH_SIZE:-5}" --repeats "${REPEATS:-1}" \
      --dt-s "${DT_S:-1e-13}"
    ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac
