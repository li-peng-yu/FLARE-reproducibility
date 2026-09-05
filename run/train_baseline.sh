#!/usr/bin/env bash

source "$(dirname -- "$0")/common.sh"

METHOD="${1:?Usage: train_baseline.sh METHOD}"
STEPS="${STEPS:-50000}"
OUTPUT_TAG="${OUTPUT_TAG:-baseline_training}"
SAVE_EVERY="${SAVE_EVERY:-${STEPS}}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"

case "${METHOD}" in
  poseidon_t) BATCH_SIZE_DEFAULT=4; ACCUM_DEFAULT=2; PRECISION_DEFAULT=fp32 ;;
  cno_fm) BATCH_SIZE_DEFAULT=2; ACCUM_DEFAULT=4; PRECISION_DEFAULT=fp32 ;;
  dpot_ti) BATCH_SIZE_DEFAULT=8; ACCUM_DEFAULT=1; PRECISION_DEFAULT=fp32 ;;
  mpp_avit_ti) BATCH_SIZE_DEFAULT=5; ACCUM_DEFAULT=1; PRECISION_DEFAULT=fp32 ;;
  pdearena_unet) BATCH_SIZE_DEFAULT=4; ACCUM_DEFAULT=2; PRECISION_DEFAULT=bf16 ;;
  le_pde) BATCH_SIZE_DEFAULT=4; ACCUM_DEFAULT=2; PRECISION_DEFAULT=bf16 ;;
  *) echo "Unknown baseline: ${METHOD}" >&2; exit 2 ;;
esac

BATCH_SIZE="${BATCH_SIZE:-${BATCH_SIZE_DEFAULT}}"
GRAD_ACCUM="${GRAD_ACCUM:-${ACCUM_DEFAULT}}"
PRECISION="${PRECISION:-${PRECISION_DEFAULT}}"
OPTIMIZER_RECIPE="${OPTIMIZER_RECIPE:-standardized}"
if [[ "${METHOD}" == mpp_avit_ti && "${OPTIMIZER_RECIPE}" == standardized ]]; then
  OPTIMIZER_RECIPE=author_mpp
fi

INIT_ARGS=()
case "${METHOD}" in
  poseidon_t|cno_fm|dpot_ti|mpp_avit_ti) INIT_ARGS=(--pretrained) ;;
esac
if [[ "${NO_CHECKPOINT:-0}" == 1 ]]; then
  INIT_ARGS+=(--no-checkpoint)
fi

exec "${NATIVE_PYTHON}" external_baselines/x5_author_native.py train \
  --method "${METHOD}" \
  --config configs/base/x5.yaml \
  --output-dir "outputs/${OUTPUT_TAG}/${METHOD}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --lr "${LEARNING_RATE:-1.0e-4}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --optimizer-recipe "${OPTIMIZER_RECIPE}" \
  --precision "${PRECISION}" \
  --log-every "${LOG_EVERY:-1}" \
  --save-every "${SAVE_EVERY}" \
  "${INIT_ARGS[@]}"
