#!/usr/bin/env bash

source "$(dirname -- "$0")/common.sh"

TARGET="${1:?Usage: evaluate_ood.sh TARGET [DRAW]}"
DRAW="${2:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ood_ring}"
mkdir -p "${OUTPUT_ROOT}/single_segment" "${OUTPUT_ROOT}/rollout/raw"

case "${TARGET}" in
  flare-single)
    exec "${PYTHON_BIN}" scripts/evaluate_x5_single_segment.py \
      --checkpoint "${FLARE_CHECKPOINT_ROOT}/flare/ood_flare_seed78.pt" --label ood_scfm \
      --output "${OUTPUT_ROOT}/single_segment/ood_scfm_ode10.csv" \
      --state ema --device "${DEVICE}" --ode-steps 10 --method heun \
      --batch-size "${BATCH_SIZE:-32}" --draws 4 --seed "${EVAL_SEED:-20260814}"
    ;;
  fno-single)
    exec "${PYTHON_BIN}" scripts/evaluate_x5_single_segment.py \
      --checkpoint "${FLARE_CHECKPOINT_ROOT}/flare/ood_fno_seed78.pt" --label ood_fno \
      --output "${OUTPUT_ROOT}/single_segment/ood_fno_ode1.csv" \
      --state ema --device "${DEVICE}" --ode-steps 1 --method euler \
      --batch-size "${BATCH_SIZE:-32}" --draws 4 --seed "${EVAL_SEED:-20260814}"
    ;;
  flare-rollout) CHECKPOINT="${FLARE_CHECKPOINT_ROOT}/flare/ood_flare_seed78.pt"; LABEL=ood_scfm; STEPS=10 ;;
  fno-rollout) CHECKPOINT="${FLARE_CHECKPOINT_ROOT}/flare/ood_fno_seed78.pt"; LABEL=ood_fno; STEPS=1 ;;
  *) echo "Unknown target: ${TARGET}" >&2; exit 2 ;;
esac

exec "${PYTHON_BIN}" scripts/evaluate_skx_x5_stage2_rollout_arg.py \
  --checkpoint "${CHECKPOINT}" --label "${LABEL}" \
  --output "${OUTPUT_ROOT}/rollout/raw/${LABEL}_draw${DRAW}.csv" \
  --state ema --device "${DEVICE}" --ode-steps "${STEPS}" \
  --batch-size "${BATCH_SIZE:-64}" --draw-index "${DRAW}" \
  --seed "${EVAL_SEED:-20260814}" --range-start-ns 1.0 --range-end-ns 6.0 \
  --checkpoint-hash skip
