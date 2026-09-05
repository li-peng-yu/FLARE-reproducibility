#!/usr/bin/env bash
source "$(dirname -- "$0")/../common.sh"
RUN_TASK_INDEX="${1:-${TASK_INDEX:-0}}"
if ! [[ "${RUN_TASK_INDEX}" =~ ^[0-9]+$ ]] || (( RUN_TASK_INDEX < 0 || RUN_TASK_INDEX > 0 )); then
  echo "Task index must be in 0..0" >&2; exit 2
fi
RUN_ID="local_${RUN_TASK_INDEX}"
RUN_GROUP_ID="${RUN_ID}"
RUN_WORKERS="${WORKERS:-8}"
export RUN_TASK_INDEX RUN_ID RUN_GROUP_ID RUN_WORKERS
# Aggregate Fair Energy Score after all exact-anchor samples and the one-draw
# same-condition roots are complete.

PROJECT="${PROJECT_ROOT}"
QUALITY=${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260830/exact_control_rollout_distribution
EXACT=${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260830/exact_control_rollout_exact_anchor
OUTPUT=${EXACT}/fair_energy

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

"${PYTHON_BIN}" scripts/evaluate_x5_exact_anchor_forecasts.py \
  --truth-root "${QUALITY}/flare" \
  --method "flare=${EXACT}/flare" \
  --method "cartesian_cfm=${EXACT}/cartesian_cfm" \
  --method "direct_unet=${QUALITY}/direct_unet" \
  --method "poseidon_t=${QUALITY}/poseidon_t" \
  --method "cno_fm=${QUALITY}/cno_fm" \
  --method "dpot_ti=${QUALITY}/dpot_ti" \
  --method "mpp_avit_ti=${QUALITY}/mpp_avit_ti" \
  --method "pdearena_unet=${QUALITY}/pdearena_unet" \
  --method "le_pde=${QUALITY}/le_pde" \
  --method "neuralmag_x5=${EXACT}/neuralmag_x5" \
  --reference-label flare \
  --output-dir "${OUTPUT}" \
  --ensemble-size 5 \
  --expected-conditions 33 \
  --bootstrap 50000 \
  --seed 208295700 \
  --workers 8
