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
# Merge the 15 exact-anchor roots and summarize the complete-path Table 16.


PROJECT="${PROJECT_ROOT}"
ROOT=${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260830/exact_control_seed_geometry
DIST=${ROOT}/distribution
EXACT=${ROOT}/exact_anchor
FAIR=${ROOT}/fair_energy
SUMMARY=${ROOT}/summary

cd "${PROJECT}"
mkdir -p "${FAIR}" "${SUMMARY}" "${PROJECT}/log/skx_paper_revision_20260830"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

"${PYTHON_BIN}" scripts/evaluate_x5_exact_anchor_forecasts.py \
  --truth-root "${DIST}/core_s78" \
  --method "core_s78=${EXACT}/core_s78" \
  --method "core_s79=${EXACT}/core_s79" \
  --method "core_s80=${EXACT}/core_s80" \
  --method "cart_s78=${EXACT}/cart_s78" \
  --method "cart_s79=${EXACT}/cart_s79" \
  --method "cart_s80=${EXACT}/cart_s80" \
  --method "tan_s78=${EXACT}/tan_s78" \
  --method "tan_s79=${EXACT}/tan_s79" \
  --method "tan_s80=${EXACT}/tan_s80" \
  --method "a2d_s78=${EXACT}/a2d_s78" \
  --method "a2d_s79=${EXACT}/a2d_s79" \
  --method "a2d_s80=${EXACT}/a2d_s80" \
  --method "rfm_s78=${EXACT}/rfm_s78" \
  --method "rfm_s79=${EXACT}/rfm_s79" \
  --method "rfm_s80=${EXACT}/rfm_s80" \
  --reference-label core_s78 \
  --output-dir "${FAIR}" \
  --ensemble-size 5 \
  --expected-conditions 33 \
  --bootstrap 50000 \
  --seed 208300161 \
  --workers 8

"${PYTHON_BIN}" \
  scripts/summarize_x5_seed_geometry_exact_control.py \
  --root "${ROOT}" \
  --output "${SUMMARY}"
