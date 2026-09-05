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
# Compute deterministic fair energy and aggregate the three training seeds.


PROJECT="${PROJECT_ROOT}"
PYTHON="${PYTHON_BIN}"
ROOT=${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260902/flow_necessity_rotation_mse
DIST=${ROOT}/distribution
EXACT=${ROOT}/exact_anchor
FAIR=${ROOT}/fair_energy
SUMMARY=${ROOT}/summary

cd "${PROJECT}"
mkdir -p "${FAIR}" "${SUMMARY}" "${PROJECT}/log/skx_paper_revision_20260902/flow_necessity"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

"${PYTHON}" scripts/evaluate_x5_exact_anchor_forecasts.py \
  --truth-root "${DIST}/rotmse_s78" \
  --method "rotmse_s78=${EXACT}/rotmse_s78" \
  --method "rotmse_s79=${EXACT}/rotmse_s79" \
  --method "rotmse_s80=${EXACT}/rotmse_s80" \
  --reference-label rotmse_s78 \
  --output-dir "${FAIR}" \
  --ensemble-size 5 \
  --expected-conditions 33 \
  --bootstrap 50000 \
  --seed 209020161 \
  --workers 8

"${PYTHON}" scripts/summarize_x5_rotation_mse_control.py \
  --root "${ROOT}" \
  --output "${SUMMARY}" \
  --cno-paired-angle-deg 34.43
