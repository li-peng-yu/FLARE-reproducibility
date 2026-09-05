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
# Merge the 33 NeuralMAG-x5 multisegment quality conditions.


PROJECT="${PROJECT_ROOT}"
OUTPUT_ROOT=${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260830/exact_control_rollout_distribution/neuralmag_x5
CHECKPOINT=${FLARE_CHECKPOINT_ROOT}/baselines/neuralmag_x5.pt

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/third_party/NeuralMAG:${PROJECT}/third_party/distribution_score/src:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/x5_nmsegmerge_mpl_${RUN_ID}

"${PYTHON_BIN}" external_baselines/neuralmag_x5.py merge \
  --output-root "${OUTPUT_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --shard-count 11 \
  --expected-conditions 33 \
  --bootstrap 5000 \
  --seed 208293830
