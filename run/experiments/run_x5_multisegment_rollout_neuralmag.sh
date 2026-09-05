#!/usr/bin/env bash
source "$(dirname -- "$0")/../common.sh"
RUN_TASK_INDEX="${1:-${TASK_INDEX:-0}}"
if ! [[ "${RUN_TASK_INDEX}" =~ ^[0-9]+$ ]] || (( RUN_TASK_INDEX < 0 || RUN_TASK_INDEX > 10 )); then
  echo "Task index must be in 0..10" >&2; exit 2
fi
RUN_ID="local_${RUN_TASK_INDEX}"
RUN_GROUP_ID="${RUN_ID}"
RUN_WORKERS="${WORKERS:-8}"
export RUN_TASK_INDEX RUN_ID RUN_GROUP_ID RUN_WORKERS
# NeuralMAG-x5 quality shards for the complete drive->post-relax piecewise-
# control rollout. Magnetization and the thermal RNG continue across the one
# physical control boundary.


PROJECT="${PROJECT_ROOT}"
BASE_CONFIG=${PROJECT}/configs/base/x5.yaml
RESULT_ROOT=${RESULT_ROOT:-${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260830/exact_control_rollout_distribution/neuralmag_x5}
CHECKPOINT=${FLARE_CHECKPOINT_ROOT}/baselines/neuralmag_x5.pt
PYTHON="${PYTHON_BIN}"
EXTRA_ARGS=()
if [[ -n "${MAX_GROUPS:-}" ]]; then
  EXTRA_ARGS+=(--max-groups "${MAX_GROUPS}")
fi

cd "${PROJECT}"
mkdir -p "${RESULT_ROOT}" "${PROJECT}/log/skx_paper_revision_20260829"

export PYTHONPATH="${PROJECT}:${PROJECT}/third_party/NeuralMAG:${PROJECT}/third_party/distribution_score/src:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/x5_nmseg_mpl_${RUN_ID}_${RUN_TASK_INDEX}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON}" external_baselines/neuralmag_x5.py score-shard \
  --config "${BASE_CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-root "${RESULT_ROOT}" \
  --shard-index "${RUN_TASK_INDEX}" \
  --shard-count 11 \
  --multisegment-rollout \
  --draws-per-anchor 1 \
  --draw-batch-size 1 \
  --bootstrap 5000 \
  --dt-s 1e-13 \
  --seed 208293830 \
  "${EXTRA_ARGS[@]}"
