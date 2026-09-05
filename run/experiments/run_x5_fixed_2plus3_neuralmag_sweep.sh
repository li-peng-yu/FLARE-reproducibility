#!/usr/bin/env bash
source "$(dirname -- "$0")/../common.sh"
RUN_TASK_INDEX="${1:-${TASK_INDEX:-0}}"
if ! [[ "${RUN_TASK_INDEX}" =~ ^[0-9]+$ ]] || (( RUN_TASK_INDEX < 0 || RUN_TASK_INDEX > 7 )); then
  echo "Task index must be in 0..7" >&2; exit 2
fi
RUN_ID="local_${RUN_TASK_INDEX}"
RUN_GROUP_ID="${RUN_ID}"
RUN_WORKERS="${WORKERS:-8}"
export RUN_TASK_INDEX RUN_ID RUN_GROUP_ID RUN_WORKERS
# Fill NeuralMAG powers-of-two batches between
# the already submitted B=1 and B=128 measurements. The portable entry point
# also includes those two endpoint jobs, with the same benchmark arguments.


PROJECT="${PROJECT_ROOT}"
BASE_CONFIG=${PROJECT}/configs/base/x5.yaml
RESULT_ROOT=${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260830/fixed_2plus3_5ns_timing/neuralmag_x5
AUTHOR_ROOT=${PROJECT}/third_party
AIPHY_PYTHON="${PYTHON_BIN}"
BATCHES=(1 2 4 8 16 32 64 128)
BATCH=${BATCHES[$RUN_TASK_INDEX]}

mkdir -p "${RESULT_ROOT}" "${PROJECT}/log/skx_paper_revision_20260830"
cd "${PROJECT}"

export PYTHONPATH="${PROJECT}:${AUTHOR_ROOT}/NeuralMAG:${PROJECT}/third_party/distribution_score/src:${PYTHONPATH:-}"
export X5_AUTHOR_REPO_ROOT=${AUTHOR_ROOT}
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/x5_t2p3nm_mpl_${RUN_ID}_${RUN_TASK_INDEX}
export TORCH_EXTENSIONS_DIR=${PROJECT}/external_baselines/torch_extensions
export MAX_JOBS=${RUN_WORKERS}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${AIPHY_PYTHON}" external_baselines/neuralmag_x5.py benchmark \
  --config "${BASE_CONFIG}" \
  --checkpoint "${FLARE_CHECKPOINT_ROOT}/baselines/neuralmag_x5.pt" \
  --output "${RESULT_ROOT}/batch_${BATCH}.json" \
  --multisegment-rollout \
  --timing-segment-durations-ns 2 3 \
  --batch-size "${BATCH}" \
  --repeats 1 \
  --repeat-conditions \
  --dt-s 1e-13
