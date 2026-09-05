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


PROJECT="${PROJECT_ROOT}"
PYTHON="${PYTHON_BIN}"

cd "${PROJECT}"
mkdir -p "${PROJECT}/log" "${PROJECT}/outputs/paper_artifacts/appendix_metric_stress"

export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/appendix_metric_stress_${RUN_ID}
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON}" scripts/analyze_appendix_metric_stress.py \
  --device cuda \
  --model-pool 40 \
  --trials 32 \
  --bootstrap 5000 \
  --pair-block 8
