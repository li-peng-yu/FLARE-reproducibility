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
DIST_ROOT=${PROJECT}/third_party/distribution_score
PYTHON="${PYTHON_BIN}"
CHECKPOINT=${FLARE_CHECKPOINT_ROOT}/flare/core_seed78.pt
STATS=${PROJECT}/configs/stats/dataset_stats_both.json
EVAL_CONFIG=${PROJECT}/configs/data/x30_evaluation.yaml
OUTPUT=${PROJECT}/outputs/skx_bt_165base_x30/evaluation_20260828/standard_prior50k_on_x30_test

cd "${PROJECT}"
mkdir -p "${OUTPUT}" "${PROJECT}/log"

export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKYRMION_CFM_FUSED_OPS=1
export PYTHONPATH="${PROJECT}:${DIST_ROOT}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/x30_standard_prior_stress_${RUN_ID}
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON}" scripts/evaluate_skx_x5_same_condition_distribution.py \
  --checkpoint "${CHECKPOINT}" \
  --stats "${STATS}" \
  --evaluation-config "${EVAL_CONFIG}" \
  --output-root "${OUTPUT}" \
  --state ema \
  --device cuda \
  --repeat-dataset skx_bt_165base_x30_sharedrelax_20260813 \
  --repeats-per-group 30 \
  --segments 1 2 \
  --num-model 128 \
  --batch-size 32 \
  --ode-steps 10 \
  --bootstrap 5000 \
  --seed 208280700

"${PYTHON}" scripts/analyze_appendix_metric_stress.py \
  --x30-root "${OUTPUT}" \
  --primary-root outputs/auxiliary_distribution \
  --device cuda \
  --model-pool 40 \
  --trials 32 \
  --bootstrap 5000 \
  --pair-block 8
