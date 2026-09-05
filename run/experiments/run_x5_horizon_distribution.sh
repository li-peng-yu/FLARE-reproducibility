#!/usr/bin/env bash
source "$(dirname -- "$0")/../common.sh"
RUN_TASK_INDEX="${1:-${TASK_INDEX:-0}}"
if ! [[ "${RUN_TASK_INDEX}" =~ ^[0-9]+$ ]] || (( RUN_TASK_INDEX < 0 || RUN_TASK_INDEX > 4 )); then
  echo "Task index must be in 0..4" >&2; exit 2
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
ROOT=${PROJECT}/outputs/paper_artifacts/horizon_distribution

DURATIONS=(1.0 1.5 2.0 2.5 3.0 3.5)
TAGS=(1p0 1p5 2p0 2p5 3p0 3p5)
DURATION=${DURATIONS[$RUN_TASK_INDEX]}
TAG=${TAGS[$RUN_TASK_INDEX]}
OUTPUT=${ROOT}/duration_${TAG}ns

cd "${PROJECT}"
mkdir -p "${OUTPUT}" "${PROJECT}/log"

export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKYRMION_CFM_FUSED_OPS=1
export PYTHONPATH="${PROJECT}:${DIST_ROOT}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/x5_horizon_distribution_${RUN_GROUP_ID}_${RUN_TASK_INDEX}
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON}" scripts/evaluate_skx_x5_same_condition_distribution.py \
  --checkpoint "${CHECKPOINT}" \
  --stats "${STATS}" \
  --output-root "${OUTPUT}" \
  --state ema \
  --device cuda \
  --disable-cudnn-sdp \
  --repeat-dataset skx_bt_1000base_x5_20260803 \
  --repeats-per-group 5 \
  --segments 1 2 \
  --duration-ns "${DURATION}" \
  --num-model 25 \
  --score-num-model 5 \
  --batch-size 25 \
  --ode-steps 10 \
  --bootstrap 5000 \
  --seed $((208281000 + 100 * RUN_TASK_INDEX))

if "${PYTHON}" -c "import json,sys; p=json.load(open('${OUTPUT}/summary/run_summary.json')); sys.exit(0 if int(p.get('conditions', 0)) > 0 else 1)"; then
  "${PYTHON}" scripts/evaluate_x5_exact_anchor_forecasts.py \
    --method "FLARE=${OUTPUT}" \
    --truth-root "${OUTPUT}" \
    --reference-label FLARE \
    --output-dir "${OUTPUT}/fair_energy" \
    --ensemble-size 5 \
    --expected-conditions 0 \
    --bootstrap 5000 \
    --seed $((208281500 + 100 * RUN_TASK_INDEX)) \
    --workers 1
fi
