#!/usr/bin/env bash
source "$(dirname -- "$0")/../common.sh"
RUN_TASK_INDEX="${1:-${TASK_INDEX:-0}}"
if ! [[ "${RUN_TASK_INDEX}" =~ ^[0-9]+$ ]] || (( RUN_TASK_INDEX < 0 || RUN_TASK_INDEX > 2 )); then
  echo "Task index must be in 0..2" >&2; exit 2
fi
RUN_ID="local_${RUN_TASK_INDEX}"
RUN_GROUP_ID="${RUN_ID}"
RUN_WORKERS="${WORKERS:-8}"
export RUN_TASK_INDEX RUN_ID RUN_GROUP_ID RUN_WORKERS
# Complete-path evaluation for the deterministic local rotation-vector MSE
# control.  Five outputs mean exactly one deterministic forecast for each of
# the five MuMax repeat anchors.  Euler-1 is required: this model is trained
# only at tau=0 and is not an ODE vector field.


PROJECT="${PROJECT_ROOT}"
PYTHON="${PYTHON_BIN}"
DIST_ROOT=${PROJECT}/third_party/distribution_score
STATS=${PROJECT}/configs/stats/dataset_stats_both.json
RESULT_ROOT=${RESULT_ROOT:-${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260902/flow_necessity_rotation_mse}
EVAL_SEED=${EVAL_SEED:-209020160}
LABELS=(rotmse_s78 rotmse_s79 rotmse_s80)
CHECKPOINTS=(
  "${FLARE_CHECKPOINT_ROOT}/flare/direct_unet_seed78.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/direct_unet_seed79.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/direct_unet_seed80.pt"
)
REFERENCES=(
  "${FLARE_CHECKPOINT_ROOT}/flare/core_seed78.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/core_seed79.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/core_seed80.pt"
)

LABEL=${LABELS[$RUN_TASK_INDEX]}
CHECKPOINT=${CHECKPOINTS[$RUN_TASK_INDEX]}
REFERENCE=${REFERENCES[$RUN_TASK_INDEX]}
QUALITY=${RESULT_ROOT}/distribution/${LABEL}
EXACT=${RESULT_ROOT}/exact_anchor/${LABEL}

cd "${PROJECT}"
test -x "${PYTHON}"
test -s "${CHECKPOINT}"
test -s "${REFERENCE}"
test -s "${STATS}"
test -s "${DIST_ROOT}/ALGORITHM_VERSION.json"
mkdir -p "${QUALITY}" "${EXACT}" "${PROJECT}/log/skx_paper_revision_20260902/flow_necessity"

export PYTHONPATH="${PROJECT}:${DIST_ROOT}/src:${PYTHONPATH:-}"
export SKYRMION_CFM_FUSED_OPS=${SKYRMION_CFM_FUSED_OPS:-1}
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/x5_rotmse_eval_${RUN_ID}_${RUN_TASK_INDEX}
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON}" scripts/audit_x5_rotation_mse_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  --reference-checkpoint "${REFERENCE}" \
  --expected-seed "$((78 + RUN_TASK_INDEX))"

"${PYTHON}" scripts/evaluate_skx_x5_same_condition_distribution.py \
  --checkpoint "${CHECKPOINT}" \
  --stats "${STATS}" \
  --output-root "${QUALITY}" \
  --state ema \
  --device cuda \
  --disable-cudnn-sdp \
  --repeat-dataset skx_bt_1000base_x5_20260803 \
  --repeats-per-group 5 \
  --multisegment-rollout \
  --num-model 5 \
  --score-num-model 5 \
  --batch-size 5 \
  --ode-steps 1 \
  --bootstrap 5000 \
  --seed "${EVAL_SEED}"

"${PYTHON}" scripts/evaluate_skx_x5_same_condition_distribution.py \
  --checkpoint "${CHECKPOINT}" \
  --stats "${STATS}" \
  --output-root "${EXACT}" \
  --state ema \
  --device cuda \
  --disable-cudnn-sdp \
  --repeat-dataset skx_bt_1000base_x5_20260803 \
  --repeats-per-group 5 \
  --multisegment-rollout \
  --num-model 5 \
  --score-num-model 5 \
  --batch-size 5 \
  --ode-steps 1 \
  --bootstrap 5000 \
  --seed "${EVAL_SEED}" \
  --generate-only
