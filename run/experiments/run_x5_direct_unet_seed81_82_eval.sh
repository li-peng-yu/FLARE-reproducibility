#!/usr/bin/env bash
source "$(dirname -- "$0")/../common.sh"
RUN_TASK_INDEX="${1:-${TASK_INDEX:-0}}"
if ! [[ "${RUN_TASK_INDEX}" =~ ^[0-9]+$ ]] || (( RUN_TASK_INDEX < 0 || RUN_TASK_INDEX > 1 )); then
  echo "Task index must be in 0..1" >&2; exit 2
fi
RUN_ID="local_${RUN_TASK_INDEX}"
RUN_GROUP_ID="${RUN_ID}"
RUN_WORKERS="${WORKERS:-8}"
export RUN_TASK_INDEX RUN_ID RUN_GROUP_ID RUN_WORKERS
# Evaluate the two additional Direct U-Net members on the exact Table-1 path.
# The seed-79 CFM checkpoint is used only as an architecture/recipe audit
# reference; seed identity is checked separately for the Direct U-Net member.


PROJECT="${PROJECT_ROOT}"
PYTHON="${PYTHON_BIN}"
DIST_ROOT=${PROJECT}/third_party/distribution_score
STATS=${PROJECT}/configs/stats/dataset_stats_both.json
ROOT=${PROJECT}/outputs/x5_deterministic_stochasticity_20260903
REFERENCE=${FLARE_CHECKPOINT_ROOT}/flare/core_seed79.pt
SEEDS=(81 82)
CHECKPOINTS=(
  "${FLARE_CHECKPOINT_ROOT}/flare/direct_unet_seed81.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/direct_unet_seed82.pt"
)

SEED=${SEEDS[$RUN_TASK_INDEX]}
CHECKPOINT=${CHECKPOINTS[$RUN_TASK_INDEX]}
OUTPUT=${ROOT}/per_seed/direct_unet_s${SEED}

cd "${PROJECT}"
test -x "${PYTHON}"
test -s "${CHECKPOINT}"
test -s "${REFERENCE}"
test -s "${STATS}"
test -s "${DIST_ROOT}/ALGORITHM_VERSION.json"
mkdir -p "${OUTPUT}" "${PROJECT}/log/x5_five_seed_ensemble_20260904"

export PYTHONPATH=${PROJECT}:${DIST_ROOT}/src:${PYTHONPATH:-}
export SKYRMION_CFM_FUSED_OPS=${SKYRMION_CFM_FUSED_OPS:-1}
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/x5_direct_s45_eval_${RUN_ID}_${RUN_TASK_INDEX}
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON}" scripts/audit_x5_rotation_mse_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  --reference-checkpoint "${REFERENCE}" \
  --expected-seed "${SEED}" \
  --expected-reference-seed 79

"${PYTHON}" scripts/evaluate_skx_x5_same_condition_distribution.py \
  --checkpoint "${CHECKPOINT}" \
  --stats "${STATS}" \
  --output-root "${OUTPUT}" \
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
  --seed 209041700
