#!/usr/bin/env bash
source "$(dirname -- "$0")/../common.sh"
RUN_TASK_INDEX="${1:-${TASK_INDEX:-0}}"
if ! [[ "${RUN_TASK_INDEX}" =~ ^[0-9]+$ ]] || (( RUN_TASK_INDEX < 0 || RUN_TASK_INDEX > 14 )); then
  echo "Task index must be in 0..14" >&2; exit 2
fi
RUN_ID="local_${RUN_TASK_INDEX}"
RUN_GROUP_ID="${RUN_ID}"
RUN_WORKERS="${WORKERS:-8}"
export RUN_TASK_INDEX RUN_ID RUN_GROUP_ID RUN_WORKERS
# Complete-path drive->prediction-handoff->post-relax evaluation for the
# five geometry/source families and training seeds 78/79/80 used in Table 16.
# Every array member uses the same evaluation seed so training-seed variation
# is not confounded by a different inference-noise realization.


PROJECT="${PROJECT_ROOT}"
PYTHON="${PYTHON_BIN}"
DIST_ROOT=${PROJECT}/third_party/distribution_score
STATS=${PROJECT}/configs/stats/dataset_stats_both.json
RESULT_ROOT=${RESULT_ROOT:-${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260830/exact_control_seed_geometry}
EVAL_SEED=${EVAL_SEED:-208300160}
BATCH_SIZE=${BATCH_SIZE:-5}
EXACT_BATCH_SIZE=${EXACT_BATCH_SIZE:-25}
EXTRA_ARGS=()
if [[ -n "${MAX_GROUPS:-}" ]]; then
  EXTRA_ARGS+=(--max-groups "${MAX_GROUPS}")
fi

LABELS=(
  core_s78 core_s79 core_s80
  cart_s78 cart_s79 cart_s80
  tan_s78 tan_s79 tan_s80
  a2d_s78 a2d_s79 a2d_s80
  rfm_s78 rfm_s79 rfm_s80
)
CHECKPOINTS=(
  "${FLARE_CHECKPOINT_ROOT}/flare/core_seed78.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/core_seed79.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/core_seed80.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/cartesian_seed78.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/cartesian_seed79.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/cartesian_seed80.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/tangent_seed78.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/tangent_seed79.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/tangent_seed80.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/alpha2d_seed78.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/alpha2d_seed79.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/alpha2d_seed80.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/rfm_seed78.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/rfm_seed79.pt"
  "${FLARE_CHECKPOINT_ROOT}/flare/rfm_seed80.pt"
)

LABEL=${LABELS[$RUN_TASK_INDEX]}
CHECKPOINT=${CHECKPOINTS[$RUN_TASK_INDEX]}
QUALITY=${RESULT_ROOT}/distribution/${LABEL}
EXACT=${RESULT_ROOT}/exact_anchor/${LABEL}

cd "${PROJECT}"
test -x "${PYTHON}"
test -s "${CHECKPOINT}"
test -s "${STATS}"
test -s "${DIST_ROOT}/ALGORITHM_VERSION.json"
mkdir -p "${QUALITY}" "${EXACT}" "${PROJECT}/log/skx_paper_revision_20260830"

export PYTHONPATH="${PROJECT}:${DIST_ROOT}/src:${PYTHONPATH:-}"
export SKYRMION_CFM_FUSED_OPS=${SKYRMION_CFM_FUSED_OPS:-1}
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/x5_seed_path_${RUN_ID}_${RUN_TASK_INDEX}
mkdir -p "${MPLCONFIGDIR}"

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
  --batch-size "${BATCH_SIZE}" \
  --ode-steps 10 \
  --bootstrap 5000 \
  --seed "${EVAL_SEED}" \
  "${EXTRA_ARGS[@]}"

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
  --num-model 25 \
  --score-num-model 5 \
  --batch-size "${EXACT_BATCH_SIZE}" \
  --ode-steps 10 \
  --bootstrap 5000 \
  --seed "${EVAL_SEED}" \
  --generate-only \
  "${EXTRA_ARGS[@]}"
