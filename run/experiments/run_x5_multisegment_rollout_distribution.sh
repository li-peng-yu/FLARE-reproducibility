#!/usr/bin/env bash
source "$(dirname -- "$0")/../common.sh"
RUN_TASK_INDEX="${1:-${TASK_INDEX:-0}}"
if ! [[ "${RUN_TASK_INDEX}" =~ ^[0-9]+$ ]] || (( RUN_TASK_INDEX < 0 || RUN_TASK_INDEX > 6 )); then
  echo "Task index must be in 0..6" >&2; exit 2
fi
RUN_ID="local_${RUN_TASK_INDEX}"
RUN_GROUP_ID="${RUN_ID}"
RUN_WORKERS="${WORKERS:-8}"
export RUN_TASK_INDEX RUN_ID RUN_GROUP_ID RUN_WORKERS
# Complete drive->post-relax quality evaluation with a prediction handoff at
# the physical control boundary.  This is Algorithm 1's segment composition,
# not an arbitrary split of a constant-control 5-ns query.


PROJECT="${PROJECT_ROOT}"
BASE_CONFIG=${PROJECT}/configs/base/x5.yaml
RESULT_ROOT=${RESULT_ROOT:-${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260830/exact_control_rollout_distribution}
DIST_ROOT=${PROJECT}/third_party/distribution_score
PYTHON="${PYTHON_BIN}"
METHODS=(flare poseidon_t cno_fm dpot_ti mpp_avit_ti pdearena_unet le_pde)
METHOD=${METHODS[$RUN_TASK_INDEX]}
EXTRA_ARGS=()
if [[ -n "${MAX_GROUPS:-}" ]]; then
  EXTRA_ARGS+=(--max-groups "${MAX_GROUPS}")
fi

cd "${PROJECT}"
mkdir -p "${RESULT_ROOT}/${METHOD}" "${PROJECT}/log/skx_paper_revision_20260829"

export PYTHONPATH="${PROJECT}:${DIST_ROOT}/src:${PYTHONPATH:-}"
export X5_AUTHOR_REPO_ROOT=${PROJECT}/third_party
export HF_HOME=/tmp/x5_qseg_hf_${RUN_ID}_${RUN_TASK_INDEX}
export TRANSFORMERS_CACHE=${HF_HOME}
export MPLCONFIGDIR=/tmp/x5_qseg_mpl_${RUN_ID}_${RUN_TASK_INDEX}
export TORCH_EXTENSIONS_DIR=${PROJECT}/external_baselines/torch_extensions
export MAX_JOBS=${RUN_WORKERS}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKYRMION_CFM_FUSED_OPS=1

if [[ "${METHOD}" == flare ]]; then
  "${PYTHON}" scripts/evaluate_skx_x5_same_condition_distribution.py \
    --checkpoint "${FLARE_CHECKPOINT_ROOT}/flare/core_seed78.pt" \
    --stats ${PROJECT}/configs/stats/dataset_stats_both.json \
    --output-root "${RESULT_ROOT}/${METHOD}" \
    --state ema \
    --device cuda \
    --disable-cudnn-sdp \
    --repeat-dataset skx_bt_1000base_x5_20260803 \
    --repeats-per-group 5 \
    --multisegment-rollout \
    --num-model 5 \
    --score-num-model 5 \
    --batch-size 5 \
    --ode-steps 10 \
    --bootstrap 5000 \
    --seed 208292700 \
    "${EXTRA_ARGS[@]}"
else
  PRECISION=fp32
  if [[ "${METHOD}" == pdearena_unet || "${METHOD}" == le_pde ]]; then
    PRECISION=bf16
  fi
  CHECKPOINT=${FLARE_CHECKPOINT_ROOT}/baselines/${METHOD}.pt
  "${PYTHON}" external_baselines/evaluate_x5_native_same_condition_distribution.py \
    --method "${METHOD}" \
    --config "${BASE_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-root "${RESULT_ROOT}/${METHOD}" \
    --precision "${PRECISION}" \
    --device cuda \
    --multisegment-rollout \
    --bootstrap 5000 \
    --seed $((208292700 + 1000 * RUN_TASK_INDEX)) \
    "${EXTRA_ARGS[@]}"
fi
