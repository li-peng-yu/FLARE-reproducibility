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
# Full supported powers-of-two sweep for the
# standardized 2+3-ns prediction-handoff workload (non-NeuralMAG methods).


PROJECT="${PROJECT_ROOT}"
BASE_CONFIG=${PROJECT}/configs/base/x5.yaml
RESULT_ROOT=${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260830/fixed_2plus3_5ns_timing
AUTHOR_ROOT=${PROJECT}/third_party
NATIVE_PYTHON="${PYTHON_BIN}"
METHODS=(flare poseidon_t cno_fm dpot_ti mpp_avit_ti pdearena_unet le_pde)
METHOD=${METHODS[$RUN_TASK_INDEX]}
METHOD_ROOT=${RESULT_ROOT}/${METHOD}

case "${METHOD}" in
  flare)
    BATCHES=(1 2 4 8 16 32 64 128)
    ;;
  poseidon_t)
    BATCHES=(1 2 4 8 16 32 64 128 256 512 1024 2048)
    ;;
  cno_fm)
    BATCHES=(1 2 4 8 16 32 64 128 256 512 1024 2048)
    ;;
  dpot_ti)
    BATCHES=(1 2 4 8 16 32 64 128 256)
    ;;
  mpp_avit_ti)
    BATCHES=(1 2 4 8 16 32 64 128)
    ;;
  pdearena_unet)
    BATCHES=(1 2 4 8 16 32 64 128 256 512 1024)
    ;;
  le_pde)
    BATCHES=(1 2 4 8 16 32 64 128 256 512 1024 2048)
    ;;
esac

mkdir -p "${METHOD_ROOT}" "${PROJECT}/log/skx_paper_revision_20260830"
cd "${PROJECT}"

export PYTHONPATH="${PROJECT}:${AUTHOR_ROOT}/NeuralMAG:${PROJECT}/third_party/distribution_score/src:${PYTHONPATH:-}"
export X5_AUTHOR_REPO_ROOT=${AUTHOR_ROOT}
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/tmp/x5_t2p3sw_hf_${RUN_ID}_${RUN_TASK_INDEX}
export TRANSFORMERS_CACHE=${HF_HOME}
export MPLCONFIGDIR=/tmp/x5_t2p3sw_mpl_${RUN_ID}_${RUN_TASK_INDEX}
export TORCH_EXTENSIONS_DIR=${PROJECT}/external_baselines/torch_extensions
export MAX_JOBS=${RUN_WORKERS}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "${METHOD}" == flare ]]; then
  "${NATIVE_PYTHON}" graph/benchmark_x5_scfm_author_native.py \
    --config "${BASE_CONFIG}" \
    --override-config "${PROJECT}/configs/flare/stage1_standard_prior50k.yaml" \
    --checkpoint "${FLARE_CHECKPOINT_ROOT}/flare/core_seed78.pt" \
    --output "${METHOD_ROOT}/sweep.json" \
    --multisegment-rollout \
    --timing-segment-durations-ns 2 3 \
    --samples 1 \
    --batch-sizes "${BATCHES[@]}" \
    --repeats 10 \
    --ode-steps 10 \
    --precision bf16 \
    --channels-last \
    --repeat-single-case
else
  PRECISION=fp32
  if [[ "${METHOD}" == pdearena_unet || "${METHOD}" == le_pde ]]; then
    PRECISION=bf16
  fi
  CHECKPOINT=${FLARE_CHECKPOINT_ROOT}/baselines/${METHOD}.pt
  "${NATIVE_PYTHON}" external_baselines/x5_author_native.py time-horizon \
    --method "${METHOD}" \
    --config "${BASE_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${METHOD_ROOT}/sweep.json" \
    --multisegment-rollout \
    --timing-segment-durations-ns 2 3 \
    --batch-sizes "${BATCHES[@]}" \
    --repeats 10 \
    --precision "${PRECISION}"
fi
