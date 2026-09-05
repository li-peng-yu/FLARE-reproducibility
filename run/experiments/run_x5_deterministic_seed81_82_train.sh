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
# Add seed 81/82 for the four deterministic external Table-1 baselines.
# Fixed method and seed order.
# while the two 2-GPU Direct U-Net jobs occupy four cards.


PROJECT="${PROJECT_ROOT}"
AUTHOR_ROOT=${PROJECT}/third_party
PYTHON="${PYTHON_BIN}"
CONFIG=${PROJECT}/configs/base/x5.yaml
ROOT=${PROJECT}/outputs/x5_deterministic_stochasticity_20260903

METHODS=(cno_fm cno_fm dpot_ti dpot_ti poseidon_t poseidon_t pdearena_unet pdearena_unet)
SEEDS=(81 82 81 82 81 82 81 82)
BATCH_SIZES=(2 2 8 8 4 4 4 4)
GRAD_ACCUM=(4 4 1 1 2 2 2 2)
PRECISIONS=(fp32 fp32 fp32 fp32 fp32 fp32 bf16 bf16)

METHOD=${METHODS[$RUN_TASK_INDEX]}
SEED=${SEEDS[$RUN_TASK_INDEX]}
BATCH_SIZE=${BATCH_SIZES[$RUN_TASK_INDEX]}
ACCUM=${GRAD_ACCUM[$RUN_TASK_INDEX]}
PRECISION=${PRECISIONS[$RUN_TASK_INDEX]}
OUTPUT=${ROOT}/train/${METHOD}/seed${SEED}
INIT_ARGS=(--pretrained)
if [[ "${METHOD}" == pdearena_unet ]]; then
  INIT_ARGS=()
fi

cd "${PROJECT}"
mkdir -p "${OUTPUT}" "${PROJECT}/log/x5_five_seed_ensemble_20260904"

export X5_AUTHOR_REPO_ROOT=${AUTHOR_ROOT}
export PYTHONPATH=${PROJECT}
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/tmp/x5_det_s45_hf_${RUN_ID}_${RUN_TASK_INDEX}
export TRANSFORMERS_CACHE=${HF_HOME}
export MPLCONFIGDIR=/tmp/x5_det_s45_mpl_${RUN_ID}_${RUN_TASK_INDEX}
export TORCH_EXTENSIONS_DIR=${PROJECT}/external_baselines/torch_extensions
export MAX_JOBS=${RUN_WORKERS}
export OMP_NUM_THREADS="${WORKERS:-8}"
export MKL_NUM_THREADS="${WORKERS:-8}"
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${HF_HOME}" "${MPLCONFIGDIR}"

if [[ -s "${OUTPUT}/checkpoint_0050000.pt" ]]; then
  echo "checkpoint already complete: ${OUTPUT}/checkpoint_0050000.pt"
  exit 0
fi

"${PYTHON}" external_baselines/x5_author_native.py train \
  --method "${METHOD}" \
  --seed "${SEED}" \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT}" \
  --steps 50000 \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${ACCUM}" \
  --num-workers 8 \
  --lr 1.0e-4 \
  --warmup-steps 500 \
  --optimizer-recipe standardized \
  --precision "${PRECISION}" \
  --log-every 50 \
  --save-every 50000 \
  "${INIT_ARGS[@]}"
