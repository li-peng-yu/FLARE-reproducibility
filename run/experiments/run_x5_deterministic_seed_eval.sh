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
# Evaluate each newly trained deterministic member on the Table-1 exact path.


PROJECT="${PROJECT_ROOT}"
PYTHON="${PYTHON_BIN}"
CONFIG=${PROJECT}/configs/base/x5.yaml
ROOT=${PROJECT}/outputs/x5_deterministic_stochasticity_20260903
DIST_ROOT=${PROJECT}/third_party/distribution_score
METHODS=(cno_fm dpot_ti poseidon_t pdearena_unet cno_fm dpot_ti poseidon_t pdearena_unet)
SEEDS=(79 79 79 79 80 80 80 80)
PRECISIONS=(fp32 fp32 fp32 bf16 fp32 fp32 fp32 bf16)
METHOD=${METHODS[$RUN_TASK_INDEX]}
SEED=${SEEDS[$RUN_TASK_INDEX]}
PRECISION=${PRECISIONS[$RUN_TASK_INDEX]}
CHECKPOINT=${ROOT}/train/${METHOD}/seed${SEED}/checkpoint_0050000.pt
OUTPUT=${ROOT}/per_seed/${METHOD}_s${SEED}

cd "${PROJECT}"
test -s "${CHECKPOINT}"
mkdir -p "${OUTPUT}" "${PROJECT}/log/x5_deterministic_stochasticity_20260903"
export PYTHONPATH=${PROJECT}:${DIST_ROOT}/src
export X5_AUTHOR_REPO_ROOT=${PROJECT}/third_party
export HF_HOME=/tmp/x5_det_seed_eval_hf_${RUN_ID}_${RUN_TASK_INDEX}
export TRANSFORMERS_CACHE=${HF_HOME}
export MPLCONFIGDIR=/tmp/x5_det_seed_eval_mpl_${RUN_ID}_${RUN_TASK_INDEX}
export TORCH_EXTENSIONS_DIR=${PROJECT}/external_baselines/torch_extensions
export MAX_JOBS=${RUN_WORKERS}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${HF_HOME}" "${MPLCONFIGDIR}" "${TORCH_EXTENSIONS_DIR}"

"${PYTHON}" external_baselines/evaluate_x5_native_same_condition_distribution.py \
  --method "${METHOD}" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output-root "${OUTPUT}" \
  --precision "${PRECISION}" \
  --device cuda \
  --multisegment-rollout \
  --bootstrap 5000 \
  --seed 209031700

