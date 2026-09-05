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
# Assemble and score five unique training seeds at the five-output Table-1
# budget.  Each repeat/anchor receives one prediction from every seed.


PROJECT="${PROJECT_ROOT}"
PYTHON="${PYTHON_BIN}"
DIST_ROOT=${PROJECT}/third_party/distribution_score
TRUTH=${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260830/exact_control_seed_geometry/distribution/core_s78
TABLE1=${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260830/exact_control_rollout_distribution
DIRECT_OLD=${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260902/flow_necessity_rotation_mse/distribution
ROOT=${PROJECT}/outputs/x5_deterministic_stochasticity_20260903
METHODS=(direct_unet cno_fm dpot_ti poseidon_t pdearena_unet)
SEEDS=(78 79 80 81 82)

METHOD=${METHODS[$RUN_TASK_INDEX]}
OUTPUT=${ROOT}/deep_ensemble5/${METHOD}
LABEL=${METHOD}_deep5

source_root() {
  local method=$1
  local seed=$2
  if [[ "${method}" == direct_unet ]]; then
    if (( seed <= 80 )); then
      echo "${DIRECT_OLD}/rotmse_s${seed}"
    else
      echo "${ROOT}/per_seed/direct_unet_s${seed}"
    fi
  elif (( seed == 78 )); then
    echo "${TABLE1}/${method}"
  else
    echo "${ROOT}/per_seed/${method}_s${seed}"
  fi
}

SOURCES=()
for SEED in "${SEEDS[@]}"; do
  SOURCES+=("$(source_root "${METHOD}" "${SEED}")")
done

cd "${PROJECT}"
mkdir -p "${OUTPUT}" "${PROJECT}/log/x5_five_seed_ensemble_20260904"
export PYTHONPATH=${PROJECT}:${DIST_ROOT}/src
export MPLCONFIGDIR=/tmp/x5_deep5_${RUN_ID}_${RUN_TASK_INDEX}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=1
mkdir -p "${MPLCONFIGDIR}"

SOURCE_ARGS=()
for SOURCE in "${SOURCES[@]}"; do
  test -s "${SOURCE}/summary/run_summary.json"
  SOURCE_ARGS+=(--source-root "${SOURCE}")
done

"${PYTHON}" scripts/assemble_x5_deep_ensemble.py \
  --label "${LABEL}" \
  "${SOURCE_ARGS[@]}" \
  --truth-root "${TRUTH}" \
  --output-root "${OUTPUT}" \
  --device cuda \
  --bootstrap 5000 \
  --seed 209043700

"${PYTHON}" scripts/evaluate_x5_exact_anchor_forecasts.py \
  --method "${LABEL}=${OUTPUT}/exact_anchor" \
  --truth-root "${TRUTH}" \
  --reference-label "${LABEL}" \
  --output-dir "${OUTPUT}/fair_energy" \
  --ensemble-size 5 \
  --expected-conditions 33 \
  --bootstrap 50000 \
  --seed 209043700 \
  --workers 8
