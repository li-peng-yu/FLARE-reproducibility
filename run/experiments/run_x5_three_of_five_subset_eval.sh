#!/usr/bin/env bash
source "$(dirname -- "$0")/../common.sh"
RUN_TASK_INDEX="${1:-${TASK_INDEX:-0}}"
if ! [[ "${RUN_TASK_INDEX}" =~ ^[0-9]+$ ]] || (( RUN_TASK_INDEX < 0 || RUN_TASK_INDEX > 49 )); then
  echo "Task index must be in 0..49" >&2; exit 2
fi
RUN_ID="local_${RUN_TASK_INDEX}"
RUN_GROUP_ID="${RUN_ID}"
RUN_WORKERS="${WORKERS:-8}"
export RUN_TASK_INDEX RUN_ID RUN_GROUP_ID RUN_WORKERS
# Score all C(5,3)=10 training-seed subsets for each deterministic method.
# Large intermediate arrays live in a task-private /tmp directory; only the
# aggregate and per-condition/per-anchor metric tables are retained.


PROJECT="${PROJECT_ROOT}"
PYTHON="${PYTHON_BIN}"
DIST_ROOT=${PROJECT}/third_party/distribution_score
TRUTH=${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260830/exact_control_seed_geometry/distribution/core_s78
TABLE1=${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260830/exact_control_rollout_distribution
DIRECT_OLD=${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260902/flow_necessity_rotation_mse/distribution
ROOT=${PROJECT}/outputs/x5_deterministic_stochasticity_20260903
METHODS=(direct_unet cno_fm dpot_ti poseidon_t pdearena_unet)
SUBSETS=(
  78_79_80
  78_79_81
  78_79_82
  78_80_81
  78_80_82
  78_81_82
  79_80_81
  79_80_82
  79_81_82
  80_81_82
)

METHOD_INDEX=$((RUN_TASK_INDEX / 10))
SUBSET_INDEX=$((RUN_TASK_INDEX % 10))
METHOD=${METHODS[$METHOD_INDEX]}
TAG=${SUBSETS[$SUBSET_INDEX]}
IFS=_ read -r -a SUBSET_SEEDS <<< "${TAG}"
OUTPUT=${ROOT}/deep_ensemble3_subsets/${METHOD}/${TAG}
LABEL=${METHOD}_deep3_${TAG}

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
for SEED in "${SUBSET_SEEDS[@]}"; do
  SOURCES+=("$(source_root "${METHOD}" "${SEED}")")
done

TASK_TMP=$(mktemp -d "/tmp/x5_deep3_subset_${RUN_ID}_${RUN_TASK_INDEX}.XXXXXX")
cleanup() {
  case "${TASK_TMP}" in
    /tmp/x5_deep3_subset_*) rm -rf -- "${TASK_TMP}" ;;
    *) echo "refusing to remove unexpected temporary path: ${TASK_TMP}" >&2 ;;
  esac
}
trap cleanup EXIT
SCRATCH_OUTPUT=${TASK_TMP}/result

cd "${PROJECT}"
mkdir -p "${OUTPUT}/distribution" "${OUTPUT}/fair_energy" \
  "${PROJECT}/log/x5_five_seed_ensemble_20260904"
export PYTHONPATH=${PROJECT}:${DIST_ROOT}/src
export MPLCONFIGDIR=${TASK_TMP}/mpl
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
  --output-root "${SCRATCH_OUTPUT}" \
  --device cuda \
  --bootstrap 5000 \
  --seed "$((209044700 + SUBSET_INDEX))"

"${PYTHON}" scripts/evaluate_x5_exact_anchor_forecasts.py \
  --method "${LABEL}=${SCRATCH_OUTPUT}/exact_anchor" \
  --truth-root "${TRUTH}" \
  --reference-label "${LABEL}" \
  --output-dir "${SCRATCH_OUTPUT}/fair_energy" \
  --ensemble-size 3 \
  --expected-conditions 33 \
  --bootstrap 50000 \
  --seed "$((209044700 + SUBSET_INDEX))" \
  --workers 8

cp "${SCRATCH_OUTPUT}/distribution/summary/run_summary.json" \
  "${OUTPUT}/distribution/run_summary.json"
cp "${SCRATCH_OUTPUT}/distribution/summary/condition_scores.csv" \
  "${OUTPUT}/distribution/condition_scores.csv"
cp "${SCRATCH_OUTPUT}/fair_energy/run_summary.json" \
  "${OUTPUT}/fair_energy/run_summary.json"
cp "${SCRATCH_OUTPUT}/fair_energy/method_summary.csv" \
  "${OUTPUT}/fair_energy/method_summary.csv"
cp "${SCRATCH_OUTPUT}/fair_energy/per_anchor_metrics.csv" \
  "${OUTPUT}/fair_energy/per_anchor_metrics.csv"
