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
# Test-time stochasticization by five smooth tangent perturbations per anchor.


PROJECT="${PROJECT_ROOT}"
PYTHON="${PYTHON_BIN}"
CONFIG=${PROJECT}/configs/base/x5.yaml
STATS=${PROJECT}/configs/stats/dataset_stats_both.json
DIST_ROOT=${PROJECT}/third_party/distribution_score
TRUTH=${PROJECT}/outputs/skx_bt_1000base_x5/paper_revision_20260830/exact_control_seed_geometry/distribution/core_s78
ROOT=${PROJECT}/outputs/x5_deterministic_stochasticity_20260903
METHODS=(direct_unet direct_unet direct_unet cno_fm cno_fm cno_fm dpot_ti dpot_ti dpot_ti poseidon_t poseidon_t poseidon_t pdearena_unet pdearena_unet pdearena_unet)
JITTERS=(0.25 0.5 1.0 0.25 0.5 1.0 0.25 0.5 1.0 0.25 0.5 1.0 0.25 0.5 1.0)
TAGS=(0p25 0p5 1p0 0p25 0p5 1p0 0p25 0p5 1p0 0p25 0p5 1p0 0p25 0p5 1p0)
METHOD=${METHODS[$RUN_TASK_INDEX]}
JITTER=${JITTERS[$RUN_TASK_INDEX]}
TAG=${TAGS[$RUN_TASK_INDEX]}
OUTPUT=${ROOT}/anchor_jitter/${METHOD}/rms_${TAG}
LABEL=${METHOD}_jitter_${TAG}

cd "${PROJECT}"
mkdir -p "${OUTPUT}" "${PROJECT}/log/x5_deterministic_stochasticity_20260903"
export PYTHONPATH=${PROJECT}:${DIST_ROOT}/src
export X5_AUTHOR_REPO_ROOT=${PROJECT}/third_party
export HF_HOME=/tmp/x5_anchor_jitter_hf_${RUN_ID}_${RUN_TASK_INDEX}
export TRANSFORMERS_CACHE=${HF_HOME}
export MPLCONFIGDIR=/tmp/x5_anchor_jitter_mpl_${RUN_ID}_${RUN_TASK_INDEX}
export TORCH_EXTENSIONS_DIR=${PROJECT}/external_baselines/torch_extensions
export MAX_JOBS=${RUN_WORKERS}
export SKYRMION_CFM_FUSED_OPS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${HF_HOME}" "${MPLCONFIGDIR}" "${TORCH_EXTENSIONS_DIR}"

if [[ "${METHOD}" == direct_unet ]]; then
  CHECKPOINT=${FLARE_CHECKPOINT_ROOT}/flare/direct_unet_seed78.pt
  COMMON=(
    --checkpoint "${CHECKPOINT}"
    --stats "${STATS}"
    --state ema
    --device cuda
    --disable-cudnn-sdp
    --repeat-dataset skx_bt_1000base_x5_20260803
    --repeats-per-group 5
    --multisegment-rollout
    --ode-steps 1
    --bootstrap 5000
    --seed 209032700
    --anchor-jitter-rms-deg "${JITTER}"
    --anchor-jitter-correlation-px 4.0
  )
  "${PYTHON}" scripts/evaluate_skx_x5_same_condition_distribution.py \
    "${COMMON[@]}" \
    --output-root "${OUTPUT}/distribution" \
    --num-model 5 \
    --score-num-model 5 \
    --batch-size 5
  "${PYTHON}" scripts/evaluate_skx_x5_same_condition_distribution.py \
    "${COMMON[@]}" \
    --output-root "${OUTPUT}/exact_anchor" \
    --num-model 25 \
    --score-num-model 5 \
    --batch-size 25 \
    --generate-only
else
  PRECISION=fp32
  if [[ "${METHOD}" == pdearena_unet ]]; then
    PRECISION=bf16
  fi
  CHECKPOINT=${FLARE_CHECKPOINT_ROOT}/baselines/${METHOD}.pt
  COMMON=(
    --method "${METHOD}"
    --config "${CONFIG}"
    --checkpoint "${CHECKPOINT}"
    --precision "${PRECISION}"
    --device cuda
    --multisegment-rollout
    --bootstrap 5000
    --seed 209032700
    --anchor-jitter-rms-deg "${JITTER}"
    --anchor-jitter-correlation-px 4.0
  )
  "${PYTHON}" external_baselines/evaluate_x5_native_same_condition_distribution.py \
    "${COMMON[@]}" \
    --output-root "${OUTPUT}/distribution" \
    --draws-per-anchor 1
  "${PYTHON}" external_baselines/evaluate_x5_native_same_condition_distribution.py \
    "${COMMON[@]}" \
    --output-root "${OUTPUT}/exact_anchor" \
    --draws-per-anchor 5 \
    --generate-only
fi

"${PYTHON}" scripts/evaluate_x5_exact_anchor_forecasts.py \
  --method "${LABEL}=${OUTPUT}/exact_anchor" \
  --truth-root "${TRUTH}" \
  --reference-label "${LABEL}" \
  --output-dir "${OUTPUT}/fair_energy" \
  --ensemble-size 5 \
  --expected-conditions 33 \
  --bootstrap 50000 \
  --seed 209032700 \
  --workers 8

