#!/usr/bin/env bash
source "$(dirname -- "$0")/../common.sh"
RUN_TASK_INDEX="${1:-${TASK_INDEX:-0}}"
if ! [[ "${RUN_TASK_INDEX}" =~ ^[0-9]+$ ]] || (( RUN_TASK_INDEX < 0 || RUN_TASK_INDEX > 0 )); then
  echo "Task index must be in 0..0" >&2; exit 2
fi
RUN_ID="local_${RUN_TASK_INDEX}"
RUN_GROUP_ID="${RUN_ID}"
RUN_WORKERS="${WORKERS:-8}"
export RUN_TASK_INDEX RUN_ID RUN_GROUP_ID RUN_WORKERS


PROJECT="${PROJECT_ROOT}"
ZFS=${PROJECT}/outputs/skx_bt_1000base_x5
PYTHON="${PYTHON_BIN}"
OUT=$PROJECT/outputs/skx_bt_1000base_x5/paper_revision_20260902/direct_unet_figure21
FLARE_CHECKPOINT=$ZFS/paper_experiments_20260813/stage1_standard_prior50k/checkpoint_final.pt
CARTESIAN_CHECKPOINT=$ZFS/paper_experiments_20260813/stage1_cartesian_cfm50k/checkpoint_final.pt
DIRECT_CHECKPOINT=$ZFS/paper_revision_20260902/flow_necessity/rotation_mse_seed78/checkpoint_final.pt
FNO_CHECKPOINT=$ZFS/paper_full_20260814/stage1_fno50k/checkpoint_final.pt

cd "$PROJECT"
export PYTHONPATH="$PROJECT"
export SKYRMION_CFM_FUSED_OPS=1
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/x5_fig21_direct_$RUN_ID
mkdir -p "$MPLCONFIGDIR" "$OUT" "$PROJECT/log"

"$PYTHON" scripts/audit_x5_rotation_mse_checkpoint.py \
  --checkpoint "$DIRECT_CHECKPOINT" \
  --reference-checkpoint "$FLARE_CHECKPOINT" \
  --expected-seed 78

"$PYTHON" scripts/render_x5_direct_unet_figure21.py \
  --selection-manifest "$PROJECT/outputs/skx_bt_1000base_x5/paper_formal_20260822/qualitative/selection_manifest.json" \
  --dataset-checkpoint "$FLARE_CHECKPOINT" \
  --flare "$FLARE_CHECKPOINT" \
  --cartesian "$CARTESIAN_CHECKPOINT" \
  --direct-unet "$DIRECT_CHECKPOINT" \
  --fno "$FNO_CHECKPOINT" \
  --output "$OUT/appendix_magnetic_baselines" \
  --device cuda \
  --seed 20260822
