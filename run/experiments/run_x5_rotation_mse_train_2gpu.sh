#!/usr/bin/env bash
source "$(dirname -- "$0")/../common.sh"
RUN_TASK_INDEX="${1:-${TASK_INDEX:-0}}"
if ! [[ "${RUN_TASK_INDEX}" =~ ^[0-9]+$ ]] || (( RUN_TASK_INDEX < 0 || RUN_TASK_INDEX > 2 )); then
  echo "Task index must be in 0..2" >&2; exit 2
fi
RUN_ID="local_${RUN_TASK_INDEX}"
RUN_GROUP_ID="${RUN_ID}"
RUN_WORKERS="${WORKERS:-8}"
export RUN_TASK_INDEX RUN_ID RUN_GROUP_ID RUN_WORKERS
# Train the three matched deterministic rotation-target regressors.
# The implementation is delegated to the same launcher used by the paper
# CFM/representation runs so DDP, precision, and checkpointing stay matched.


PROJECT="${PROJECT_ROOT}"
BASE_CONFIG=${PROJECT}/configs/base/x5.yaml
OVERRIDES=(
  "${PROJECT}/configs/flare/stage1_rotation_mse50k_seed78.yaml"
  "${PROJECT}/configs/flare/stage1_rotation_mse50k_seed79.yaml"
  "${PROJECT}/configs/flare/stage1_rotation_mse50k_seed80.yaml"
)
OVERRIDE_CONFIG=${OVERRIDES[$RUN_TASK_INDEX]}

export PROJECT_DIR=${PROJECT}
export BASE_CONFIG
export OVERRIDE_CONFIG
export NPROC_PER_NODE=2
exec bash "${PROJECT}/run/train_flare.sh" "${BASE_CONFIG}" "${OVERRIDE_CONFIG}"
