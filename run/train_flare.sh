#!/usr/bin/env bash

source "$(dirname -- "$0")/common.sh"

BASE_CONFIG="${1:?Usage: train_flare.sh BASE_CONFIG OVERRIDE_CONFIG}"
OVERRIDE_CONFIG="${2:?Usage: train_flare.sh BASE_CONFIG OVERRIDE_CONFIG}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export NVIDIA_TF32_OVERRIDE=1
export TORCH_DISTRIBUTED_TIMEOUT_MINUTES=120

test -f "${BASE_CONFIG}"
test -f "${OVERRIDE_CONFIG}"

if (( NPROC_PER_NODE == 1 )); then
  exec "${PYTHON_BIN}" -m scripts.train_with_yaml_override \
    --base "${BASE_CONFIG}" \
    --override "${OVERRIDE_CONFIG}"
fi

exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE}" \
  -m scripts.train_with_yaml_override \
  --base "${BASE_CONFIG}" \
  --override "${OVERRIDE_CONFIG}"
