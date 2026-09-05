#!/usr/bin/env bash

set -euo pipefail

TASK_ID="${1:?Usage: run_mumax_task.sh TASK_ID}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to a generated dataset directory}"
RUN_LIST="${RUN_LIST:-${RUN_ROOT}/run_list.txt}"

command -v mumax3 >/dev/null 2>&1 || {
  echo "mumax3 is not available on PATH" >&2
  exit 2
}
[[ -s "${RUN_LIST}" ]] || {
  echo "Missing or empty run list: ${RUN_LIST}" >&2
  exit 2
}

LINE_NUMBER=$((TASK_ID + 1))
LISTED_RUN_DIR="$(sed -n "${LINE_NUMBER}p" "${RUN_LIST}")"
[[ -n "${LISTED_RUN_DIR}" ]] || {
  echo "No run at zero-based task index ${TASK_ID}" >&2
  exit 2
}

RUN_DIR="${RUN_ROOT}/$(basename -- "${LISTED_RUN_DIR}")"
[[ -s "${RUN_DIR}/run.mx3" ]] || {
  echo "Missing run.mx3: ${RUN_DIR}" >&2
  exit 2
}

if [[ -s "${RUN_DIR}/run.out/m_final.ovf" ]]; then
  echo "Already complete: ${RUN_DIR}"
  exit 0
fi

cd "${RUN_DIR}"
exec mumax3 -f run.mx3
