#!/usr/bin/env bash

set -euo pipefail

TASK_ID="${1:?Usage: run_x30_mumax.sh TASK_ID}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the FLARE_dataset directory}"
X30_ROOT="${X30_ROOT:-${DATASET_ROOT}/self_consistency_x30/skx_bt_165base_x30_sharedrelax_20260813}"
RUN_LIST="${RUN_LIST:-${X30_ROOT}/new_run_list.txt}"
LOG_DIR="${LOG_DIR:-${X30_ROOT}/generation_status}"
RUNS_PER_TASK="${RUNS_PER_TASK:-5}"

command -v mumax3 >/dev/null 2>&1 || {
  echo "mumax3 is not available on PATH" >&2
  exit 2
}
[[ -s "${RUN_LIST}" ]] || {
  echo "Missing or empty run list: ${RUN_LIST}" >&2
  exit 2
}

mkdir -p "${LOG_DIR}"
TOTAL="$(wc -l < "${RUN_LIST}")"
START_INDEX=$((TASK_ID * RUNS_PER_TASK))
STATUS_FILE="${LOG_DIR}/task_${TASK_ID}.tsv"
printf 'index\trun_dir\tstatus\texit_code\tduration_s\tstarted_at\tfinished_at\n' > "${STATUS_FILE}"

FAILED=0
for ((OFFSET = 0; OFFSET < RUNS_PER_TASK; OFFSET++)); do
  INDEX=$((START_INDEX + OFFSET))
  LINE_NUMBER=$((INDEX + 1))
  (( LINE_NUMBER <= TOTAL )) || continue
  LISTED_RUN_DIR="$(sed -n "${LINE_NUMBER}p" "${RUN_LIST}")"
  RUN_DIR="${X30_ROOT}/$(basename -- "${LISTED_RUN_DIR}")"
  STARTED_AT="$(date -Iseconds)"
  START_SECONDS="$(date +%s)"

  if [[ ! -s "${RUN_DIR}/run.mx3" ]]; then
    printf '%s\t%s\tmissing_input\t2\t%s\t%s\t%s\n' \
      "${INDEX}" "${RUN_DIR}" "0" "${STARTED_AT}" "$(date -Iseconds)" >> "${STATUS_FILE}"
    FAILED=$((FAILED + 1))
    continue
  fi

  if [[ -s "${RUN_DIR}/run.out/m_final.ovf" && \
        -s "${RUN_DIR}/run.out/m_initial.ovf" && \
        -s "${RUN_DIR}/run.out/table.txt" ]]; then
    printf '%s\t%s\talready_complete\t0\t%s\t%s\t%s\n' \
      "${INDEX}" "${RUN_DIR}" "0" "${STARTED_AT}" "$(date -Iseconds)" >> "${STATUS_FILE}"
    continue
  fi

  if [[ -d "${RUN_DIR}/run.out" ]]; then
    mv "${RUN_DIR}/run.out" "${RUN_DIR}/run.incomplete.$(date +%s).out"
  fi

  if (cd "${RUN_DIR}" && mumax3 -f run.mx3); then
    RUN_RC=0
  else
    RUN_RC=$?
  fi

  FINISHED_AT="$(date -Iseconds)"
  DURATION=$(( $(date +%s) - START_SECONDS ))
  shopt -s nullglob
  FRAMES=("${RUN_DIR}"/run.out/m[0-9]*.ovf)
  shopt -u nullglob

  if (( RUN_RC == 0 && ${#FRAMES[@]} >= 16 )) && \
     [[ -s "${RUN_DIR}/run.out/m_final.ovf" && \
        -s "${RUN_DIR}/run.out/m_initial.ovf" && \
        -s "${RUN_DIR}/run.out/table.txt" ]]; then
    printf 'index\trun_dir\tthermal_frames\telapsed_s\tcompleted_at\n%s\t%s\t%s\t%s\t%s\n' \
      "${INDEX}" "${RUN_DIR}" "${#FRAMES[@]}" "${DURATION}" "${FINISHED_AT}" \
      > "${RUN_DIR}/.mumax_complete.tsv"
    STATUS=complete
  else
    STATUS=failed_validation
    FAILED=$((FAILED + 1))
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${INDEX}" "${RUN_DIR}" "${STATUS}" "${RUN_RC}" "${DURATION}" "${STARTED_AT}" "${FINISHED_AT}" \
    >> "${STATUS_FILE}"
done

(( FAILED == 0 ))
