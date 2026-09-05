#!/usr/bin/env bash
# Generate one fresh 400 K shared relaxation and its five thermal branches.
# The array index is one-based and maps directly to base_run_list.txt.

set -euo pipefail

OOD_ROOT="${1:?Supply generated dataset root}"
TARGET_TEMP_TOKEN="${2:?Supply temperature}"
RUN_TASK_INDEX="${3:?Supply base task 1..36}"
RUN_ID="local"
BASE_LIST=${BASE_LIST:-${OOD_ROOT}/base_run_list.txt}
STATUS_ROOT=${STATUS_ROOT:-${OOD_ROOT}/generation_status}
MUMAX_BIN="${MUMAX_BIN:?Set MUMAX_BIN to the recorded MuMax3.12 CUDA12 binary}"
EXPECTED_MUMAX_SHA256=${EXPECTED_MUMAX_SHA256:-37eff7666a86c349089daaa8a637c4f4270a6a79c9135b8fc19a7115e628d06f}

export OMP_NUM_THREADS=2

test -x "${MUMAX_BIN}" || {
    echo "MuMax binary is unavailable: ${MUMAX_BIN}" >&2
    exit 2
}
test -s "${BASE_LIST}" || {
    echo "Missing base list: ${BASE_LIST}" >&2
    exit 2
}

BASE_TOKEN="$(sed -n "${RUN_TASK_INDEX}p" "${BASE_LIST}")"
[[ "${BASE_TOKEN}" =~ ^[0-9]{4}$ ]] || {
    echo "Invalid base token at task ${RUN_TASK_INDEX}: ${BASE_TOKEN}" >&2
    exit 2
}

mkdir -p "${STATUS_ROOT}" "${OOD_ROOT}/logs"
STATUS_FILE="${STATUS_ROOT}/base${BASE_TOKEN}.tsv"
TEMP_STATUS="${STATUS_FILE}.tmp.${RUN_ID}.${RUN_TASK_INDEX}"
printf 'stage\tbase\trun_dir\tstatus\texit_code\tduration_s\tstarted_at\tfinished_at\tmumax_version\tmumax_commit\tmumax_sha256\thost\tgpu\n' > "${TEMP_STATUS}"

MUMAX_INFO="$("${MUMAX_BIN}" -v 2>&1)"
MUMAX_VERSION="$(printf '%s\n' "${MUMAX_INFO}" | sed -n '1p')"
MUMAX_COMMIT="$(printf '%s\n' "${MUMAX_INFO}" | sed -n '2p')"
MUMAX_SHA256="$(sha256sum "${MUMAX_BIN}" | awk '{print $1}')"
HOST_NAME="local"
GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed -n '1p')"

[[ "${MUMAX_VERSION}" == *'mumax 3.12'* && "${MUMAX_VERSION}" == *'CUDA-12.0'* ]] || {
    echo "Unexpected MuMax version: ${MUMAX_VERSION}" >&2
    exit 4
}
[[ "${MUMAX_COMMIT}" == *'6e5c98bb'* ]] || {
    echo "Unexpected MuMax commit: ${MUMAX_COMMIT}" >&2
    exit 4
}
[[ "${MUMAX_SHA256}" == "${EXPECTED_MUMAX_SHA256}" ]] || {
    echo "Unexpected MuMax SHA256: ${MUMAX_SHA256}" >&2
    exit 4
}

run_input() {
    local stage="$1"
    local run_dir="$2"
    local output_name="$3"
    local required_initial="$4"
    local required_final="$5"
    local started_at start_s end_s finished_at duration rc status archived

    started_at="$(date -Iseconds)"
    start_s="$(date +%s)"
    rc=0
    status=complete

    if [[ -s "${run_dir}/${output_name}/${required_initial}" \
          && -s "${run_dir}/${output_name}/${required_final}" \
          && -s "${run_dir}/${output_name}/table.txt" \
          && -s "${run_dir}/${output_name}/log.txt" ]] \
        && grep -q 'mumax 3.12.*CUDA-12.0' "${run_dir}/${output_name}/log.txt" \
        && grep -q '6e5c98bb' "${run_dir}/${output_name}/log.txt"; then
        status=already_complete
    else
        if [[ -d "${run_dir}/${output_name}" ]]; then
            archived="${run_dir}/${output_name}.incomplete.$(date +%s)"
            mv "${run_dir}/${output_name}" "${archived}"
        fi
        set +e
        (
            cd "${run_dir}"
            "${MUMAX_BIN}" -f "${stage}.mx3"
        )
        rc=$?
        set -e
        if [[ "${rc}" -ne 0 \
              || ! -s "${run_dir}/${output_name}/${required_initial}" \
              || ! -s "${run_dir}/${output_name}/${required_final}" \
              || ! -s "${run_dir}/${output_name}/table.txt" \
              || ! -s "${run_dir}/${output_name}/log.txt" ]]; then
            status=failed_validation
            [[ "${rc}" -ne 0 ]] || rc=3
        fi
    fi

    end_s="$(date +%s)"
    finished_at="$(date -Iseconds)"
    duration=$((end_s - start_s))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${stage}" "${BASE_TOKEN}" "${run_dir}" "${status}" "${rc}" \
        "${duration}" "${started_at}" "${finished_at}" "${MUMAX_VERSION}" \
        "${MUMAX_COMMIT}" "${MUMAX_SHA256}" "${HOST_NAME}" "${GPU_INFO}" \
        >> "${TEMP_STATUS}"
    if [[ "${rc}" -ne 0 ]]; then
        mv "${TEMP_STATUS}" "${STATUS_FILE}"
        exit "${rc}"
    fi
}

RELAX_DIR="${OOD_ROOT}/shared_relax/base${BASE_TOKEN}"
test -s "${RELAX_DIR}/relax.mx3" || {
    echo "Missing relaxation input: ${RELAX_DIR}/relax.mx3" >&2
    exit 2
}
run_input relax "${RELAX_DIR}" relax.out m_analytic_initial.ovf m_initial.ovf

mapfile -t BRANCH_DIRS < <(
    find "${OOD_ROOT}" -mindepth 1 -maxdepth 1 -type d \
        -name "r_*_base${BASE_TOKEN}_tr*_T${TARGET_TEMP_TOKEN}K_*" -print | sort
)
if [[ "${#BRANCH_DIRS[@]}" -ne 5 ]]; then
    echo "Expected five branches for base ${BASE_TOKEN}, found ${#BRANCH_DIRS[@]}" >&2
    exit 3
fi
for run_dir in "${BRANCH_DIRS[@]}"; do
    test -s "${run_dir}/run.mx3" || {
        echo "Missing branch input: ${run_dir}/run.mx3" >&2
        exit 2
    }
    run_input run "${run_dir}" run.out m_initial.ovf m_final.ovf
done

mv "${TEMP_STATUS}" "${STATUS_FILE}"
echo "base=${BASE_TOKEN} status=complete branches=${#BRANCH_DIRS[@]} host=${HOST_NAME} gpu=${GPU_INFO}"
