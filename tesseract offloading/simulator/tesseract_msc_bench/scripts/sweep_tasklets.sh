#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
    echo "usage: $0 TRACE_JSON RESULT_DIR [BATCH_SIZE] [REPETITIONS] [TASKLETS_CSV]" >&2
    exit 2
fi

TESSERACT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESSERACT_PROJECT_ROOT="$(cd "${TESSERACT_SCRIPT_DIR}/.." && pwd)"
TRACE_PATH="$(realpath "$1")"
RESULT_DIR="$(realpath -m "$2")"
BATCH_SIZE="${3:-16}"
REPETITIONS="${4:-3}"
TASKLETS_CSV="${5:-11,16,24}"
TESSERACT_BUILD_ROOT="${TESSERACT_PROJECT_ROOT}/uPIMulator/benchmark/build/TESSERACT/async_host"

"${TESSERACT_SCRIPT_DIR}/setup_upmem_host_runtime.sh"
source "${TESSERACT_PROJECT_ROOT}/upmem_sdk/upmem_env.sh" simulator
export LD_LIBRARY_PATH="${TESSERACT_BUILD_ROOT}/runtime/python3.10/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
mkdir -p "${RESULT_DIR}"
SUMMARY="${RESULT_DIR}/tasklet_sweep.csv"
echo "tasklets,candidate_throughput_per_second,end_to_end_seconds,h2d_bytes" > "${SUMMARY}"

IFS=',' read -r -a TASKLET_COUNTS <<< "${TASKLETS_CSV}"
for tasklets in "${TASKLET_COUNTS[@]}"; do
    case "${tasklets}" in
        ''|*[!0-9]*) echo "invalid tasklet count: ${tasklets}" >&2; exit 2 ;;
    esac
    if (( tasklets < 1 || tasklets > 24 )); then
        echo "tasklet count must be in [1,24]: ${tasklets}" >&2
        exit 2
    fi

    "${TESSERACT_SCRIPT_DIR}/build_async_host.sh" "${tasklets}"
    RESULT="${RESULT_DIR}/tasklets_${tasklets}.csv"
    "${TESSERACT_BUILD_ROOT}/tesseract_async_host" \
        "${TRACE_PATH}" \
        "${TESSERACT_BUILD_ROOT}/tesseract_dpu_t${tasklets}" \
        "${RESULT}" \
        "${BATCH_SIZE}" \
        "${REPETITIONS}"
    throughput="$(awk -F, '$1 == "candidate_throughput_per_second" {print $2}' "${RESULT}")"
    seconds="$(awk -F, '$1 == "end_to_end_seconds" {print $2}' "${RESULT}")"
    bytes="$(awk -F, '$1 == "h2d_bytes" {print $2}' "${RESULT}")"
    echo "${tasklets},${throughput},${seconds},${bytes}" >> "${SUMMARY}"
done

echo "tasklet sweep: ${SUMMARY}"
