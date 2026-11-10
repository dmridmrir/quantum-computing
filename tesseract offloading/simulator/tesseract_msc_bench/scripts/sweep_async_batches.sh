#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
    echo "usage: $0 TRACE_JSON RESULT_DIR [REPETITIONS] [TASKLETS] [BATCH_SIZES_CSV]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRACE_PATH="$(realpath "$1")"
RESULT_DIR="$(realpath -m "$2")"
REPETITIONS="${3:-3}"
TASKLETS="${4:-16}"
BATCH_SIZES_CSV="${5:-8,16,32,64,128}"

"${SCRIPT_DIR}/build_async_host.sh" "${TASKLETS}"
"${SCRIPT_DIR}/setup_upmem_host_runtime.sh"
source "${PROJECT_ROOT}/upmem_sdk/upmem_env.sh" simulator
BUILD_ROOT="${PROJECT_ROOT}/uPIMulator/benchmark/build/TESSERACT/async_host"
export LD_LIBRARY_PATH="${BUILD_ROOT}/runtime/python3.10/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
mkdir -p "${RESULT_DIR}"
SUMMARY="${RESULT_DIR}/batch_sweep.csv"
echo "batch_size,candidate_throughput_per_second,end_to_end_seconds,h2d_bytes" > "${SUMMARY}"

IFS=',' read -r -a BATCH_SIZES <<< "${BATCH_SIZES_CSV}"
for batch_size in "${BATCH_SIZES[@]}"; do
    case "${batch_size}" in
        ''|*[!0-9]*) echo "invalid batch size: ${batch_size}" >&2; exit 2 ;;
    esac
    RESULT="${RESULT_DIR}/batch_${batch_size}.csv"
    "${BUILD_ROOT}/tesseract_async_host" \
        "${TRACE_PATH}" \
        "${BUILD_ROOT}/tesseract_dpu_t${TASKLETS}" \
        "${RESULT}" \
        "${batch_size}" \
        "${REPETITIONS}"
    throughput="$(awk -F, '$1 == "candidate_throughput_per_second" {print $2}' "${RESULT}")"
    seconds="$(awk -F, '$1 == "end_to_end_seconds" {print $2}' "${RESULT}")"
    bytes="$(awk -F, '$1 == "h2d_bytes" {print $2}' "${RESULT}")"
    echo "${batch_size},${throughput},${seconds},${bytes}" >> "${SUMMARY}"
done

echo "async batch sweep: ${SUMMARY}"
