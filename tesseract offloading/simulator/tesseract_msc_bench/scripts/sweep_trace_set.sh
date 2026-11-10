#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
    echo "usage: $0 TRACE_DIR RESULT_DIR [BATCH_SIZE] [REPETITIONS] [TASKLETS]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRACE_DIR="$(realpath "$1")"
RESULT_DIR="$(realpath -m "$2")"
BATCH_SIZE="${3:-64}"
REPETITIONS="${4:-1}"
TASKLETS="${5:-16}"
BUILD_ROOT="${PROJECT_ROOT}/uPIMulator/benchmark/build/TESSERACT/async_host"

"${SCRIPT_DIR}/build_async_host.sh" "${TASKLETS}"
"${SCRIPT_DIR}/setup_upmem_host_runtime.sh"
source "${PROJECT_ROOT}/upmem_sdk/upmem_env.sh" simulator
export LD_LIBRARY_PATH="${BUILD_ROOT}/runtime/python3.10/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
mkdir -p "${RESULT_DIR}"

SUMMARY="${RESULT_DIR}/trace_sweep.csv"
echo "trace,candidates,candidate_throughput_per_second,end_to_end_seconds,h2d_bytes" > "${SUMMARY}"

shopt -s nullglob
traces=("${TRACE_DIR}"/*.json)
if (( ${#traces[@]} == 0 )); then
    echo "no JSON traces found in ${TRACE_DIR}" >&2
    exit 1
fi

for trace in "${traces[@]}"; do
    name="$(basename "${trace}" .json)"
    result="${RESULT_DIR}/${name}.csv"
    "${BUILD_ROOT}/tesseract_async_host" \
        "${trace}" \
        "${BUILD_ROOT}/tesseract_dpu_t${TASKLETS}" \
        "${result}" \
        "${BATCH_SIZE}" \
        "${REPETITIONS}"
    candidates="$(awk -F, '$1 == "candidates" {print $2}' "${result}")"
    throughput="$(awk -F, '$1 == "candidate_throughput_per_second" {print $2}' "${result}")"
    seconds="$(awk -F, '$1 == "end_to_end_seconds" {print $2}' "${result}")"
    bytes="$(awk -F, '$1 == "h2d_bytes" {print $2}' "${result}")"
    echo "${name},${candidates},${throughput},${seconds},${bytes}" >> "${SUMMARY}"
done

echo "trace-set sweep: ${SUMMARY}"
