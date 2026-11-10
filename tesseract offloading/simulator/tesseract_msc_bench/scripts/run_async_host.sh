#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 ]]; then
    echo "usage: $0 TRACE_JSON [BATCH_SIZE] [REPETITIONS] [RESULT_DIR] [TASKLETS]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRACE_PATH="$(realpath "$1")"
BATCH_SIZE="${2:-64}"
REPETITIONS="${3:-1}"
RESULT_DIR="$(realpath -m "${4:-${PROJECT_ROOT}/results/async_host}")"
TASKLETS="${5:-16}"

"${SCRIPT_DIR}/build_async_host.sh" "${TASKLETS}"
"${SCRIPT_DIR}/setup_upmem_host_runtime.sh"
source "${PROJECT_ROOT}/upmem_sdk/upmem_env.sh" simulator
BUILD_ROOT="${PROJECT_ROOT}/uPIMulator/benchmark/build/TESSERACT/async_host"
export LD_LIBRARY_PATH="${BUILD_ROOT}/runtime/python3.10/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
mkdir -p "${RESULT_DIR}"

"${BUILD_ROOT}/tesseract_async_host" \
    "${TRACE_PATH}" \
    "${BUILD_ROOT}/tesseract_dpu_t${TASKLETS}" \
    "${RESULT_DIR}/summary.csv" \
    "${BATCH_SIZE}" \
    "${REPETITIONS}"

echo "async host result: ${RESULT_DIR}/summary.csv"
