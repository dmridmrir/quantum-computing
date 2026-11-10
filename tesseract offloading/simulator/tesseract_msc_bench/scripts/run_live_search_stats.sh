#!/usr/bin/env bash
# Live DPU mode with --threads 1 --inflight-shots 1 over one shot range,
# recording per-shot search statistics (thread CPU seconds, DPU wait, candidates)
# to RESULT_DIR/search_stats.csv. Several ranges can run concurrently because
# thread CPU time is unaffected by contention.
#
# usage: run_live_search_stats.sh MODEL_DEM SYNDROMES OBSERVABLES RESULT_DIR SHOT_BEGIN NUM_SHOTS [TASKLETS] [Q_BITS]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
TESSERACT_BIN="${TESSERACT_BIN:-${REPOSITORY_ROOT}/third_party/tesseract-decoder/cmake-build/tesseract}"

DEM_PATH="$(realpath "$1")"
SYNDROME_PATH="$(realpath "$2")"
OBSERVABLE_PATH="$(realpath "$3")"
RESULT_DIR="$(realpath -m "$4")"
SHOT_BEGIN="$5"
NUM_SHOTS="$6"
TASKLETS="${7:-16}"
Q_BITS="${8:-16}"
SHOT_END="$((SHOT_BEGIN + NUM_SHOTS))"

if [[ "${SYNDROME_PATH}" == *.b8 ]]; then SHOT_FORMAT=b8; else SHOT_FORMAT=01; fi

source "${PROJECT_ROOT}/upmem_sdk/upmem_env.sh" "${TESSERACT_UPMEM_BACKEND:-simulator}"
BUILD_ROOT="${PROJECT_ROOT}/uPIMulator/benchmark/build/TESSERACT/async_host"
export LD_LIBRARY_PATH="${BUILD_ROOT}/runtime/python3.10/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
PRELOAD_DIR="$(mktemp -d)"
trap 'rm -f -- "${PRELOAD_DIR}/libtesseract_dpu_live.so"; rmdir -- "${PRELOAD_DIR}"' EXIT
ln -s "${BUILD_ROOT}/libtesseract_dpu_live.so" "${PRELOAD_DIR}/libtesseract_dpu_live.so"

mkdir -p "${RESULT_DIR}"
rm -f "${RESULT_DIR}/search_stats.csv"
start="$(date +%s)"
env TESSERACT_DPU_LIVE=1 \
    TESSERACT_DPU_BINARY="${BUILD_ROOT}/tesseract_dpu_t${TASKLETS}" \
    TESSERACT_DPU_LIVE_Q_BITS="${Q_BITS}" \
    TESSERACT_DPU_LIVE_STATS=1 \
    TESSERACT_SEARCH_STATS_PATH="${RESULT_DIR}/search_stats.csv" \
    LD_PRELOAD="${PRELOAD_DIR}/libtesseract_dpu_live.so" \
    "${TESSERACT_BIN}" \
    --dem "${DEM_PATH}" \
    --in "${SYNDROME_PATH}" --in-format "${SHOT_FORMAT}" \
    --obs-in "${OBSERVABLE_PATH}" --obs-in-format "${SHOT_FORMAT}" \
    --shot-range-begin "${SHOT_BEGIN}" --shot-range-end "${SHOT_END}" \
    --io-chunk-shots "${TESSERACT_IO_CHUNK_SHOTS:-65536}" \
    --threads 1 --inflight-shots 1 \
    --out "${RESULT_DIR}/predictions.01" --out-format 01 \
    --stats-out "${RESULT_DIR}/stats.json" \
    2> "${RESULT_DIR}/dpu_live.log"
echo "live shots ${SHOT_BEGIN}..${SHOT_END} done in $(( $(date +%s) - start ))s: ${RESULT_DIR}/search_stats.csv"
