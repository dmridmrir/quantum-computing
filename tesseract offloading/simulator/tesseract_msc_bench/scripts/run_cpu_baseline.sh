#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 CIRCUIT_STIM NUM_SHOTS RESULT_DIR [SAMPLE_SEED]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TESSERACT_ROOT="${TESSERACT_ROOT:-${REPOSITORY_ROOT}/third_party/tesseract-decoder}"
TESSERACT_BIN="${TESSERACT_BIN:-${TESSERACT_ROOT}/cmake-build/tesseract}"
CIRCUIT_PATH="$(realpath "$1")"
NUM_SHOTS="$2"
RESULT_DIR="$(realpath -m "$3")"
SAMPLE_SEED="${4:-1}"

if [[ ! -f "${CIRCUIT_PATH}" ]]; then
    echo "Stim circuit not found: ${CIRCUIT_PATH}" >&2
    exit 1
fi
if ! [[ "${NUM_SHOTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_SHOTS must be a positive integer" >&2
    exit 2
fi
if [[ ! -x "${TESSERACT_BIN}" ]]; then
    "${SCRIPT_DIR}/build_tesseract.sh"
fi

mkdir -p "${RESULT_DIR}"
STATS_PATH="${RESULT_DIR}/stats.json"
PREDICTIONS_PATH="${RESULT_DIR}/predictions.01"
SUMMARY_PATH="${RESULT_DIR}/summary.csv"

# A CPU baseline must never inherit trace instrumentation from the caller.
unset TESSERACT_DPU_TRACE_PATH TESSERACT_DPU_TRACE_Q_BITS

START_NS="$(date +%s%N)"
"${TESSERACT_BIN}" \
    --circuit "${CIRCUIT_PATH}" \
    --sample-num-shots "${NUM_SHOTS}" \
    --sample-seed "${SAMPLE_SEED}" \
    --threads 1 \
    --out "${PREDICTIONS_PATH}" \
    --out-format 01 \
    --stats-out "${STATS_PATH}"
END_NS="$(date +%s%N)"

COMPLETED_SHOTS="$(sed -n 's/.*"num_shots":\([0-9][0-9]*\).*/\1/p' "${STATS_PATH}")"
DECODER_SECONDS="$(sed -n 's/.*"total_time_seconds":\([0-9.eE+-]*\).*/\1/p' "${STATS_PATH}")"
if [[ -z "${COMPLETED_SHOTS}" || -z "${DECODER_SECONDS}" ]]; then
    echo "failed to parse Tesseract stats: ${STATS_PATH}" >&2
    exit 1
fi

WALL_SECONDS="$(awk -v start="${START_NS}" -v end="${END_NS}" \
    'BEGIN { printf "%.9f", (end - start) / 1000000000 }')"
DECODER_THROUGHPUT="$(awk -v shots="${COMPLETED_SHOTS}" -v seconds="${DECODER_SECONDS}" \
    'BEGIN { if (seconds == 0) print "inf"; else printf "%.6f", shots / seconds }')"
WALL_THROUGHPUT="$(awk -v shots="${COMPLETED_SHOTS}" -v seconds="${WALL_SECONDS}" \
    'BEGIN { if (seconds == 0) print "inf"; else printf "%.6f", shots / seconds }')"
CSV_CIRCUIT_PATH="${CIRCUIT_PATH//\"/\"\"}"

{
    echo "metric,value"
    echo "mode,cpu_only"
    echo "threads,1"
    echo "circuit,\"${CSV_CIRCUIT_PATH}\""
    echo "sample_seed,${SAMPLE_SEED}"
    echo "completed_shots,${COMPLETED_SHOTS}"
    echo "decoder_seconds,${DECODER_SECONDS}"
    echo "decoder_throughput_per_second,${DECODER_THROUGHPUT}"
    echo "process_wall_seconds,${WALL_SECONDS}"
    echo "process_wall_throughput_per_second,${WALL_THROUGHPUT}"
} > "${SUMMARY_PATH}"

echo "CPU-only baseline: ${SUMMARY_PATH}"
