#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
    echo "usage: $0 MODEL_DEM SYNDROMES_01 OBSERVABLES_01 RESULT_DIR [NUM_SHOTS] [CPU_THREADS]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TESSERACT_ROOT="${TESSERACT_ROOT:-${REPOSITORY_ROOT}/third_party/tesseract-decoder}"
TESSERACT_BIN="${TESSERACT_BIN:-${TESSERACT_ROOT}/cmake-build/tesseract}"
DEM_PATH="$(realpath "$1")"
SYNDROME_PATH="$(realpath "$2")"
OBSERVABLE_PATH="$(realpath "$3")"
RESULT_DIR="$(realpath -m "$4")"

for path in "${DEM_PATH}" "${SYNDROME_PATH}" "${OBSERVABLE_PATH}"; do
    if [[ ! -f "${path}" ]]; then
        echo "MSC workload input not found: ${path}" >&2
        exit 1
    fi
done
NUM_SHOTS="${5:-}"
CPU_THREADS="${6:-1}"
IO_CHUNK_SHOTS="${TESSERACT_IO_CHUNK_SHOTS:-65536}"
SHOT_FORMAT="${TESSERACT_SHOT_FORMAT:-}"
if [[ -z "${SHOT_FORMAT}" ]]; then
    if [[ "${SYNDROME_PATH}" == *.b8 && "${OBSERVABLE_PATH}" == *.b8 ]]; then
        SHOT_FORMAT=b8
    elif [[ "${SYNDROME_PATH}" == *.01 && "${OBSERVABLE_PATH}" == *.01 ]]; then
        SHOT_FORMAT=01
    else
        echo "cannot infer matching shot format from input extensions; set TESSERACT_SHOT_FORMAT" >&2
        exit 2
    fi
fi
if [[ -z "${NUM_SHOTS}" ]]; then
    if [[ "${SHOT_FORMAT}" != 01 ]]; then
        echo "NUM_SHOTS is required for non-01 input" >&2
        exit 2
    fi
    AVAILABLE_SHOTS="$(wc -l < "${SYNDROME_PATH}")"
    OBSERVABLE_SHOTS="$(wc -l < "${OBSERVABLE_PATH}")"
    if (( AVAILABLE_SHOTS != OBSERVABLE_SHOTS )); then
        echo "inputs contain different shot counts: ${AVAILABLE_SHOTS}/${OBSERVABLE_SHOTS}" >&2
        exit 2
    fi
    NUM_SHOTS="${AVAILABLE_SHOTS}"
fi
if ! [[ "${NUM_SHOTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_SHOTS must be a positive integer" >&2
    exit 2
fi
if ! [[ "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CPU_THREADS must be a positive integer" >&2
    exit 2
fi
if ! [[ "${IO_CHUNK_SHOTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TESSERACT_IO_CHUNK_SHOTS must be a positive integer" >&2
    exit 2
fi
if [[ ! -x "${TESSERACT_BIN}" ]]; then
    "${SCRIPT_DIR}/build_tesseract.sh"
fi

CONFIG_ARGS=()
case "${TESSERACT_PAPER_CONFIG:-}" in
  short)
    CONFIG_ARGS=(--beam 15 --beam-climbing --no-revisit-dets --pqlimit 200000
                 --num-det-orders 16 --det-order-index --det-order-seed 2384753)
    ;;
  long)
    CONFIG_ARGS=(--beam 20 --beam-climbing --no-revisit-dets --pqlimit 1000000
                 --num-det-orders 21 --det-order-index --det-order-seed 2384753)
    ;;
  "") ;;
  *) echo "TESSERACT_PAPER_CONFIG must be short, long, or empty" >&2; exit 2 ;;
esac

mkdir -p "${RESULT_DIR}"
STATS_PATH="${RESULT_DIR}/stats.json"
PREDICTIONS_PATH="${RESULT_DIR}/predictions.01"
SUMMARY_PATH="${RESULT_DIR}/summary.csv"
WALL_SECONDS_PATH="${RESULT_DIR}/wall_seconds.txt"
unset TESSERACT_DPU_TRACE_PATH TESSERACT_DPU_TRACE_Q_BITS
unset TESSERACT_DPU_LIVE TESSERACT_DPU_RESIDENT TESSERACT_DPU_BINARY
unset TESSERACT_DPU_LIVE_Q_BITS TESSERACT_DPU_LIVE_STATS

/usr/bin/time -f '%e' -o "${WALL_SECONDS_PATH}" \
    "${TESSERACT_BIN}" \
    --dem "${DEM_PATH}" \
    --in "${SYNDROME_PATH}" \
    --in-format "${SHOT_FORMAT}" \
    --obs-in "${OBSERVABLE_PATH}" \
    --obs-in-format "${SHOT_FORMAT}" \
    --shot-range-begin 0 \
    --shot-range-end "${NUM_SHOTS}" \
    --io-chunk-shots "${IO_CHUNK_SHOTS}" \
    --threads "${CPU_THREADS}" \
    "${CONFIG_ARGS[@]}" \
    --out "${PREDICTIONS_PATH}" \
    --out-format 01 \
    --stats-out "${STATS_PATH}"

COMPLETED_SHOTS="$(sed -n 's/.*"num_shots":\([0-9][0-9]*\).*/\1/p' "${STATS_PATH}")"
DECODER_SECONDS="$(sed -n 's/.*"total_time_seconds":\([0-9.eE+-]*\).*/\1/p' "${STATS_PATH}")"
if [[ -z "${COMPLETED_SHOTS}" || -z "${DECODER_SECONDS}" ]]; then
    echo "failed to parse Tesseract stats: ${STATS_PATH}" >&2
    exit 1
fi
if ! awk -v value="${DECODER_SECONDS}" 'BEGIN { exit !(value > 0) }'; then
    echo "invalid decoder timing (check WSL clock): ${DECODER_SECONDS}" >&2
    exit 1
fi

WALL_SECONDS="$(<"${WALL_SECONDS_PATH}")"
if (( NUM_SHOTS >= 1000 )) &&
   ! awk -v value="${WALL_SECONDS}" 'BEGIN { exit !(value >= 0.5) }'; then
    echo "implausible process wall time (check WSL clock): ${WALL_SECONDS}" >&2
    exit 1
fi
if ! awk -v wall="${WALL_SECONDS}" -v service="${DECODER_SECONDS}" \
    -v threads="${CPU_THREADS}" 'BEGIN { exit !(wall >= 0.9 * service / threads) }'; then
    echo "process wall time is below conservative parallel-work lower bound: ${WALL_SECONDS}" >&2
    exit 1
fi
DECODER_SERVICE_RATE="$(awk -v shots="${COMPLETED_SHOTS}" -v seconds="${DECODER_SECONDS}" \
    'BEGIN { if (seconds == 0) print "inf"; else printf "%.6f", shots / seconds }')"
WALL_THROUGHPUT="$(awk -v shots="${COMPLETED_SHOTS}" -v seconds="${WALL_SECONDS}" \
    'BEGIN { if (seconds == 0) print "inf"; else printf "%.6f", shots / seconds }')"

{
    echo "metric,value"
    echo "mode,cpu_only_msc_dem"
    echo "paper_config,${TESSERACT_PAPER_CONFIG:-none}"
    echo "threads,${CPU_THREADS}"
    echo "io_chunk_shots,${IO_CHUNK_SHOTS}"
    echo "shot_format,${SHOT_FORMAT}"
    echo "completed_shots,${COMPLETED_SHOTS}"
    echo "decoder_seconds,${DECODER_SECONDS}"
    echo "summed_decoder_cpu_seconds,${DECODER_SECONDS}"
    echo "per_cpu_service_rate_per_second,${DECODER_SERVICE_RATE}"
    echo "process_wall_seconds,${WALL_SECONDS}"
    echo "process_wall_throughput_per_second,${WALL_THROUGHPUT}"
} > "${SUMMARY_PATH}"

echo "MSC DEM CPU-only baseline: ${SUMMARY_PATH}"
