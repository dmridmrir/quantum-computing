#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="${PROJECT_ROOT}/uPIMulator"
SDK_ROOT="${PROJECT_ROOT}/upmem_sdk"

TASKLETS="${1:-16}"
CANDIDATES="${2:-64}"
DETECTORS="${3:-256}"
ERRORS="${4:-512}"
SEED="${5:-42}"
Q_BITS="${6:-16}"
TIMEOUT_SECONDS="${TESSERACT_SIM_TIMEOUT:-900}"
TRACE_PATH="${TESSERACT_TRACE:-}"
NUM_CHANNELS="${TESSERACT_NUM_CHANNELS:-1}"
NUM_RANKS_PER_CHANNEL="${TESSERACT_NUM_RANKS_PER_CHANNEL:-1}"
NUM_DPUS_PER_RANK="${TESSERACT_NUM_DPUS_PER_RANK:-1}"
SIMULATION_THREADS="${TESSERACT_SIMULATION_THREADS:-16}"
LOGIC_FREQUENCY_MHZ="${TESSERACT_LOGIC_FREQUENCY_MHZ:-350}"
TRACE_ARGS=()

for value_name in TASKLETS NUM_CHANNELS NUM_RANKS_PER_CHANNEL NUM_DPUS_PER_RANK SIMULATION_THREADS LOGIC_FREQUENCY_MHZ; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer (got ${value})" >&2
        exit 2
    fi
done
NUM_DPUS="$((NUM_CHANNELS * NUM_RANKS_PER_CHANNEL * NUM_DPUS_PER_RANK))"

for command in go cmake ninja python3 timeout; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "missing required command: ${command}" >&2
        exit 1
    fi
done

if [[ ! -f "${SDK_ROOT}/upmem_env.sh" ]]; then
    echo "bundled UPMEM SDK not found at ${SDK_ROOT}" >&2
    exit 1
fi

source "${SDK_ROOT}/upmem_env.sh" simulator

if [[ -n "${TRACE_PATH}" ]]; then
    if [[ ! -f "${TRACE_PATH}" ]]; then
        echo "TESSERACT trace not found: ${TRACE_PATH}" >&2
        exit 1
    fi
    TRACE_PATH="$(realpath "${TRACE_PATH}")"
    TRACE_ARGS=(--tesseract_trace "${TRACE_PATH}")
fi

BENCH_BUILD="${SIM_ROOT}/benchmark/build"
SDK_BUILD="${SIM_ROOT}/sdk/build"
GO_BUILD="${SIM_ROOT}/build"
BIN_DIR="${SIM_ROOT}/bin"
if [[ -n "${TRACE_PATH}" ]]; then
    TRACE_NAME="$(basename "${TRACE_PATH}" .json)"
    RESULT_DIR="${SIM_ROOT}/results/dpu${NUM_DPUS}_t${TASKLETS}_trace_${TRACE_NAME}"
    CSV_TRACE_PATH="${TRACE_PATH//\"/\"\"}"
else
    RESULT_DIR="${SIM_ROOT}/results/dpu${NUM_DPUS}_t${TASKLETS}_c${CANDIDATES}_d${DETECTORS}_e${ERRORS}_s${SEED}_q${Q_BITS}"
fi

rm -rf -- "${BIN_DIR:?}" "${RESULT_DIR:?}"
mkdir -p "${BENCH_BUILD}" "${SDK_BUILD}" "${GO_BUILD}" "${BIN_DIR}" "${RESULT_DIR}"

cmake -D "NR_TASKLETS=${TASKLETS}" \
    -S "${SIM_ROOT}/benchmark" -B "${BENCH_BUILD}" -G Ninja
cmake --build "${BENCH_BUILD}" --target TESSERACT_device --parallel 1

cmake -D "NR_TASKLETS=${TASKLETS}" \
    -S "${SIM_ROOT}/sdk" -B "${SDK_BUILD}" -G Ninja
cmake --build "${SDK_BUILD}" --parallel 1

GOMAXPROCS=1 go test -C "${SIM_ROOT}" -p 1 ./...
GOMAXPROCS=1 go build -C "${SIM_ROOT}" -p 1 -o "${GO_BUILD}/uPIMulator" ./src

timeout "${TIMEOUT_SECONDS}" "${GO_BUILD}/uPIMulator" \
    --root_dirpath "${SIM_ROOT}" \
    --bin_dirpath "${BIN_DIR}" \
    --benchmark TESSERACT \
    --num_channels "${NUM_CHANNELS}" \
    --num_ranks_per_channel "${NUM_RANKS_PER_CHANNEL}" \
    --num_dpus_per_rank "${NUM_DPUS_PER_RANK}" \
    --num_simulation_threads "${SIMULATION_THREADS}" \
    --num_tasklets "${TASKLETS}" \
    --logic_frequency "${LOGIC_FREQUENCY_MHZ}" \
    --data_prep_params "${CANDIDATES},${DETECTORS},${ERRORS},${SEED},${Q_BITS}" \
    "${TRACE_ARGS[@]}"

cp "${BIN_DIR}/log.txt" "${RESULT_DIR}/log.txt"
cp "${BIN_DIR}/args.txt" "${RESULT_DIR}/args.txt"
cp "${BIN_DIR}/options.txt" "${RESULT_DIR}/options.txt"

read -r OBSERVED_DPUS MIN_LOGIC_CYCLES MAX_LOGIC_CYCLES SUM_LOGIC_CYCLES <<< "$(
    awk -F': ' '
        /Logic\[[0-9]+_[0-9]+_[0-9]+\]_logic_cycle/ {
            value = $2 + 0
            count++
            sum += value
            if (count == 1 || value < min) min = value
            if (count == 1 || value > max) max = value
        }
        END { print count + 0, min + 0, max + 0, sum + 0 }
    ' "${BIN_DIR}/log.txt"
)"
SUM_INSTRUCTIONS="$(awk -F': ' '
    /Logic\[[0-9]+_[0-9]+_[0-9]+\]_num_instructions/ { sum += $2 + 0 }
    END { print sum + 0 }
' "${BIN_DIR}/log.txt")"
if (( OBSERVED_DPUS != NUM_DPUS )); then
    echo "simulator reported ${OBSERVED_DPUS} DPUs, expected ${NUM_DPUS}" >&2
    exit 1
fi
AVERAGE_LOGIC_CYCLES="$(awk -v sum="${SUM_LOGIC_CYCLES}" -v count="${OBSERVED_DPUS}" \
    'BEGIN { printf "%.3f", sum / count }')"
DPU_KERNEL_SECONDS_PER_WAVE="$(awk -v cycles="${MAX_LOGIC_CYCLES}" -v mhz="${LOGIC_FREQUENCY_MHZ}" \
    'BEGIN { printf "%.12f", cycles / (mhz * 1000000) }')"
DPU_KERNEL_BATCH_THROUGHPUT="$(awk -v dpus="${NUM_DPUS}" -v seconds="${DPU_KERNEL_SECONDS_PER_WAVE}" \
    'BEGIN { if (seconds == 0) print "inf"; else printf "%.6f", dpus / seconds }')"
if [[ -n "${TRACE_PATH}" ]]; then
    CANDIDATES_IN_BATCH="$(python3 - "${TRACE_PATH}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    trace = json.load(f)
print(len(trace["candidate_detector_offsets"]) - 1)
PY
)"
else
    CANDIDATES_IN_BATCH="${CANDIDATES}"
fi
DPU_KERNEL_CANDIDATE_THROUGHPUT="$(awk -v batches="${DPU_KERNEL_BATCH_THROUGHPUT}" -v candidates="${CANDIDATES_IN_BATCH}" \
    'BEGIN { printf "%.6f", batches * candidates }')"

{
    echo "metric,value"
    echo "num_channels,${NUM_CHANNELS}"
    echo "num_ranks_per_channel,${NUM_RANKS_PER_CHANNEL}"
    echo "num_dpus_per_rank,${NUM_DPUS_PER_RANK}"
    echo "num_dpus,${NUM_DPUS}"
    echo "simulation_host_threads,${SIMULATION_THREADS}"
    echo "tasklets,${TASKLETS}"
    if [[ -n "${TRACE_PATH}" ]]; then
        echo "workload,trace"
        echo "trace_file,\"${CSV_TRACE_PATH}\""
    else
        echo "workload,synthetic"
        echo "candidates,${CANDIDATES}"
        echo "detectors,${DETECTORS}"
        echo "errors,${ERRORS}"
        echo "seed,${SEED}"
        echo "q_bits,${Q_BITS}"
    fi
    echo "candidates_per_dpu_batch,${CANDIDATES_IN_BATCH}"
    echo "logic_frequency_mhz,${LOGIC_FREQUENCY_MHZ}"
    echo "logic_cycles_min_per_dpu,${MIN_LOGIC_CYCLES}"
    echo "logic_cycles_max_per_dpu,${MAX_LOGIC_CYCLES}"
    echo "logic_cycles_average_per_dpu,${AVERAGE_LOGIC_CYCLES}"
    echo "instructions_sum_all_dpus,${SUM_INSTRUCTIONS}"
    echo "dpu_kernel_seconds_per_wave,${DPU_KERNEL_SECONDS_PER_WAVE}"
    echo "dpu_kernel_only_batch_throughput_per_second,${DPU_KERNEL_BATCH_THROUGHPUT}"
    echo "dpu_kernel_only_candidate_throughput_per_second,${DPU_KERNEL_CANDIDATE_THROUGHPUT}"
} > "${RESULT_DIR}/summary.csv"

echo "TESSERACT uPIMulator smoke run passed"
echo "results: ${RESULT_DIR}"
