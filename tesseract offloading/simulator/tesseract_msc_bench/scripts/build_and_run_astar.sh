#!/usr/bin/env bash
# Build the DPU-resident A* kernel and run one syndrome through the
# cycle-level uPIMulator. Candidate 0 of the trace is the syndrome; the Go
# oracle in tesseract_astar.go must match the DPU output byte-for-byte.
#
# usage: TESSERACT_TRACE=traces/x.json build_and_run_astar.sh [TASKLETS] [CAPACITY] [DET_BEAM] [MAX_EXPANSIONS] [REVERSE_ORDER] [NO_REVISIT] [PQLIMIT] [VISITED_CAPACITY]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="${PROJECT_ROOT}/uPIMulator"
SDK_ROOT="${PROJECT_ROOT}/upmem_sdk"

TASKLETS="${1:-16}"
CAPACITY="${2:-1048576}"
DET_BEAM="${3:--1}"
MAX_EXPANSIONS="${4:--1}"
REVERSE_ORDER="${5:-0}"
NO_REVISIT="${6:-0}"
PQLIMIT="${7:--1}"
VISITED_CAPACITY="${8:-131072}"
TIMEOUT_SECONDS="${TESSERACT_SIM_TIMEOUT:-14400}"
TRACE_PATH="${TESSERACT_TRACE:?set TESSERACT_TRACE to a tesseract-dpu-trace-v1 file}"
SIMULATION_THREADS="${TESSERACT_SIMULATION_THREADS:-1}"
LOGIC_FREQUENCY_MHZ="${TESSERACT_LOGIC_FREQUENCY_MHZ:-350}"
SKIP_BUILD="${TESSERACT_ASTAR_SKIP_BUILD:-0}"
ASTAR_GROUPS="${TESSERACT_ASTAR_GROUPS:-1}"

for command in go cmake ninja python3 timeout; do
    command -v "${command}" >/dev/null 2>&1 || { echo "missing required command: ${command}" >&2; exit 1; }
done
source "${SDK_ROOT}/upmem_env.sh" simulator
TRACE_PATH="$(realpath "${TRACE_PATH}")"
TRACE_NAME="$(basename "${TRACE_PATH}" .json)"
TRACE2_ARGS=()
if [[ -n "${TESSERACT_TRACE2:-}" ]]; then
    TRACE2_PATH="$(realpath "${TESSERACT_TRACE2}")"
    TRACE2_ARGS=(--tesseract_trace2 "${TRACE2_PATH}")
    TRACE_NAME="${TRACE_NAME}__$(basename "${TRACE2_PATH}" .json)"
fi

BENCH_BUILD="${SIM_ROOT}/benchmark/build"
SDK_BUILD="${SIM_ROOT}/sdk/build"
GO_BUILD="${SIM_ROOT}/build"
BIN_DIR="${TESSERACT_ASTAR_BIN_DIR:-${SIM_ROOT}/bin_astar_${TRACE_NAME}}"
RESULT_DIR="${SIM_ROOT}/results/astar_dpu1_t${TASKLETS}_g${ASTAR_GROUPS}_trace_${TRACE_NAME}"

rm -rf -- "${BIN_DIR:?}" "${RESULT_DIR:?}"
mkdir -p "${BENCH_BUILD}" "${SDK_BUILD}" "${GO_BUILD}" "${BIN_DIR}" "${RESULT_DIR}"

if [[ "${SKIP_BUILD}" != 1 ]]; then
    cmake -D "NR_TASKLETS=${TASKLETS}" -D "TESSERACT_ASTAR_GROUPS=${ASTAR_GROUPS}" -S "${SIM_ROOT}/benchmark" -B "${BENCH_BUILD}" -G Ninja
    cmake --build "${BENCH_BUILD}" --target TESSERACT_ASTAR_device --parallel 1
    cmake -D "NR_TASKLETS=${TASKLETS}" -S "${SIM_ROOT}/sdk" -B "${SDK_BUILD}" -G Ninja
    cmake --build "${SDK_BUILD}" --parallel 1
    GOMAXPROCS=1 go build -C "${SIM_ROOT}" -p 1 -o "${GO_BUILD}/uPIMulator" ./src
fi

start="$(date +%s)"
timeout "${TIMEOUT_SECONDS}" "${GO_BUILD}/uPIMulator" \
    --root_dirpath "${SIM_ROOT}" \
    --bin_dirpath "${BIN_DIR}" \
    --benchmark TESSERACT_ASTAR \
    --num_channels 1 --num_ranks_per_channel 1 --num_dpus_per_rank 1 \
    --num_simulation_threads "${SIMULATION_THREADS}" \
    --num_tasklets "${TASKLETS}" \
    --logic_frequency "${LOGIC_FREQUENCY_MHZ}" \
    --data_prep_params "${CAPACITY},${DET_BEAM},${MAX_EXPANSIONS},${REVERSE_ORDER},${NO_REVISIT},${PQLIMIT},${VISITED_CAPACITY}" \
    --tesseract_trace "${TRACE_PATH}" ${TRACE2_ARGS[@]+"${TRACE2_ARGS[@]}"} | tee "${RESULT_DIR}/stdout.log"
end="$(date +%s)"

cp "${BIN_DIR}/log.txt" "${BIN_DIR}/options.txt" "${RESULT_DIR}/"
CYCLES="$(awk -F': ' '/Logic\[[0-9]+_[0-9]+_[0-9]+\]_logic_cycle/ { print $2 + 0; exit }' "${BIN_DIR}/log.txt")"
INSTRUCTIONS="$(awk -F': ' '/Logic\[[0-9]+_[0-9]+_[0-9]+\]_num_instructions/ { print $2 + 0; exit }' "${BIN_DIR}/log.txt")"
ORACLE="$(grep '^TESSERACT_ASTAR oracle\[0\]:' "${RESULT_DIR}/stdout.log" | head -1)"
CANDIDATES="$(sed -n 's/.*candidates=\([0-9]*\).*/\1/p' <<< "${ORACLE}")"
EXPANSIONS="$(sed -n 's/.*expansions=\([0-9]*\).*/\1/p' <<< "${ORACLE}")"
{
    echo "metric,value"
    echo "trace_file,\"${TRACE_PATH}\""
    echo "tasklets,${TASKLETS}"
    echo "groups,${ASTAR_GROUPS}"
    echo "capacity,${CAPACITY}"
    echo "det_beam,${DET_BEAM}"
    echo "reverse_order,${REVERSE_ORDER}"
    echo "no_revisit,${NO_REVISIT}"
    echo "pqlimit,${PQLIMIT}"
    echo "visited_capacity,${VISITED_CAPACITY}"
    echo "logic_frequency_mhz,${LOGIC_FREQUENCY_MHZ}"
    echo "logic_cycles,${CYCLES}"
    echo "instructions,${INSTRUCTIONS}"
    echo "expansions,${EXPANSIONS}"
    echo "candidates,${CANDIDATES}"
    echo "dpu_seconds,$(awk -v c="${CYCLES}" -v m="${LOGIC_FREQUENCY_MHZ}" 'BEGIN { printf "%.9f", c / (m * 1e6) }')"
    echo "sim_wall_seconds,$((end - start))"
} > "${RESULT_DIR}/summary.csv"
echo "TESSERACT_ASTAR uPIMulator run passed"
cat "${RESULT_DIR}/summary.csv"
