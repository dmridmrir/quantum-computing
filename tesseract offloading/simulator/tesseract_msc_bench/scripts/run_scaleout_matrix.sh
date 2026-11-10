#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 8 ]]; then
    echo "usage: $0 MODEL_DEM SYNDROMES_01 OBSERVABLES_01 RESULT_DIR [NUM_SHOTS] [DPU_COUNT] [TASKLETS] [CPU_THREADS_CSV]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DEM="$(realpath "$1")"
SYNDROMES_01="$(realpath "$2")"
OBSERVABLES_01="$(realpath "$3")"
RESULT_DIR="$(realpath -m "$4")"
NUM_SHOTS="${5:-5120}"
DPU_COUNT="${6:-2560}"
TASKLETS="${7:-16}"
CPU_THREADS_CSV="${8:-8,16}"
Q_BITS="${TESSERACT_Q_BITS:-16}"

for value_name in NUM_SHOTS DPU_COUNT TASKLETS Q_BITS; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer" >&2
        exit 2
    fi
done
if (( NUM_SHOTS < DPU_COUNT )); then
    echo "NUM_SHOTS (${NUM_SHOTS}) must be at least DPU_COUNT (${DPU_COUNT}) so every DPU can receive one syndrome" >&2
    exit 2
fi

IFS=',' read -r -a CPU_THREADS <<< "${CPU_THREADS_CSV}"
if (( ${#CPU_THREADS[@]} == 0 )); then
    echo "CPU_THREADS_CSV must contain at least one thread count" >&2
    exit 2
fi
for threads in "${CPU_THREADS[@]}"; do
    if ! [[ "${threads}" =~ ^[1-9][0-9]*$ ]]; then
        echo "invalid CPU thread count: ${threads}" >&2
        exit 2
    fi
done

mkdir -p "${RESULT_DIR}"
REPEATED_DIR="${RESULT_DIR}/repeated_workload_${NUM_SHOTS}"
python3 "${SCRIPT_DIR}/repeat_01_workload.py" \
    --syndromes "${SYNDROMES_01}" \
    --observables "${OBSERVABLES_01}" \
    --out-dir "${REPEATED_DIR}" \
    --shots "${NUM_SHOTS}"

MATRIX_PATH="${RESULT_DIR}/throughput_matrix.csv"
HEADER_WRITTEN=false
for threads in "${CPU_THREADS[@]}"; do
    RUN_DIR="${RESULT_DIR}/cpu${threads}_dpu${DPU_COUNT}"
    "${SCRIPT_DIR}/run_live_dem_comparison.sh" \
        "${MODEL_DEM}" \
        "${REPEATED_DIR}/syndromes.01" \
        "${REPEATED_DIR}/observables.01" \
        "${RUN_DIR}" \
        0 "${NUM_SHOTS}" "${TASKLETS}" "${Q_BITS}" "${threads}" "${DPU_COUNT}"
    if [[ "${HEADER_WRITTEN}" == false ]]; then
        printf 'cpu_configuration,' > "${MATRIX_PATH}"
        head -n 1 "${RUN_DIR}/comparison.csv" >> "${MATRIX_PATH}"
        HEADER_WRITTEN=true
    fi
    while IFS= read -r row; do
        printf '%s,%s\n' "${threads}" "${row}" >> "${MATRIX_PATH}"
    done < <(tail -n +2 "${RUN_DIR}/comparison.csv")
done

echo "CPU workers ${CPU_THREADS_CSV} versus ${DPU_COUNT}-DPU throughput matrix: ${MATRIX_PATH}"
