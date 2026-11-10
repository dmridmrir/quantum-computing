#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 9 ]]; then
    echo "usage: $0 MODEL_DEM SYNDROMES_01 OBSERVABLES_01 RESULT_DIR [NUM_SHOTS] [DPU_COUNTS_CSV] [TASKLETS] [CPU_THREADS_CSV] [TARGET_DPUS]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DEM="$(realpath "$1")"
SYNDROMES_01="$(realpath "$2")"
OBSERVABLES_01="$(realpath "$3")"
RESULT_DIR="$(realpath -m "$4")"
NUM_SHOTS="${5:-64}"
DPU_COUNTS_CSV="${6:-4,8,16,32,64}"
TASKLETS="${7:-16}"
CPU_THREADS_CSV="${8:-8,16}"
TARGET_DPUS="${9:-2560}"
Q_BITS="${TESSERACT_Q_BITS:-16}"
UPMEM_BACKEND="${TESSERACT_UPMEM_BACKEND:-simulator}"
IO_CHUNK_SHOTS="${TESSERACT_IO_CHUNK_SHOTS:-65536}"

for value_name in NUM_SHOTS TASKLETS TARGET_DPUS Q_BITS IO_CHUNK_SHOTS; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer" >&2
        exit 2
    fi
done

IFS=',' read -r -a DPU_COUNTS <<< "${DPU_COUNTS_CSV}"
IFS=',' read -r -a CPU_THREADS <<< "${CPU_THREADS_CSV}"
if (( ${#DPU_COUNTS[@]} < 2 || ${#CPU_THREADS[@]} == 0 )); then
    echo "at least two DPU counts and one CPU thread count are required" >&2
    exit 2
fi
for dpu_count in "${DPU_COUNTS[@]}"; do
    if ! [[ "${dpu_count}" =~ ^[1-9][0-9]*$ ]] || (( dpu_count > NUM_SHOTS )); then
        echo "each DPU count must be positive and no larger than NUM_SHOTS (${NUM_SHOTS})" >&2
        exit 2
    fi
    if (( dpu_count > IO_CHUNK_SHOTS )); then
        echo "each DPU count must be no larger than TESSERACT_IO_CHUNK_SHOTS (${IO_CHUNK_SHOTS}) so one batch can use the full configured DPU pool" >&2
        exit 2
    fi
done
for cpu_threads in "${CPU_THREADS[@]}"; do
    if ! [[ "${cpu_threads}" =~ ^[1-9][0-9]*$ ]]; then
        echo "invalid CPU thread count: ${cpu_threads}" >&2
        exit 2
    fi
done

mkdir -p "${RESULT_DIR}"
MEASUREMENTS="${RESULT_DIR}/measurements.csv"
echo "backend,cpu_threads,dpu_count,shots,tasklets_per_dpu,q_bits,process_wall_seconds,hybrid_end_to_end_throughput_per_second,cpu_only_full_system_throughput_per_second,predictions_match_cpu,io_chunk_shots" > "${MEASUREMENTS}"

for cpu_threads in "${CPU_THREADS[@]}"; do
    for dpu_count in "${DPU_COUNTS[@]}"; do
        RUN_DIR="${RESULT_DIR}/cpu${cpu_threads}_dpu${dpu_count}"
        "${SCRIPT_DIR}/run_live_dem_comparison.sh" \
            "${MODEL_DEM}" "${SYNDROMES_01}" "${OBSERVABLES_01}" "${RUN_DIR}" \
            0 "${NUM_SHOTS}" "${TASKLETS}" "${Q_BITS}" "${cpu_threads}" "${dpu_count}" \
            "${IO_CHUNK_SHOTS}"

        CPU_RATE="$(awk -F, '$1 == "cpu_only" { print $12 }' "${RUN_DIR}/comparison.csv")"
        IFS=',' read -r LIVE_WALL LIVE_RATE MATCH <<< "$(awk -F, '$1 == "dpu_live" { print $11 "," $12 "," $15 }' "${RUN_DIR}/comparison.csv")"
        if [[ -z "${CPU_RATE}" || -z "${LIVE_WALL}" || -z "${LIVE_RATE}" || -z "${MATCH}" ]]; then
            echo "failed to parse ${RUN_DIR}/comparison.csv" >&2
            exit 1
        fi
        echo "${UPMEM_BACKEND},${cpu_threads},${dpu_count},${NUM_SHOTS},${TASKLETS},${Q_BITS},${LIVE_WALL},${LIVE_RATE},${CPU_RATE},${MATCH},${IO_CHUNK_SHOTS}" >> "${MEASUREMENTS}"
    done
done

python3 "${SCRIPT_DIR}/fit_scaleout.py" \
    "${MEASUREMENTS}" "${RESULT_DIR}/extrapolated_${TARGET_DPUS}_dpu.csv" \
    --target-dpus "${TARGET_DPUS}" --cap-by-shots

echo "measured DPU counts: ${DPU_COUNTS_CSV}"
echo "target DPU count: ${TARGET_DPUS}"
echo "results: ${RESULT_DIR}"
