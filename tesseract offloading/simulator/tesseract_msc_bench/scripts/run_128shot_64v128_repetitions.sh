#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 MODEL_DEM SYNDROMES_01 OBSERVABLES_01 RESULT_DIR REPETITIONS" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DEM="$(realpath "$1")"
SYNDROMES_01="$(realpath "$2")"
OBSERVABLES_01="$(realpath "$3")"
RESULT_DIR="$(realpath -m "$4")"
REPETITIONS="$5"
if ! [[ "${REPETITIONS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPETITIONS must be a positive integer" >&2
    exit 2
fi
if [[ -e "${RESULT_DIR}/runs.csv" ]]; then
    echo "refusing to overwrite existing results: ${RESULT_DIR}/runs.csv" >&2
    exit 1
fi

mkdir -p "${RESULT_DIR}"
echo "repeat,dpu_count,shots,cpu_threads,io_chunk_shots,process_wall_seconds,throughput_shot_per_s,predictions_match_cpu" > "${RESULT_DIR}/runs.csv"

for ((rep = 1; rep <= REPETITIONS; ++rep)); do
    if (( rep % 2 == 1 )); then
        counts=(64 128)
    else
        counts=(128 64)
    fi
    for dpu_count in "${counts[@]}"; do
        run_dir="${RESULT_DIR}/r${rep}_dpu${dpu_count}"
        "${SCRIPT_DIR}/run_live_dem_comparison.sh" \
            "${MODEL_DEM}" "${SYNDROMES_01}" "${OBSERVABLES_01}" "${run_dir}" \
            0 128 16 16 16 "${dpu_count}" 128
        awk -F, -v rep="${rep}" -v dpus="${dpu_count}" \
            '$1 == "dpu_live" { print rep "," dpus "," $2 "," $4 "," $16 "," $11 "," $12 "," $15 }' \
            "${run_dir}/comparison.csv" >> "${RESULT_DIR}/runs.csv"
    done
done
