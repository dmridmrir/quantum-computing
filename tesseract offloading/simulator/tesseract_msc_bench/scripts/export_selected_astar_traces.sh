#!/usr/bin/env bash
# Export compact resident-A* traces for global shot IDs selected from the
# complete 100M validation.  Direct byte seeks avoid rescanning earlier shots.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 SELECTED_TRACES_CSV [OUTPUT_DIR]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKLOAD="${PROJECT_ROOT}/workloads/msc_grafted_d3_d7_100m"
SELECTION="$(realpath "$1")"
OUTPUT_DIR="$(realpath -m "${2:-${PROJECT_ROOT}/results/full_validation_100m/cycle_traces}")"
TMP_DIR="${OUTPUT_DIR}/tmp"
mkdir -p "${OUTPUT_DIR}" "${TMP_DIR}"
trap 'rm -f -- "${TMP_DIR}/syndrome.b8" "${TMP_DIR}/observable.b8"' EXIT INT TERM

header="$(head -n 1 "${SELECTION}")"
if [[ "${header}" != global_shot_index,chunk,candidates,* ]]; then
    echo "unexpected selection CSV header: ${header}" >&2
    exit 2
fi

tail -n +2 "${SELECTION}" | while IFS=, read -r shot chunk candidates expansions dpu_candidates dpu_expansions rest; do
    if ! [[ "${shot}" =~ ^[0-9]+$ && "${candidates}" =~ ^[0-9]+$ && "${dpu_candidates}" =~ ^[0-9]+$ ]]; then
        echo "invalid selection row for shot=${shot} CPU candidates=${candidates} DPU candidates=${dpu_candidates}" >&2
        exit 1
    fi
    trace="${OUTPUT_DIR}/shot_$(printf '%09d' "${shot}")_cpu_${candidates}_dpu_${dpu_candidates}.json"
    if [[ -s "${trace}" ]]; then
        echo "skip existing trace: ${trace}"
        continue
    fi
    dd if="${WORKLOAD}/syndromes.b8" of="${TMP_DIR}/syndrome.b8" \
        bs=62 skip="${shot}" count=1 status=none
    dd if="${WORKLOAD}/observables.b8" of="${TMP_DIR}/observable.b8" \
        bs=1 skip="${shot}" count=1 status=none
    TESSERACT_TRACE_SYNDROME_ONLY=1 \
        "${SCRIPT_DIR}/export_tesseract_dem_trace.sh" \
        "${WORKLOAD}/model.dem" \
        "${TMP_DIR}/syndrome.b8" \
        "${TMP_DIR}/observable.b8" \
        "${trace}" 0 16
done

rm -rf -- "${TMP_DIR}"
echo "compact resident-A* traces: ${OUTPUT_DIR}"
