#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
TESSERACT_BIN="${TESSERACT_BIN:-${REPOSITORY_ROOT}/third_party/tesseract-decoder/cmake-build/tesseract}"
WORKLOAD="${PROJECT_ROOT}/workloads/msc_grafted_d3_d7"
OUT="${1:-${PROJECT_ROOT}/results/verification/cpu_modes_20260921}"

mkdir -p "${OUT}"
for spec in "base:0:1" "chunk1:1:1" "chunk7:7:1" "thread16:7:16"; do
    IFS=: read -r name chunk threads <<<"${spec}"
    args=(
        --dem "${WORKLOAD}/model.dem"
        --in "${WORKLOAD}/syndromes.01" --in-format 01
        --obs-in "${WORKLOAD}/observables.01" --obs-in-format 01
        --shot-range-begin 0 --shot-range-end 64
        --threads "${threads}"
        --out "${OUT}/${name}.01" --out-format 01
        --stats-out "${OUT}/${name}.json"
    )
    if ((chunk)); then
        args+=(--io-chunk-shots "${chunk}")
    fi
    env -u TESSERACT_DPU_LIVE \
        -u TESSERACT_DPU_RESIDENT \
        -u TESSERACT_DPU_RESIDENT_ORACLE \
        "${TESSERACT_BIN}" "${args[@]}"
done

sha256sum "${OUT}"/*.01
cmp "${OUT}/base.01" "${OUT}/chunk1.01"
cmp "${OUT}/base.01" "${OUT}/chunk7.01"
cmp "${OUT}/base.01" "${OUT}/thread16.01"
cmp "${OUT}/base.01" \
    "${PROJECT_ROOT}/results/live_msc_grafted_d3_d7_all64/cpu/predictions.01"
echo "CPU_MODES_MATCH"
