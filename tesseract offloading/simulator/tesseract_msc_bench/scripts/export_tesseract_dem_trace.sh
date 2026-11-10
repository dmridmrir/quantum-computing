#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
    echo "usage: $0 MODEL_DEM SYNDROMES OBSERVABLES OUTPUT_JSON [SHOT_INDEX] [Q_BITS]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TESSERACT_ROOT="${TESSERACT_ROOT:-${REPOSITORY_ROOT}/third_party/tesseract-decoder}"
TESSERACT_BIN="${TESSERACT_BIN:-${TESSERACT_ROOT}/cmake-build/tesseract}"
DEM_PATH="$(realpath "$1")"
SYNDROME_PATH="$(realpath "$2")"
OBSERVABLE_PATH="$(realpath "$3")"
TRACE_PATH="$(realpath -m "$4")"
SHOT_INDEX="${5:-0}"
Q_BITS="${6:-16}"
SHOT_FORMAT="${TESSERACT_SHOT_FORMAT:-}"
IO_CHUNK_SHOTS="${TESSERACT_IO_CHUNK_SHOTS:-1}"
SYNDROME_ONLY="${TESSERACT_TRACE_SYNDROME_ONLY:-1}"

for path in "${DEM_PATH}" "${SYNDROME_PATH}" "${OBSERVABLE_PATH}"; do
    if [[ ! -f "${path}" ]]; then
        echo "MSC workload input not found: ${path}" >&2
        exit 1
    fi
done
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
if ! [[ "${SHOT_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "SHOT_INDEX must be a non-negative integer" >&2
    exit 2
fi
if ! [[ "${IO_CHUNK_SHOTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TESSERACT_IO_CHUNK_SHOTS must be a positive integer" >&2
    exit 2
fi
if [[ ! -x "${TESSERACT_BIN}" ]]; then
    "${SCRIPT_DIR}/build_tesseract.sh"
fi

mkdir -p "$(dirname "${TRACE_PATH}")"
PREDICTION_OUTPUT="$(mktemp)"
trap 'rm -f -- "${PREDICTION_OUTPUT}"' EXIT

TESSERACT_DPU_TRACE_PATH="${TRACE_PATH}" \
TESSERACT_DPU_TRACE_Q_BITS="${Q_BITS}" \
TESSERACT_DPU_TRACE_SYNDROME_ONLY="${SYNDROME_ONLY}" \
    "${TESSERACT_BIN}" \
    --dem "${DEM_PATH}" \
    --in "${SYNDROME_PATH}" \
    --in-format "${SHOT_FORMAT}" \
    --obs-in "${OBSERVABLE_PATH}" \
    --obs-in-format "${SHOT_FORMAT}" \
    --shot-range-begin "${SHOT_INDEX}" \
    --shot-range-end "$((SHOT_INDEX + 1))" \
    --io-chunk-shots "${IO_CHUNK_SHOTS}" \
    --threads 1 \
    --out "${PREDICTION_OUTPUT}" \
    --out-format 01

echo "MSC DEM/syndrome trace: ${TRACE_PATH}"
