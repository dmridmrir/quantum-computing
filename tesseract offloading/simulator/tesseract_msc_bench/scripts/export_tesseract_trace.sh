#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "usage: $0 CIRCUIT_STIM OUTPUT_JSON [SAMPLE_SEED] [Q_BITS]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TESSERACT_ROOT="${TESSERACT_ROOT:-${REPOSITORY_ROOT}/third_party/tesseract-decoder}"
TESSERACT_BIN="${TESSERACT_BIN:-${TESSERACT_ROOT}/cmake-build/tesseract}"
CIRCUIT_PATH="$(realpath "$1")"
TRACE_PATH="$(realpath -m "$2")"
SAMPLE_SEED="${3:-1}"
Q_BITS="${4:-16}"

if [[ ! -f "${CIRCUIT_PATH}" ]]; then
    echo "Stim circuit not found: ${CIRCUIT_PATH}" >&2
    exit 1
fi
if [[ ! -x "${TESSERACT_BIN}" ]]; then
    "${SCRIPT_DIR}/build_tesseract.sh"
fi

mkdir -p "$(dirname "${TRACE_PATH}")"
PREDICTION_OUTPUT="$(mktemp)"
trap 'rm -f -- "${PREDICTION_OUTPUT}"' EXIT

TESSERACT_DPU_TRACE_PATH="${TRACE_PATH}" \
TESSERACT_DPU_TRACE_Q_BITS="${Q_BITS}" \
    "${TESSERACT_BIN}" \
    --circuit "${CIRCUIT_PATH}" \
    --sample-num-shots 1 \
    --sample-seed "${SAMPLE_SEED}" \
    --threads 1 \
    --out "${PREDICTION_OUTPUT}" \
    --out-format 01

echo "Tesseract DPU trace: ${TRACE_PATH}"
