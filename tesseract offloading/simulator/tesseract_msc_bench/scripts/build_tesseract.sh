#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TESSERACT_ROOT="${TESSERACT_ROOT:-${REPOSITORY_ROOT}/third_party/tesseract-decoder}"
BUILD_DIR="${TESSERACT_BUILD_DIR:-${TESSERACT_ROOT}/cmake-build}"
BOOST_ARGS=()

for command in cmake ninja; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "missing required command: ${command}" >&2
        exit 1
    fi
done

if [[ ! -f "${TESSERACT_ROOT}/CMakeLists.txt" ]]; then
    echo "Tesseract source not found at ${TESSERACT_ROOT}" >&2
    exit 1
fi

if [[ -n "${TESSERACT_BOOST_INCLUDE:-}" ]]; then
    BOOST_ARGS=(
        -D "Boost_NO_SYSTEM_PATHS=ON"
        -D "Boost_INCLUDE_DIR=${TESSERACT_BOOST_INCLUDE}"
    )
fi

if [[ ! -f "${BUILD_DIR}/build.ninja" ]]; then
    CMAKE_BUILD_PARALLEL_LEVEL=1 cmake \
        -S "${TESSERACT_ROOT}" \
        -B "${BUILD_DIR}" \
        -G Ninja \
        -D "CMAKE_BUILD_TYPE=Release" \
        -D "TESSERACT_BUILD_PYTHON=OFF" \
        "${BOOST_ARGS[@]}"
fi

cmake --build "${BUILD_DIR}" --target tesseract --parallel 1
echo "Tesseract C++ decoder: ${BUILD_DIR}/tesseract"
