#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
SDK_ROOT="${PROJECT_ROOT}/upmem_sdk"
BENCHMARK_ROOT="${PROJECT_ROOT}/uPIMulator/benchmark/TESSERACT"
BUILD_ROOT="${PROJECT_ROOT}/uPIMulator/benchmark/build/TESSERACT/async_host"
TASKLETS="${1:-16}"

UPMEM_BACKEND="${TESSERACT_UPMEM_BACKEND:-simulator}"
source "${SDK_ROOT}/upmem_env.sh" "${UPMEM_BACKEND}"
JSON_INCLUDE="${REPOSITORY_ROOT}/third_party/tesseract-decoder/cmake-build/_deps/nlohmann_json-src/include"
if [[ ! -f "${JSON_INCLUDE}/nlohmann/json.hpp" ]]; then
    "${SCRIPT_DIR}/build_tesseract.sh"
fi

mkdir -p "${BUILD_ROOT}"
DPU_BINARY="${BUILD_ROOT}/tesseract_dpu_t${TASKLETS}"
HOST_BINARY="${BUILD_ROOT}/tesseract_async_host"
LIVE_LIBRARY="${BUILD_ROOT}/libtesseract_dpu_live.so"

if [[ ! -x "${DPU_BINARY}" || \
      "${BENCHMARK_ROOT}/dpu/task.c" -nt "${DPU_BINARY}" || \
      "${BENCHMARK_ROOT}/support/common.h" -nt "${DPU_BINARY}" ]]; then
    dpu-upmem-dpurte-clang \
        -O2 \
        -D "NR_TASKLETS=${TASKLETS}" \
        -D TESSERACT_ASTAR_GROUPS=1 \
        -D BL=8 \
        -I "${BENCHMARK_ROOT}/support" \
        "${BENCHMARK_ROOT}/dpu/task.c" \
        -o "${DPU_BINARY}"
fi

if [[ ! -x "${HOST_BINARY}" || \
      "${BENCHMARK_ROOT}/host/async_host.cpp" -nt "${HOST_BINARY}" || \
      "${BENCHMARK_ROOT}/support/common.h" -nt "${HOST_BINARY}" || \
      "${REPOSITORY_ROOT}/third_party/tesseract-decoder/src/tesseract_dpu_live.h" -nt "${HOST_BINARY}" ]]; then
    g++ \
        -std=c++17 \
        -O2 \
        -pthread \
        -I "${SDK_ROOT}/include/dpu" \
        -I "${JSON_INCLUDE}" \
        -I "${BENCHMARK_ROOT}/support" \
        -I "${REPOSITORY_ROOT}/third_party/tesseract-decoder/src" \
        "${BENCHMARK_ROOT}/host/async_host.cpp" \
        -L "${SDK_ROOT}/lib" \
        -Wl,--allow-shlib-undefined \
        -ldpu \
        -o "${HOST_BINARY}"
fi

if [[ ! -f "${LIVE_LIBRARY}" || \
      "${BENCHMARK_ROOT}/host/async_host.cpp" -nt "${LIVE_LIBRARY}" || \
      "${BENCHMARK_ROOT}/support/common.h" -nt "${LIVE_LIBRARY}" || \
      "${REPOSITORY_ROOT}/third_party/tesseract-decoder/src/tesseract_dpu_live.h" -nt "${LIVE_LIBRARY}" ]]; then
    g++ \
        -std=c++17 \
        -O2 \
        -fPIC \
        -shared \
        -pthread \
        -D TESSERACT_DPU_LIVE_LIBRARY \
        -I "${SDK_ROOT}/include/dpu" \
        -I "${JSON_INCLUDE}" \
        -I "${BENCHMARK_ROOT}/support" \
        -I "${REPOSITORY_ROOT}/third_party/tesseract-decoder/src" \
        "${BENCHMARK_ROOT}/host/async_host.cpp" \
        -L "${SDK_ROOT}/lib" \
        -Wl,--allow-shlib-undefined \
        -ldpu \
        -o "${LIVE_LIBRARY}"
fi

echo "DPU binary: ${DPU_BINARY}"
echo "async host: ${HOST_BINARY}"
echo "live bridge: ${LIVE_LIBRARY}"
