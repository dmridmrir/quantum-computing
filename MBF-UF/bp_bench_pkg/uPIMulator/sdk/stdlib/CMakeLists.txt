get_filename_component(SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
get_filename_component(UPIM_ROOT "${SDK_ROOT}/.." ABSOLUTE)
get_filename_component(PKG_ROOT "${UPIM_ROOT}/.." ABSOLUTE)
set(CMAKE_C_COMPILER "${PKG_ROOT}/qec_pim_package/sdk/bin/dpu-upmem-dpurte-clang")
set(CMAKE_C_FLAGS "-O3 -S -DNR_TASKLETS=${NR_TASKLETS} -DNR_THREADS=24")

include_directories("${SDK_ROOT}/misc")
include_directories("${SDK_ROOT}/stdlib")
include_directories("${SDK_ROOT}/syslib")

file(GLOB_RECURSE SRCS *.c)

add_library(stdlib ${SRCS})
