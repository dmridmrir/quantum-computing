SET(BL 8)

get_filename_component(UPIM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
get_filename_component(PKG_ROOT "${UPIM_ROOT}/.." ABSOLUTE)

set(CMAKE_C_COMPILER "${PKG_ROOT}/qec_pim_package/sdk/bin/dpu-upmem-dpurte-clang")
set(CMAKE_C_FLAGS "-w -I${UPIM_ROOT}/benchmark/BPMS/support -O2 -S -g0 -fno-stack-size-section -DNR_TASKLETS=${NR_TASKLETS} -DBL=${BL}")

file(GLOB_RECURSE SRCS *.c)

add_executable(BPMS_device ${SRCS})
