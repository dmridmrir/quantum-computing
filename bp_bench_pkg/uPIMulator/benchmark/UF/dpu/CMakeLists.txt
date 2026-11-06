SET(BL 8)

set(CMAKE_C_COMPILER "/root/document/upmem_bp_bench/qec_pim_package/sdk/bin/dpu-upmem-dpurte-clang")
set(CMAKE_C_FLAGS "-w -I/root/uPIMulator/golang/uPIMulator/benchmark/UF/support -O2 -S -DNR_TASKLETS=${NR_TASKLETS} -DBL=${BL} -DNODES_PER_TASKLET=${NODES_PER_TASKLET}")

file(GLOB_RECURSE SRCS *.c)

add_executable(UF_device ${SRCS})
