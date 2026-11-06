set(CMAKE_C_COMPILER "/mnt/c/Users/wjdrm/bp_bench_pkg/qec_pim_package/sdk/bin/dpu-upmem-dpurte-clang")
set(CMAKE_C_FLAGS "-O3 -S -DNR_TASKLETS=${NR_TASKLETS} -DNR_THREADS=24")

include_directories("/home/ubuntu/bp_bench_pkg/uPIMulator/sdk/misc")
include_directories("/home/ubuntu/bp_bench_pkg/uPIMulator/sdk/stdlib")
include_directories("/home/ubuntu/bp_bench_pkg/uPIMulator/sdk/syslib")

file(GLOB_RECURSE SRCS *.c)

add_library(misc ${SRCS})
