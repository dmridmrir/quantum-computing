if(NOT DEFINED ENV{UPMEM_HOME})
    message(FATAL_ERROR "UPMEM_HOME is not set; source the bundled upmem_env.sh first")
endif()
set(CMAKE_C_COMPILER "$ENV{UPMEM_HOME}/bin/dpu-upmem-dpurte-clang")
set(CMAKE_C_FLAGS "-O3 -S -DNR_TASKLETS=${NR_TASKLETS} -DNR_THREADS=24")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../misc")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../stdlib")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../syslib")

file(GLOB_RECURSE SRCS *.c)

add_library(misc ${SRCS})
