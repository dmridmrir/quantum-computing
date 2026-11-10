set(BL 8)

if(NOT DEFINED ENV{UPMEM_HOME})
    message(FATAL_ERROR "UPMEM_HOME is not set; source the bundled upmem_env.sh first")
endif()

set(CMAKE_C_COMPILER "$ENV{UPMEM_HOME}/bin/dpu-upmem-dpurte-clang")

file(GLOB_RECURSE SRCS *.c)
add_library(TESSERACT_ASTAR_device OBJECT ${SRCS})

target_include_directories(TESSERACT_ASTAR_device PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../support"
)
target_compile_options(TESSERACT_ASTAR_device PRIVATE
    -w
    -O2
    -S
    -g0
    -fno-stack-size-section
)
target_compile_definitions(TESSERACT_ASTAR_device PRIVATE
    NR_TASKLETS=${NR_TASKLETS}
    BL=${BL}
    TESSERACT_ASTAR_GROUPS=${TESSERACT_ASTAR_GROUPS}
)
