#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
    uint32_t L;
    uint32_t num_vertices;
    uint32_t num_edges;
    uint32_t num_events;
    uint32_t size;
    uint32_t transfer_size;
    enum kernels {
        kernel1 = 0,
        nr_kernels = 1,
    } kernel;
} dpu_arguments_t;

/* [[144,12,12]] bivariate bicycle code */
#define BB_NC 72
#define BB_NV 144
#define BB_NNZ 432
#define BB_CHK_DEG 6
#define BB_VAR_DEG 3

#define MAX_V_DPU BB_NV
#define MAX_C_DPU BB_NC
#define MAX_NNZ_DPU BB_NNZ

#define FP_ALPHA 160
#define FP_MAX 32000

#ifndef BL
#define BLOCK_SIZE_LOG2 8
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#define BL BLOCK_SIZE_LOG2
#else
#define BLOCK_SIZE_LOG2 BL
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#endif

#define PRINT 0
#define ENERGY 0

#endif
