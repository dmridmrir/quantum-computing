#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
    uint32_t L;            /* toric code size */
    uint32_t num_vertices; /* L*L */
    uint32_t num_edges;    /* 2*L*L */
    uint32_t num_events;   /* syndrome event count (per tasklet) */
    uint32_t size;
    uint32_t transfer_size;
    enum kernels {
        kernel1 = 0,
        nr_kernels = 1,
    } kernel;
} dpu_arguments_t;

/* Tight limits for WRAM — sized for max L=10, p=0.005 */
#define MAX_L_DPU 10
#define MAX_V_DPU (MAX_L_DPU * MAX_L_DPU)  /* 100 */
#define MAX_E_DPU (2 * MAX_V_DPU)           /* 200 */
#define MAX_FUSE  128
#define MAX_BORDER 128
#define MAX_PEEL  128

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
