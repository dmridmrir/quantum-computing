#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
    uint32_t d, T, num_edges, max_iter;
    int32_t  alpha_fp;
    uint32_t kernel, num_nodes;
    uint32_t n_right, n_down, n_future;
    uint32_t pad;
} dpu_arguments_t;

typedef struct {
    uint64_t cycle_count;
    uint32_t iterations_run;
    uint32_t total_msgs_processed;
} bp3d_result_t;

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

#define ALIGN8(x) (((x) + 7) & ~7u)
#define LLR_INF 1000000
#define MAX_NODES 3000
#define BLK 512
#define BLK_B 2048
#define ALPHA_NUM 768
#define ALPHA_SHIFT 10

#define OFF_V2C  0
#define OFF_C2V(ne)  (ALIGN8((ne)*4))
#define OFF_RES(ne)  (2*ALIGN8((ne)*4))
#define OFF_LLR(ne)  (2*ALIGN8((ne)*4) + ALIGN8(sizeof(bp3d_result_t)))

#endif
