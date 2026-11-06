#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

/* Shared args between host and DPU */
typedef struct {
    uint32_t d;
    uint32_t T;
    uint32_t total_nodes;
    uint32_t num_events;
    uint32_t size;            /* total MRAM bytes used */
    uint32_t transfer_size;   /* events data size */
    enum kernels {
        kernel1 = 0,
        nr_kernels = 1,
    } kernel;
} dpu_arguments_t;

typedef struct {
    uint64_t cycle_count;
    uint32_t num_clusters;
    uint32_t num_merges;
    uint32_t easy_case;
    uint32_t pad;
} dpu_results_t;

#define MAX_NODES  5120
#define MAX_EVENTS 256

/* Block size for MRAM transfers */
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
