#ifndef _BFUF_COMMON_H_
#define _BFUF_COMMON_H_

#include <stdint.h>

typedef struct {
    uint32_t n_codewords;   /* codewords per DPU launch */
    uint32_t transfer_size;
    enum kernels {
        kernel1 = 0,
        nr_kernels = 1,
    } kernel;
} dpu_arguments_t;

/* [[144,12,12]] BB-QLDPC */
#define BB_NC      72
#define BB_NV     144
#define BB_CHK_DEG  6
#define BB_VAR_DEG  3

/* Node encoding in BF-UF: 0..71 = checks, 72..215 = variables */
#define BFUF_TOTAL  (BB_NC + BB_NV)   /* 216 */
#define BFUF_VAR_NODE(v)  ((uint8_t)((v) + BB_NC))
#define BFUF_GET_VAR(n)   ((uint8_t)((n) - BB_NC))

/*
 * GF(2) Gaussian elimination row format (DPU is 32-bit).
 * ceil((BB_NV + 1) / 32) = ceil(145/32) = 5 words per row.
 * 5 words × 4 B × BB_NC rows = 1440 B per tasklet — fits WRAM.
 */
#define GF2_WORDS 5

/* MRAM layout per tasklet:
 *   Input  slot: BB_NC bytes (syndrome), padded to 8-byte alignment → 72 B (already divisible by 8)
 *   Output slot: BB_NV bytes (correction), padded to 8-byte alignment → 144 B (already)
 */
#define SYN_SLOT_SIZE   72    /* bytes, 8-byte aligned (72 % 8 == 0) */
#define CORR_SLOT_SIZE 144    /* bytes, 8-byte aligned                */

#ifndef BL
#define BLOCK_SIZE_LOG2 8
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#define BL BLOCK_SIZE_LOG2
#else
#define BLOCK_SIZE_LOG2 BL
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#endif

#endif /* _BFUF_COMMON_H_ */
