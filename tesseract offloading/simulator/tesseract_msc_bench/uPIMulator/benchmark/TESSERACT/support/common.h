#ifndef TESSERACT_COMMON_H
#define TESSERACT_COMMON_H

#include <stdint.h>

/* Host-to-DPU control block. Every offset is relative to
 * DPU_MRAM_HEAP_POINTER and every MRAM region is 8-byte aligned. */
typedef struct {
    uint32_t num_detectors;
    uint32_t num_errors;
    uint32_t num_candidates;
    uint32_t num_d2e_ids;
    uint32_t num_e2d_ids;
    uint32_t d2e_offsets_offset;
    uint32_t d2e_error_ids_offset;
    uint32_t e2d_offsets_offset;
    uint32_t e2d_detector_ids_offset;
    uint32_t cost_lut_offsets_offset;
    uint32_t cost_lut_values_offset;
    uint32_t candidate_offsets_offset;
    uint32_t candidate_detector_ids_offset;
    uint32_t candidate_residual_words_offset;
    uint32_t blocked_words_offset;
    uint32_t blocked_record_offsets_offset;
    uint32_t blocked_records_offset;
    uint32_t output_costs_offset;
    uint32_t detector_penalty_q;
    uint32_t q_bits;
    enum kernels {
        kernel1 = 0,
        nr_kernels = 1,
    } kernel;
    uint32_t reserved;
} dpu_arguments_t;

/* Initial functional limit for the host-packed candidate syndrome bitsets. */
#define TESSERACT_MAX_DETECTORS 4096u
#define TESSERACT_MAX_SPARSE_BLOCKED 128u
#define TESSERACT_BLOCKED_BLOOM_WORDS 8u
#define TESSERACT_NO_BLOCKED_WORDS UINT32_MAX
#define TESSERACT_INF_COST UINT32_MAX

#ifndef BL
#define BLOCK_SIZE_LOG2 8
#define BLOCK_SIZE (1u << BLOCK_SIZE_LOG2)
#define BL BLOCK_SIZE_LOG2
#else
#define BLOCK_SIZE_LOG2 BL
#define BLOCK_SIZE (1u << BLOCK_SIZE_LOG2)
#endif

#define PRINT 0
#define ENERGY 0

#endif
