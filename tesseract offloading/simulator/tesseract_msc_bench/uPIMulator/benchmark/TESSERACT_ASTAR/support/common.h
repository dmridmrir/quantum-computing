#ifndef TESSERACT_ASTAR_COMMON_H
#define TESSERACT_ASTAR_COMMON_H

#include <stdint.h>

/* Independent search groups per DPU. Each group owns TASKLETS_PER_GROUP
 * tasklets and decodes one syndrome; groups never share search state. */
#ifndef TESSERACT_ASTAR_GROUPS
#define TESSERACT_ASTAR_GROUPS 1
#endif
/* The argument block always carries two slots so the host ABI does not
 * depend on how the kernel was compiled. */
#define TESSERACT_ASTAR_MAX_GROUPS 2

/* Host-to-DPU control block for the DPU-resident Tesseract A* search. The
 * static model regions use the same layout as the TESSERACT candidate kernel
 * so the host can reuse its packer; the search state lives in MRAM after the
 * static image. All offsets are relative to DPU_MRAM_HEAP_POINTER. */
typedef struct {
    uint32_t num_detectors;
    uint32_t num_errors;
    uint32_t d2e_offsets_offset;
    uint32_t d2e_error_ids_offset;
    uint32_t e2d_offsets_offset;
    uint32_t e2d_detector_ids_offset;
    uint32_t cost_lut_offsets_offset;
    uint32_t cost_lut_values_offset;
    uint32_t error_records_offset;    /* num_errors * astar_error_record_t */
    uint32_t eneighbor_offsets_offset; /* num_errors+1 u32: detectors of errors sharing a detector, minus own */
    uint32_t eneighbor_ids_offset;    /* packed u16 detector IDs */
    uint32_t detector_penalty_q;
    uint32_t det_beam;                /* 0xFFFFFFFF = no beam */
    uint32_t pqlimit;                 /* pushes including root; 0xFFFFFFFF = unlimited */
    uint32_t no_revisit_dets;         /* exact residual-state suppression */
    uint32_t max_expansions;          /* 0 = unlimited */
    uint32_t num_syndromes;           /* 1..TESSERACT_ASTAR_MAX_GROUPS; groups beyond the kernel's count are ignored */
    uint32_t heap_capacity;           /* nodes per group heap */
    uint32_t arena_capacity;          /* links per group arena */
    uint32_t syndrome_words_offset[TESSERACT_ASTAR_MAX_GROUPS]; /* (num_detectors+31)/32 u32, 8-byte padded */
    uint32_t heap_offset[TESSERACT_ASTAR_MAX_GROUPS];           /* heap_capacity * 16 bytes */
    uint32_t arena_offset[TESSERACT_ASTAR_MAX_GROUPS];          /* arena_capacity * 8 bytes */
    uint32_t output_offset[TESSERACT_ASTAR_MAX_GROUPS];         /* tesseract_astar_output_t */
    uint32_t visited_offset[TESSERACT_ASTAR_MAX_GROUPS];        /* exact visited-state table */
    uint32_t visited_capacity;        /* power of two; zero disables no-revisit */
    uint32_t visited_generation;      /* avoids clearing MRAM between launches */
    uint32_t order_positions_offset;  /* packed u16 position[order][detector]; 0xFFFFFFFF = identity */
    uint32_t order_index[TESSERACT_ASTAR_MAX_GROUPS];           /* detector order used by each group */
    enum astar_kernels {
        astar_kernel1 = 0,
        astar_nr_kernels = 1,
    } kernel;
} astar_arguments_t;

#define TESSERACT_ASTAR_STATUS_SOLVED 0u
#define TESSERACT_ASTAR_STATUS_LOW_CONFIDENCE 1u
#define TESSERACT_ASTAR_STATUS_FALLBACK 2u

#define TESSERACT_ASTAR_MAX_DEPTH 256u

typedef struct {
    uint32_t status;
    uint32_t depth;
    uint32_t expansions;
    uint32_t candidates;
    uint32_t heap_peak;
    uint32_t arena_used;
    uint32_t pushed;
    uint32_t reserved;
    uint32_t heap_size;               /* live queue entries (for CPU resume after fallback) */
    uint32_t min_num_dets;
    uint32_t max_num_dets;
    uint32_t pad;
    uint16_t errors[TESSERACT_ASTAR_MAX_DEPTH];
} tesseract_astar_output_t;             /* 560 bytes */

/* Per-error record so one aligned MRAM read serves the inner scan, including
 * the error's detector list (degree <= TESSERACT_ASTAR_MAX_ERROR_DEGREE). */
#define TESSERACT_ASTAR_MAX_ERROR_DEGREE 11u
typedef struct {
    uint32_t lut_begin;
    uint32_t lower_bound;             /* cost_lut[e][degree] */
    uint16_t degree;
    uint16_t detectors[TESSERACT_ASTAR_MAX_ERROR_DEGREE];
} astar_error_record_t;              /* 32 bytes */

/* Kernel-side limits. */
#define TESSERACT_ASTAR_MAX_DETECTORS 512u
#define TESSERACT_ASTAR_MAX_ERRORS 10240u
#define TESSERACT_ASTAR_MAX_SIBLINGS 255u   /* u8 sibling positions; larger lists fall back */
#define TESSERACT_ASTAR_INF_COST UINT32_MAX
#define TESSERACT_ASTAR_NO_POSITION 0xFFu
#define TESSERACT_ASTAR_VISITED_ENTRY_BYTES 80u

#ifndef BL
#define BLOCK_SIZE_LOG2 8
#define BLOCK_SIZE (1u << BLOCK_SIZE_LOG2)
#define BL BLOCK_SIZE_LOG2
#else
#define BLOCK_SIZE_LOG2 BL
#define BLOCK_SIZE (1u << BLOCK_SIZE_LOG2)
#endif

#endif
