/* Candidate-batch heuristic evaluator for MSC grafted-code Tesseract.
 *
 * The CPU retains A* search, priority-queue management, pruning, and
 * candidate generation. Each tasklet independently evaluates candidate
 * residual syndromes using detector/error CSR and a fixed-point cost LUT.
 */

#include <stdint.h>
#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <mutex.h>

#include "../support/common.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
BARRIER_INIT(start_barrier, NR_TASKLETS);
MUTEX_INIT(candidate_queue_mutex);

#define TESSERACT_STREAM_CACHE_BYTES 64u
#define TESSERACT_STREAM_CACHE_WORDS (TESSERACT_STREAM_CACHE_BYTES / 4u)

typedef struct {
    uint32_t words[TESSERACT_STREAM_CACHE_WORDS] __attribute__((aligned(8)));
    uint32_t tag;
} mram_u32_cache_t;

typedef struct {
    mram_u32_cache_t d2e_error_ids;
    uint32_t residual_words[TESSERACT_MAX_DETECTORS / 32u]
        __attribute__((aligned(8)));
    /* Sparse record image: [u32 count, u32 pad, packed u16 IDs..., pad]. */
    uint32_t blocked_count;
    uint32_t blocked_pad;
    uint16_t blocked_error_ids[TESSERACT_MAX_SPARSE_BLOCKED + 1u];
    uint32_t blocked_bloom[TESSERACT_BLOCKED_BLOOM_WORDS];
} evaluation_cache_t;

static uint32_t next_candidate;

static inline uint32_t mram_u32(uint32_t heap_base, uint32_t region_offset,
                                uint32_t index) {
    uint32_t pair[2] __attribute__((aligned(8)));
    uint32_t pair_index = index & ~1u;
    mram_read((__mram_ptr const void *)(heap_base + region_offset + pair_index * 4u),
              &pair, sizeof(pair));
    return pair[index & 1u];
}

static inline uint32_t cached_mram_u32(uint32_t heap_base,
                                       uint32_t region_offset,
                                       uint32_t index,
                                       mram_u32_cache_t *cache) {
    uint32_t address = heap_base + region_offset + index * 4u;
    uint32_t block_address = address & ~(TESSERACT_STREAM_CACHE_BYTES - 1u);
    if (cache->tag != block_address) {
        mram_read((__mram_ptr const void *)block_address,
                  cache->words,
                  TESSERACT_STREAM_CACHE_BYTES);
        cache->tag = block_address;
    }
    return cache->words[(address - block_address) >> 2];
}

static inline uint32_t residual_has(const evaluation_cache_t *cache,
                                    uint32_t detector_id) {
    return (cache->residual_words[detector_id >> 5] >>
            (detector_id & 31u)) & 1u;
}

static inline void load_candidate_residual(uint32_t heap_base,
                                           uint32_t candidate_id,
                                           evaluation_cache_t *cache) {
    uint32_t words_per_candidate =
        (DPU_INPUT_ARGUMENTS.num_detectors + 31u) >> 5;
    uint32_t stride_words = (words_per_candidate + 1u) & ~1u;
    mram_read(
        (__mram_ptr const void *)(heap_base +
            DPU_INPUT_ARGUMENTS.candidate_residual_words_offset +
            candidate_id * stride_words * 4u),
        cache->residual_words,
        stride_words * 4u);
}

static inline void load_candidate_blocked(uint32_t heap_base,
                                          uint32_t candidate_id,
                                          evaluation_cache_t *cache) {
    if (DPU_INPUT_ARGUMENTS.blocked_records_offset ==
        TESSERACT_NO_BLOCKED_WORDS) {
        return;
    }
    for (uint32_t word = 0; word < TESSERACT_BLOCKED_BLOOM_WORDS; ++word) {
        cache->blocked_bloom[word] = 0u;
    }

    uint32_t record_word_offset = mram_u32(
        heap_base,
        DPU_INPUT_ARGUMENTS.blocked_record_offsets_offset,
        candidate_id);
    uint32_t record_address = heap_base +
        DPU_INPUT_ARGUMENTS.blocked_records_offset + record_word_offset * 4u;
    uint32_t header[2] __attribute__((aligned(8)));
    mram_read((__mram_ptr const void *)record_address, header, sizeof(header));
    uint32_t count = header[0];
    uint32_t record_words = (2u + ((count + 1u) >> 1) + 1u) & ~1u;
    mram_read((__mram_ptr const void *)record_address,
              &cache->blocked_count,
              record_words * 4u);
    for (uint32_t index = 0; index < count; ++index) {
        uint32_t error_id = cache->blocked_error_ids[index];
        uint32_t bloom_bit = (error_id ^ (error_id >> 8)) & 255u;
        cache->blocked_bloom[bloom_bit >> 5] |=
            1u << (bloom_bit & 31u);
    }
}

static inline uint32_t error_is_blocked(uint32_t heap_base,
                                        uint32_t candidate_id,
                                        uint32_t error_id,
                                        const evaluation_cache_t *cache) {
    if (DPU_INPUT_ARGUMENTS.blocked_records_offset !=
        TESSERACT_NO_BLOCKED_WORDS) {
        uint32_t bloom_bit = (error_id ^ (error_id >> 8)) & 255u;
        if (((cache->blocked_bloom[bloom_bit >> 5] >>
              (bloom_bit & 31u)) & 1u) == 0u) {
            return 0u;
        }
        uint32_t lo = 0u;
        uint32_t hi = cache->blocked_count;
        while (lo < hi) {
            uint32_t mid = lo + ((hi - lo) >> 1);
            uint32_t value = cache->blocked_error_ids[mid];
            if (value < error_id) {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        return lo < cache->blocked_count &&
               cache->blocked_error_ids[lo] == error_id;
    }
    if (DPU_INPUT_ARGUMENTS.blocked_words_offset ==
        TESSERACT_NO_BLOCKED_WORDS) {
        return 0u;
    }

    uint32_t words_per_candidate = (DPU_INPUT_ARGUMENTS.num_errors + 31u) >> 5;
    uint32_t word_index = candidate_id * words_per_candidate + (error_id >> 5);
    uint32_t word = mram_u32(heap_base,
                             DPU_INPUT_ARGUMENTS.blocked_words_offset,
                             word_index);
    return (word >> (error_id & 31u)) & 1u;
}

static uint32_t evaluate_candidate(uint32_t heap_base, uint32_t candidate_id,
                                   evaluation_cache_t *cache) {
    load_candidate_residual(heap_base, candidate_id, cache);
    load_candidate_blocked(heap_base, candidate_id, cache);
    uint32_t begin = mram_u32(heap_base,
                              DPU_INPUT_ARGUMENTS.candidate_offsets_offset,
                              candidate_id);
    uint32_t end = mram_u32(heap_base,
                            DPU_INPUT_ARGUMENTS.candidate_offsets_offset,
                            candidate_id + 1u);

    uint32_t total = 0u;
    for (uint32_t pos = begin; pos < end; ++pos) {
        uint32_t detector_id = mram_u32(
            heap_base, DPU_INPUT_ARGUMENTS.candidate_detector_ids_offset, pos);
        uint32_t error_begin = mram_u32(
            heap_base, DPU_INPUT_ARGUMENTS.d2e_offsets_offset, detector_id);
        uint32_t error_end = mram_u32(
            heap_base, DPU_INPUT_ARGUMENTS.d2e_offsets_offset, detector_id + 1u);
        uint32_t minimum = TESSERACT_INF_COST;

        for (uint32_t error_pos = error_begin; error_pos < error_end;
             ++error_pos) {
            uint32_t error_id = cached_mram_u32(
                heap_base,
                DPU_INPUT_ARGUMENTS.d2e_error_ids_offset,
                error_pos,
                &cache->d2e_error_ids);
            if (error_is_blocked(heap_base, candidate_id, error_id, cache)) {
                continue;
            }

            uint32_t detector_begin = mram_u32(
                heap_base, DPU_INPUT_ARGUMENTS.e2d_offsets_offset, error_id);
            uint32_t detector_end = mram_u32(
                heap_base, DPU_INPUT_ARGUMENTS.e2d_offsets_offset, error_id + 1u);
            uint32_t lut_begin = mram_u32(
                heap_base, DPU_INPUT_ARGUMENTS.cost_lut_offsets_offset,
                error_id);
            uint32_t lower_bound = mram_u32(
                heap_base, DPU_INPUT_ARGUMENTS.cost_lut_values_offset,
                lut_begin + detector_end - detector_begin);
            if (minimum != TESSERACT_INF_COST && lower_bound >= minimum) {
                break;
            }
            uint32_t overlap = 0u;
            for (uint32_t detector_pos = detector_begin;
                 detector_pos < detector_end; ++detector_pos) {
                uint32_t incident_detector = mram_u32(
                    heap_base, DPU_INPUT_ARGUMENTS.e2d_detector_ids_offset,
                    detector_pos);
                overlap += residual_has(cache, incident_detector);
            }
            if (overlap == 0u) {
                continue;
            }

            uint32_t cost = mram_u32(
                heap_base, DPU_INPUT_ARGUMENTS.cost_lut_values_offset,
                lut_begin + overlap);
            if (cost < minimum) {
                minimum = cost;
            }
        }

        if (minimum == TESSERACT_INF_COST ||
            minimum >= TESSERACT_INF_COST -
                           DPU_INPUT_ARGUMENTS.detector_penalty_q) {
            total = TESSERACT_INF_COST;
            break;
        }
        minimum += DPU_INPUT_ARGUMENTS.detector_penalty_q;
        if (minimum >= TESSERACT_INF_COST - total) {
            total = TESSERACT_INF_COST;
            break;
        }
        total += minimum;
    }

    return total;
}

extern int main_kernel1(void);
int main(void) {
    if (DPU_INPUT_ARGUMENTS.kernel != kernel1) {
        return 1;
    }
    return main_kernel1();
}

int main_kernel1(void) {
    uint32_t tasklet_id = me();
    uint32_t heap_base = (uint32_t)DPU_MRAM_HEAP_POINTER;
    evaluation_cache_t cache;
    cache.d2e_error_ids.tag = UINT32_MAX;

    if (DPU_INPUT_ARGUMENTS.num_detectors > TESSERACT_MAX_DETECTORS) {
        return 1;
    }

    if (tasklet_id == 0u) {
        next_candidate = 0u;
    }
    barrier_wait(&start_barrier);
    while (1) {
        mutex_lock(candidate_queue_mutex);
        uint32_t candidate_id = next_candidate++;
        mutex_unlock(candidate_queue_mutex);
        if (candidate_id >= DPU_INPUT_ARGUMENTS.num_candidates) {
            break;
        }

        uint64_t output __attribute__((aligned(8))) =
            evaluate_candidate(heap_base, candidate_id, &cache);
        mram_write(&output,
                   (__mram_ptr void *)(heap_base +
                       DPU_INPUT_ARGUMENTS.output_costs_offset +
                       candidate_id * sizeof(uint64_t)),
                   sizeof(output));
    }

    return 0;
}
