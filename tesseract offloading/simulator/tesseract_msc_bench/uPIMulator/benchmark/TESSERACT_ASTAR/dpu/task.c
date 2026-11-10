/* DPU-resident Tesseract A* search.
 *
 * The DPU's tasklets are split into TESSERACT_ASTAR_GROUPS independent groups;
 * each group decodes one syndrome with its own priority queue and error-chain
 * arena in MRAM. Within a group, the leader tasklet performs one A* expansion
 * per round: pop, rebuild the node state, list the sibling candidates. All
 * group tasklets then evaluate the node's per-detector heuristic terms, then
 * the sibling heuristics incrementally (Tesseract's detector_cost_cache
 * scheme), and the leader pushes the survivors. The host only sends the
 * syndromes and reads back the selected error chains.
 */

#include <stdint.h>
#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <mutex.h>

#include "../support/common.h"

#define TASKLETS_PER_GROUP (NR_TASKLETS / TESSERACT_ASTAR_GROUPS)
#if TESSERACT_ASTAR_GROUPS < 1 || TESSERACT_ASTAR_GROUPS > 2
#error "this kernel instantiates barriers and mutexes for one or two groups"
#endif

__host astar_arguments_t DPU_INPUT_ARGUMENTS;
BARRIER_INIT(group_barrier_0, TASKLETS_PER_GROUP);
BARRIER_INIT(group_barrier_1, TASKLETS_PER_GROUP);
MUTEX_INIT(queue_mutex_0);
MUTEX_INIT(queue_mutex_1);
MUTEX_INIT(total_mutex_0);
MUTEX_INIT(total_mutex_1);

#define STREAM_CACHE_BYTES 64u
#define STREAM_CACHE_WORDS (STREAM_CACHE_BYTES / 4u)
#define RESIDUAL_WORDS (TESSERACT_ASTAR_MAX_DETECTORS / 32u)
#define BLOCKED_WORDS (TESSERACT_ASTAR_MAX_ERRORS / 32u)
#define NO_PREFIX 0xFFFFFFFFu

typedef struct {
    uint32_t words[STREAM_CACHE_WORDS] __attribute__((aligned(8)));
    uint32_t tag;
} mram_u32_cache_t;

typedef struct {
    uint32_t cost;
    uint32_t path;
    uint32_t chain;
    uint16_t num_dets;
    uint16_t depth;
} __attribute__((aligned(8))) heap_node_t;

typedef struct {
    uint32_t parent;
    uint16_t error;
    uint16_t min_detector;
} __attribute__((aligned(8))) arena_node_t;

typedef struct {
    uint32_t generation;
    uint32_t hash;
    uint16_t num_dets;
    uint16_t reserved;
    uint32_t residual[RESIDUAL_WORDS];
    uint32_t pad;
} __attribute__((aligned(8))) visited_entry_t;

_Static_assert(sizeof(visited_entry_t) == TESSERACT_ASTAR_VISITED_ENTRY_BYTES,
               "visited entry ABI size mismatch");

#define PHASE_EVALUATE 0u
#define PHASE_DONE 1u

/* All per-group search state. Written by the group's leader between
 * barriers; read by every tasklet of the group. */
typedef struct {
    uint32_t group;
    uint32_t phase;
    uint32_t residual[RESIDUAL_WORDS] __attribute__((aligned(8)));
    uint32_t blocked[BLOCKED_WORDS] __attribute__((aligned(8)));
    uint8_t sibling_position[TESSERACT_ASTAR_MAX_ERRORS];
    uint32_t sibling_begin;
    uint32_t sibling_count;
    uint32_t sibling_min_detector;
    uint32_t sibling_max_num_dets;
    uint32_t sibling_cost[TESSERACT_ASTAR_MAX_SIBLINGS];
    uint16_t sibling_num_dets[TESSERACT_ASTAR_MAX_SIBLINGS];
    uint8_t sibling_evaluated[TESSERACT_ASTAR_MAX_SIBLINGS];
    uint32_t next_sibling;
    uint32_t parent_cost[TESSERACT_ASTAR_MAX_DETECTORS];
    uint16_t parent_detectors[TESSERACT_ASTAR_MAX_DETECTORS];
    uint32_t parent_detector_count;
    uint32_t next_parent_detector;
    uint32_t parent_total;
    uint32_t parent_has_inf;
    /* leader state */
    uint32_t heap_offset;
    uint32_t arena_offset;
    uint32_t heap_size;
    uint32_t heap_peak;
    uint32_t arena_size;
    uint32_t total_expansions;
    uint32_t total_candidates;
    uint32_t total_pushed;
    uint32_t visited_offset;
    uint32_t visited_count;
    uint32_t initial_residual[RESIDUAL_WORDS] __attribute__((aligned(8)));
    uint16_t detector_position[TESSERACT_ASTAR_MAX_DETECTORS] __attribute__((aligned(8)));
    uint32_t use_positions;
    uint32_t min_num_dets;
    uint32_t max_num_dets;
    heap_node_t current_node;
    tesseract_astar_output_t output __attribute__((aligned(8)));
    uint16_t debug_info[8];
} search_state_t;

static search_state_t groups[TESSERACT_ASTAR_GROUPS];
static uint32_t residual_words;
static uint32_t blocked_words;

static inline uint32_t heap_base(void) {
    return (uint32_t)DPU_MRAM_HEAP_POINTER;
}

static inline void group_barrier(uint32_t group) {
    barrier_wait(group == 0u ? &group_barrier_0 : &group_barrier_1);
}

static inline mutex_id_t group_queue_mutex(uint32_t group) {
    return group == 0u ? queue_mutex_0 : queue_mutex_1;
}

static inline mutex_id_t group_total_mutex(uint32_t group) {
    return group == 0u ? total_mutex_0 : total_mutex_1;
}

static inline uint32_t mram_u32(uint32_t region_offset, uint32_t index) {
    uint32_t pair[2] __attribute__((aligned(8)));
    uint32_t pair_index = index & ~1u;
    mram_read((__mram_ptr const void *)(heap_base() + region_offset + pair_index * 4u),
              pair, sizeof(pair));
    return pair[index & 1u];
}

static inline uint32_t cached_mram_u32(uint32_t region_offset, uint32_t index,
                                       mram_u32_cache_t *cache) {
    uint32_t address = heap_base() + region_offset + index * 4u;
    uint32_t block_address = address & ~(STREAM_CACHE_BYTES - 1u);
    if (cache->tag != block_address) {
        mram_read((__mram_ptr const void *)block_address, cache->words, STREAM_CACHE_BYTES);
        cache->tag = block_address;
    }
    return cache->words[(address - block_address) >> 2];
}

static inline uint32_t cached_mram_u16(uint32_t region_offset, uint32_t index,
                                       mram_u32_cache_t *cache) {
    uint32_t word = cached_mram_u32(region_offset, index >> 1, cache);
    return (index & 1u) ? (word >> 16) : (word & 0xFFFFu);
}

static inline void read_error_record(uint32_t error_id, astar_error_record_t *record) {
    mram_read((__mram_ptr const void *)(heap_base() + DPU_INPUT_ARGUMENTS.error_records_offset +
                                        error_id * sizeof(astar_error_record_t)),
              record, sizeof(*record));
}

static inline uint32_t popcount32(uint32_t value) {
    value = value - ((value >> 1) & 0x55555555u);
    value = (value & 0x33333333u) + ((value >> 2) & 0x33333333u);
    value = (value + (value >> 4)) & 0x0F0F0F0Fu;
    return (value * 0x01010101u) >> 24;
}

static inline uint32_t ctz32(uint32_t value) {
    uint32_t count = 0;
    while ((value & 1u) == 0u) {
        value >>= 1;
        ++count;
    }
    return count;
}

static inline uint32_t residual_has(const uint32_t *residual, uint32_t detector) {
    return (residual[detector >> 5] >> (detector & 31u)) & 1u;
}

/* ---------------------------------------------------------------- heap */

static inline void heap_read(const search_state_t *st, uint32_t index, heap_node_t *node) {
    mram_read((__mram_ptr const void *)(heap_base() + st->heap_offset + index * sizeof(heap_node_t)),
              node, sizeof(heap_node_t));
}

static inline void heap_write(const search_state_t *st, uint32_t index, const heap_node_t *node) {
    mram_write((void *)node,
               (__mram_ptr void *)(heap_base() + st->heap_offset + index * sizeof(heap_node_t)),
               sizeof(heap_node_t));
}

/* Tesseract pops the lowest cost first; ties prefer the node with more
 * remaining detectors. */
static inline uint32_t heap_less(const heap_node_t *a, const heap_node_t *b) {
    return a->cost < b->cost || (a->cost == b->cost && a->num_dets > b->num_dets);
}

static uint32_t heap_push(search_state_t *st, const heap_node_t *node) {
    if (st->heap_size >= DPU_INPUT_ARGUMENTS.heap_capacity) {
        return 0u;
    }
    uint32_t index = st->heap_size++;
    if (st->heap_size > st->heap_peak) {
        st->heap_peak = st->heap_size;
    }
    heap_node_t parent;
    while (index > 0u) {
        uint32_t parent_index = (index - 1u) >> 1;
        heap_read(st, parent_index, &parent);
        if (!heap_less(node, &parent)) {
            break;
        }
        heap_write(st, index, &parent);
        index = parent_index;
    }
    heap_write(st, index, node);
    return 1u;
}

static void heap_pop(search_state_t *st, heap_node_t *out) {
    heap_read(st, 0u, out);
    --st->heap_size;
    if (st->heap_size == 0u) {
        return;
    }
    heap_node_t last;
    heap_read(st, st->heap_size, &last);
    uint32_t index = 0u;
    heap_node_t child;
    heap_node_t other;
    while (1) {
        uint32_t left = index * 2u + 1u;
        if (left >= st->heap_size) {
            break;
        }
        uint32_t best = left;
        heap_read(st, left, &child);
        if (left + 1u < st->heap_size) {
            heap_read(st, left + 1u, &other);
            if (heap_less(&other, &child)) {
                best = left + 1u;
                child = other;
            }
        }
        if (!heap_less(&child, &last)) {
            break;
        }
        heap_write(st, index, &child);
        index = best;
    }
    heap_write(st, index, &last);
}

/* --------------------------------------------------------------- arena */

static inline void arena_read(const search_state_t *st, uint32_t index, arena_node_t *node) {
    mram_read((__mram_ptr const void *)(heap_base() + st->arena_offset + index * sizeof(arena_node_t)),
              node, sizeof(arena_node_t));
}

static inline void arena_write(const search_state_t *st, uint32_t index, const arena_node_t *node) {
    mram_write((void *)node,
               (__mram_ptr void *)(heap_base() + st->arena_offset + index * sizeof(arena_node_t)),
               sizeof(arena_node_t));
}

/* ------------------------------------------------ exact no-revisit table */

static uint32_t residual_hash(const uint32_t *residual, uint32_t num_dets) {
    uint32_t hash = 2166136261u ^ num_dets;
    for (uint32_t word = 0; word < residual_words; ++word) {
        hash ^= residual[word];
        hash *= 16777619u;
        hash ^= hash >> 13;
    }
    return hash == 0u ? 1u : hash;
}

/* Return 1 when present, 0 when absent (and optionally inserted), 2 when the
 * exact table cannot accept another entry. Hash matches are verified against
 * all residual words, so collisions never prune a distinct search state. */
static uint32_t visited_probe(search_state_t *st, const uint32_t *residual,
                              uint32_t num_dets, uint32_t insert) {
    if (DPU_INPUT_ARGUMENTS.no_revisit_dets == 0u) {
        return 0u;
    }
    uint32_t capacity = DPU_INPUT_ARGUMENTS.visited_capacity;
    if (capacity == 0u) {
        return insert ? 2u : 0u;
    }
    if (insert && st->visited_count * 10u >= capacity * 7u) {
        return 2u;
    }
    uint32_t hash = residual_hash(residual, num_dets);
    uint32_t slot = hash & (capacity - 1u);
    visited_entry_t entry __attribute__((aligned(8)));
    for (uint32_t probe = 0; probe < capacity; ++probe) {
        uint32_t index = (slot + probe) & (capacity - 1u);
        mram_read((__mram_ptr const void *)(heap_base() + st->visited_offset +
                                            index * sizeof(visited_entry_t)),
                  &entry, sizeof(entry));
        if (entry.generation != DPU_INPUT_ARGUMENTS.visited_generation) {
            if (!insert) {
                return 0u;
            }
            entry.generation = DPU_INPUT_ARGUMENTS.visited_generation;
            entry.hash = hash;
            entry.num_dets = (uint16_t)num_dets;
            entry.reserved = 0u;
            entry.pad = 0u;
            for (uint32_t word = 0; word < RESIDUAL_WORDS; ++word) {
                entry.residual[word] = word < residual_words ? residual[word] : 0u;
            }
            mram_write(&entry,
                       (__mram_ptr void *)(heap_base() + st->visited_offset +
                                            index * sizeof(visited_entry_t)),
                       sizeof(entry));
            ++st->visited_count;
            return 0u;
        }
        if (entry.hash != hash || entry.num_dets != num_dets) {
            continue;
        }
        uint32_t equal = 1u;
        for (uint32_t word = 0; word < residual_words; ++word) {
            if (entry.residual[word] != residual[word]) {
                equal = 0u;
                break;
            }
        }
        if (equal) {
            return 1u;
        }
    }
    return insert ? 2u : 0u;
}

/* ----------------------------------------------------------- heuristic */

/* Cheapest unblocked incident error cost of one detector plus the detector
 * penalty, or INF. `sibling_index` = NO_PREFIX evaluates under the expanded
 * node's blocked set only; otherwise errors at sibling positions <= index are
 * blocked too. */
static uint32_t evaluate_detector(const search_state_t *st, const uint32_t *residual,
                                  uint32_t detector_id, uint32_t sibling_index,
                                  mram_u32_cache_t *stream) {
    uint32_t error_begin = mram_u32(DPU_INPUT_ARGUMENTS.d2e_offsets_offset, detector_id);
    uint32_t error_end = mram_u32(DPU_INPUT_ARGUMENTS.d2e_offsets_offset, detector_id + 1u);
    uint32_t minimum = TESSERACT_ASTAR_INF_COST;
    for (uint32_t error_pos = error_begin; error_pos < error_end; ++error_pos) {
        uint32_t error_id = cached_mram_u32(DPU_INPUT_ARGUMENTS.d2e_error_ids_offset,
                                            error_pos, stream);
        if ((st->blocked[error_id >> 5] >> (error_id & 31u)) & 1u) {
            continue;
        }
        if (sibling_index != NO_PREFIX && st->sibling_position[error_id] <= sibling_index) {
            continue;
        }
        astar_error_record_t record __attribute__((aligned(8)));
        read_error_record(error_id, &record);
        if (minimum != TESSERACT_ASTAR_INF_COST && record.lower_bound >= minimum) {
            break;
        }
        uint32_t overlap = 0u;
        for (uint32_t k = 0; k < record.degree; ++k) {
            overlap += residual_has(residual, record.detectors[k]);
        }
        if (overlap == 0u) {
            continue;
        }
        uint32_t cost = mram_u32(DPU_INPUT_ARGUMENTS.cost_lut_values_offset,
                                 record.lut_begin + overlap);
        if (cost < minimum) {
            minimum = cost;
        }
    }
    if (minimum == TESSERACT_ASTAR_INF_COST ||
        minimum >= TESSERACT_ASTAR_INF_COST - DPU_INPUT_ARGUMENTS.detector_penalty_q) {
        return TESSERACT_ASTAR_INF_COST;
    }
    return minimum + DPU_INPUT_ARGUMENTS.detector_penalty_q;
}

/* Incremental heuristic of sibling j: the parent total minus the terms of
 * detectors flipped off or re-evaluated, plus their new values. Only detectors
 * in D(e_j) and in eneighbors(e_j) can change (Tesseract's detector_cost_cache
 * argument): errors incident to any other residual detector share no detector
 * with e_j, and every error blocked by the sibling prefix is incident to
 * min_detector and therefore a neighbor.
 *
 * noinline: dpu-upmem-dpurte-clang 12 -O2 miscompiles this body when it is
 * inlined into evaluate_siblings (the search diverges from the Go oracle). */
static __attribute__((noinline)) uint32_t sibling_heuristic(
    const search_state_t *st, const uint32_t *residual, const astar_error_record_t *record,
    uint32_t error_id, uint32_t j, mram_u32_cache_t *stream) {
    uint32_t total = st->parent_total;
    uint32_t infinite = 0u;
    for (uint32_t k = 0; k < record->degree && !infinite; ++k) {
        uint32_t detector = record->detectors[k];
        if (residual_has(st->residual, detector)) {
            total -= st->parent_cost[detector];
        } else {
            uint32_t cost = evaluate_detector(st, residual, detector, j, stream);
            if (cost == TESSERACT_ASTAR_INF_COST || cost >= TESSERACT_ASTAR_INF_COST - total) {
                infinite = 1u;
            } else {
                total += cost;
            }
        }
    }
    uint32_t neighbor_begin = mram_u32(DPU_INPUT_ARGUMENTS.eneighbor_offsets_offset, error_id);
    uint32_t neighbor_end = mram_u32(DPU_INPUT_ARGUMENTS.eneighbor_offsets_offset, error_id + 1u);
    for (uint32_t pos = neighbor_begin; pos < neighbor_end && !infinite; ++pos) {
        uint32_t detector = cached_mram_u16(DPU_INPUT_ARGUMENTS.eneighbor_ids_offset, pos, stream);
        if (!residual_has(st->residual, detector)) {
            continue;
        }
        total -= st->parent_cost[detector];
        uint32_t cost = evaluate_detector(st, residual, detector, j, stream);
        if (cost == TESSERACT_ASTAR_INF_COST || cost >= TESSERACT_ASTAR_INF_COST - total) {
            infinite = 1u;
        } else {
            total += cost;
        }
    }
    return infinite ? TESSERACT_ASTAR_INF_COST : total;
}

/* ------------------------------------------------------------- leader */

static void flip_error_detectors(uint32_t *residual, uint32_t error_id) {
    astar_error_record_t record __attribute__((aligned(8)));
    read_error_record(error_id, &record);
    for (uint32_t k = 0; k < record.degree; ++k) {
        uint32_t detector = record.detectors[k];
        residual[detector >> 5] ^= 1u << (detector & 31u);
    }
}

static inline uint32_t error_cost_q(uint32_t error_id) {
    uint32_t lut_begin = mram_u32(DPU_INPUT_ARGUMENTS.cost_lut_offsets_offset, error_id);
    return mram_u32(DPU_INPUT_ARGUMENTS.cost_lut_values_offset, lut_begin + 1u);
}

static void finish(search_state_t *st, uint32_t status) {
    st->output.status = status;
    st->output.expansions = st->total_expansions;
    st->output.candidates = st->total_candidates;
    st->output.heap_peak = st->heap_peak;
    st->output.arena_used = st->arena_size;
    st->output.pushed = st->total_pushed;
    st->output.heap_size = st->heap_size;
    st->output.min_num_dets = st->min_num_dets;
    st->output.max_num_dets = st->max_num_dets;
    st->output.pad = 0u;
#ifdef ASTAR_VERIFY_INCREMENTAL
    for (uint32_t i = 0; i < 8u; ++i) {
        st->output.errors[TESSERACT_ASTAR_MAX_DEPTH - 8u + i] = st->debug_info[i];
    }
#endif
    mram_write(&st->output,
               (__mram_ptr void *)(heap_base() + DPU_INPUT_ARGUMENTS.output_offset[st->group]),
               sizeof(st->output));
    st->phase = PHASE_DONE;
}

/* Rebuild the popped node's residual and blocked set from its error chain and
 * publish the sibling list. Returns 0 when the search ended. */
static uint32_t prepare_expansion(search_state_t *st, mram_u32_cache_t *stream) {
    /* Fallback checks happen before any queue entry is consumed so the host
     * can resume from exactly the exported queue. */
    if (DPU_INPUT_ARGUMENTS.max_expansions != 0u &&
        st->total_expansions >= DPU_INPUT_ARGUMENTS.max_expansions) {
        finish(st, TESSERACT_ASTAR_STATUS_FALLBACK);
        return 0u;
    }
pop_next_node:
    while (1) {
        if (st->heap_size == 0u) {
            finish(st, TESSERACT_ASTAR_STATUS_LOW_CONFIDENCE);
            return 0u;
        }
        heap_pop(st, &st->current_node);
        if (st->current_node.num_dets <= st->max_num_dets) {
            break;
        }
    }

    for (uint32_t word = 0; word < residual_words; ++word) {
        st->residual[word] = st->initial_residual[word];
    }
    for (uint32_t word = 0; word < blocked_words; ++word) {
        st->blocked[word] = 0u;
    }

    uint32_t walker = st->current_node.chain;
    uint32_t depth = 0u;
    arena_node_t link;
    while (walker != UINT32_MAX) {
        arena_read(st, walker, &link);
        uint32_t begin = mram_u32(DPU_INPUT_ARGUMENTS.d2e_offsets_offset, link.min_detector);
        uint32_t end = mram_u32(DPU_INPUT_ARGUMENTS.d2e_offsets_offset, link.min_detector + 1u);
        for (uint32_t pos = begin; pos < end; ++pos) {
            uint32_t other = cached_mram_u32(DPU_INPUT_ARGUMENTS.d2e_error_ids_offset, pos, stream);
            st->blocked[other >> 5] |= 1u << (other & 31u);
            if (other == link.error) {
                break;
            }
        }
        flip_error_detectors(st->residual, link.error);
        ++depth;
        walker = link.parent;
    }

    if (st->current_node.num_dets == 0u) {
        st->output.depth = depth;
        if (depth > TESSERACT_ASTAR_MAX_DEPTH) {
            /* The CPU resume must see the solved node that was just popped. */
            heap_push(st, &st->current_node);
            st->output.depth = 0u;
            finish(st, TESSERACT_ASTAR_STATUS_FALLBACK);
            return 0u;
        }
        walker = st->current_node.chain;
        uint32_t index = depth;
        while (walker != UINT32_MAX) {
            arena_read(st, walker, &link);
            st->output.errors[--index] = link.error;
            walker = link.parent;
        }
        finish(st, TESSERACT_ASTAR_STATUS_SOLVED);
        return 0u;
    }

    uint32_t visit = visited_probe(st, st->residual, st->current_node.num_dets, 1u);
    if (visit == 1u) {
        goto pop_next_node;
    }
    if (visit == 2u) {
        heap_push(st, &st->current_node);
        finish(st, TESSERACT_ASTAR_STATUS_FALLBACK);
        return 0u;
    }

    if (st->current_node.num_dets < st->min_num_dets) {
        st->min_num_dets = st->current_node.num_dets;
        uint32_t limit = DPU_INPUT_ARGUMENTS.det_beam == UINT32_MAX
                             ? UINT32_MAX
                             : st->min_num_dets + DPU_INPUT_ARGUMENTS.det_beam;
        if (limit < st->max_num_dets) {
            st->max_num_dets = limit;
        }
    }
    /* The expanded detector is the first active one in the detector order. */
    uint32_t min_detector = 0u;
    uint32_t min_position = UINT32_MAX;
    st->parent_detector_count = 0u;
    for (uint32_t word = 0; word < residual_words; ++word) {
        uint32_t bits_left = st->residual[word];
        while (bits_left != 0u) {
            uint32_t detector = word * 32u + ctz32(bits_left);
            bits_left &= bits_left - 1u;
            uint32_t position = st->use_positions ? st->detector_position[detector] : detector;
            if (position < min_position) {
                min_position = position;
                min_detector = detector;
            }
            st->parent_detectors[st->parent_detector_count++] = (uint16_t)detector;
        }
    }
    st->sibling_begin = mram_u32(DPU_INPUT_ARGUMENTS.d2e_offsets_offset, min_detector);
    uint32_t sibling_end = mram_u32(DPU_INPUT_ARGUMENTS.d2e_offsets_offset, min_detector + 1u);
    st->sibling_count = sibling_end - st->sibling_begin;
    if (st->sibling_count > TESSERACT_ASTAR_MAX_SIBLINGS ||
        st->heap_size + st->sibling_count > DPU_INPUT_ARGUMENTS.heap_capacity ||
        st->arena_size + st->sibling_count > DPU_INPUT_ARGUMENTS.arena_capacity) {
        /* Put the node back so the exported queue is complete. */
        heap_push(st, &st->current_node);
        finish(st, TESSERACT_ASTAR_STATUS_FALLBACK);
        return 0u;
    }
    ++st->total_expansions;
    st->next_parent_detector = 0u;
    st->parent_total = 0u;
    st->parent_has_inf = 0u;
    for (uint32_t j = 0; j < st->sibling_count; ++j) {
        uint32_t error_id = cached_mram_u32(DPU_INPUT_ARGUMENTS.d2e_error_ids_offset,
                                            st->sibling_begin + j, stream);
        st->sibling_position[error_id] = (uint8_t)j;
    }
    st->sibling_max_num_dets = st->max_num_dets;
    st->sibling_min_detector = min_detector;
    st->next_sibling = 0u;
    st->phase = PHASE_EVALUATE;
    return 1u;
}

static void push_results(search_state_t *st, mram_u32_cache_t *stream) {
    uint32_t min_detector = st->sibling_min_detector;
    for (uint32_t j = 0; j < st->sibling_count; ++j) {
        uint32_t error_id = cached_mram_u32(DPU_INPUT_ARGUMENTS.d2e_error_ids_offset,
                                            st->sibling_begin + j, stream);
        uint32_t heuristic = st->sibling_cost[j];
        st->sibling_position[error_id] = TESSERACT_ASTAR_NO_POSITION;
        st->total_candidates += st->sibling_evaluated[j];
        if (heuristic == TESSERACT_ASTAR_INF_COST) {
            continue;
        }
        uint32_t step = error_cost_q(error_id);
        if (step >= TESSERACT_ASTAR_INF_COST - st->current_node.path) {
            continue;
        }
        uint32_t path = st->current_node.path + step;
        if (heuristic >= TESSERACT_ASTAR_INF_COST - path) {
            continue;
        }
        if (st->arena_size >= DPU_INPUT_ARGUMENTS.arena_capacity) {
            finish(st, TESSERACT_ASTAR_STATUS_FALLBACK);
            return;
        }
        arena_node_t link;
        link.parent = st->current_node.chain;
        link.error = (uint16_t)error_id;
        link.min_detector = (uint16_t)min_detector;
        arena_write(st, st->arena_size, &link);

        heap_node_t node;
        node.cost = path + heuristic;
        node.path = path;
        node.chain = st->arena_size;
        node.num_dets = st->sibling_num_dets[j];
        node.depth = (uint16_t)(st->current_node.depth + 1u);
        ++st->arena_size;
        if (!heap_push(st, &node)) {
            finish(st, TESSERACT_ASTAR_STATUS_FALLBACK);
            return;
        }
        ++st->total_pushed;
        if (DPU_INPUT_ARGUMENTS.pqlimit != UINT32_MAX &&
            st->total_pushed + 1u > DPU_INPUT_ARGUMENTS.pqlimit) {
            finish(st, TESSERACT_ASTAR_STATUS_LOW_CONFIDENCE);
            return;
        }
    }
}

static uint32_t initialize_search(search_state_t *st, uint32_t group) {
    st->group = group;
    st->phase = PHASE_EVALUATE;
    st->heap_offset = DPU_INPUT_ARGUMENTS.heap_offset[group];
    st->arena_offset = DPU_INPUT_ARGUMENTS.arena_offset[group];
    st->visited_offset = DPU_INPUT_ARGUMENTS.visited_offset[group];
    st->output.reserved = 0u;
    if (DPU_INPUT_ARGUMENTS.num_detectors > TESSERACT_ASTAR_MAX_DETECTORS ||
        DPU_INPUT_ARGUMENTS.num_errors > TESSERACT_ASTAR_MAX_ERRORS) {
        finish(st, TESSERACT_ASTAR_STATUS_FALLBACK);
        return 0u;
    }
    for (uint32_t e = 0; e < DPU_INPUT_ARGUMENTS.num_errors; ++e) {
        st->sibling_position[e] = TESSERACT_ASTAR_NO_POSITION;
    }
    for (uint32_t i = 0; i < TESSERACT_ASTAR_MAX_DEPTH; ++i) {
        st->output.errors[i] = 0u;
    }
    st->output.depth = 0u;

    uint32_t stride_words = (residual_words + 1u) & ~1u;
    mram_read((__mram_ptr const void *)(heap_base() +
                                        DPU_INPUT_ARGUMENTS.syndrome_words_offset[group]),
              st->initial_residual, stride_words * 4u);
    st->use_positions = DPU_INPUT_ARGUMENTS.order_positions_offset != UINT32_MAX;
    if (st->use_positions) {
        uint32_t table_bytes = (DPU_INPUT_ARGUMENTS.num_detectors * 2u + 7u) & ~7u;
        mram_read((__mram_ptr const void *)(heap_base() + DPU_INPUT_ARGUMENTS.order_positions_offset +
                                            DPU_INPUT_ARGUMENTS.order_index[group] * table_bytes),
                  st->detector_position, table_bytes);
    }
    uint32_t num_dets = 0u;
    for (uint32_t word = 0; word < residual_words; ++word) {
        uint32_t bits = st->initial_residual[word];
        num_dets += popcount32(bits);
        while (bits != 0u) {
            uint32_t detector = word * 32u + ctz32(bits);
            bits &= bits - 1u;
            uint32_t begin = mram_u32(DPU_INPUT_ARGUMENTS.d2e_offsets_offset, detector);
            uint32_t end = mram_u32(DPU_INPUT_ARGUMENTS.d2e_offsets_offset, detector + 1u);
            if (begin == end) {
                finish(st, TESSERACT_ASTAR_STATUS_LOW_CONFIDENCE);
                return 0u;
            }
        }
    }

    st->heap_size = 0u;
    st->heap_peak = 0u;
    st->arena_size = 0u;
    st->total_expansions = 0u;
    st->total_candidates = 0u;
    st->total_pushed = 0u;
    st->visited_count = 0u;
    st->min_num_dets = num_dets;
    st->max_num_dets = DPU_INPUT_ARGUMENTS.det_beam == UINT32_MAX
                           ? UINT32_MAX
                           : num_dets + DPU_INPUT_ARGUMENTS.det_beam;

    heap_node_t root;
    root.cost = 0u;
    root.path = 0u;
    root.chain = UINT32_MAX;
    root.num_dets = (uint16_t)num_dets;
    root.depth = 0u;
    heap_push(st, &root);
    return 1u;
}

/* ------------------------------------------------------------- tasklets */

/* Per-detector heuristic terms of the expanded node, shared by its siblings. */
static void evaluate_parent(search_state_t *st, mram_u32_cache_t *stream) {
    uint32_t local_total = 0u;
    uint32_t local_inf = 0u;
    mutex_id_t queue = group_queue_mutex(st->group);
    while (1) {
        mutex_lock(queue);
        uint32_t index = st->next_parent_detector++;
        mutex_unlock(queue);
        if (index >= st->parent_detector_count) {
            break;
        }
        uint32_t detector = st->parent_detectors[index];
        uint32_t cost = evaluate_detector(st, st->residual, detector, NO_PREFIX, stream);
        st->parent_cost[detector] = cost;
        if (cost == TESSERACT_ASTAR_INF_COST) {
            local_inf = 1u;
        } else {
            local_total += cost;
        }
    }
    mutex_id_t total = group_total_mutex(st->group);
    mutex_lock(total);
    st->parent_total += local_total;
    st->parent_has_inf |= local_inf;
    mutex_unlock(total);
}

static void evaluate_siblings(search_state_t *st, mram_u32_cache_t *stream) {
    uint32_t residual[RESIDUAL_WORDS] __attribute__((aligned(8)));
    mutex_id_t queue = group_queue_mutex(st->group);
    while (1) {
        mutex_lock(queue);
        uint32_t j = st->next_sibling++;
        mutex_unlock(queue);
        if (j >= st->sibling_count) {
            break;
        }
        uint32_t error_id = cached_mram_u32(DPU_INPUT_ARGUMENTS.d2e_error_ids_offset,
                                            st->sibling_begin + j, stream);
        st->sibling_cost[j] = TESSERACT_ASTAR_INF_COST;
        st->sibling_evaluated[j] = 0u;
        if ((st->blocked[error_id >> 5] >> (error_id & 31u)) & 1u) {
            continue;
        }
        astar_error_record_t record __attribute__((aligned(8)));
        read_error_record(error_id, &record);
        for (uint32_t word = 0; word < residual_words; ++word) {
            residual[word] = st->residual[word];
        }
        for (uint32_t k = 0; k < record.degree; ++k) {
            uint32_t detector = record.detectors[k];
            residual[detector >> 5] ^= 1u << (detector & 31u);
        }
        uint32_t num_dets = 0u;
        for (uint32_t word = 0; word < residual_words; ++word) {
            num_dets += popcount32(residual[word]);
        }
        if (num_dets > st->sibling_max_num_dets) {
            continue;
        }
        if (visited_probe(st, residual, num_dets, 0u) == 1u) {
            continue;
        }
        st->sibling_num_dets[j] = (uint16_t)num_dets;
        st->sibling_evaluated[j] = 1u;
        if (st->parent_has_inf) {
            continue;
        }
        st->sibling_cost[j] = sibling_heuristic(st, residual, &record, error_id, j, stream);
#ifdef ASTAR_VERIFY_INCREMENTAL
        {
            uint32_t reference = 0u;
            uint32_t reference_inf = 0u;
            for (uint32_t word = 0; word < residual_words && !reference_inf; ++word) {
                uint32_t bits_left = residual[word];
                while (bits_left != 0u) {
                    uint32_t detector = word * 32u + ctz32(bits_left);
                    bits_left &= bits_left - 1u;
                    uint32_t cost = evaluate_detector(st, residual, detector, j, stream);
                    if (cost == TESSERACT_ASTAR_INF_COST) {
                        reference_inf = 1u;
                        break;
                    }
                    reference += cost;
                }
            }
            uint32_t reference_value = reference_inf ? TESSERACT_ASTAR_INF_COST : reference;
            if (reference_value != st->sibling_cost[j] && st->output.reserved == 0u) {
                mutex_id_t total = group_total_mutex(st->group);
                mutex_lock(total);
                st->output.reserved = 1u;
                st->debug_info[0] = (uint16_t)st->total_expansions;
                st->debug_info[1] = (uint16_t)j;
                st->debug_info[2] = (uint16_t)error_id;
                st->debug_info[3] = (uint16_t)(st->sibling_cost[j] & 0xFFFFu);
                st->debug_info[4] = (uint16_t)(st->sibling_cost[j] >> 16);
                st->debug_info[5] = (uint16_t)(reference_value & 0xFFFFu);
                st->debug_info[6] = (uint16_t)(reference_value >> 16);
                st->debug_info[7] = (uint16_t)st->sibling_min_detector;
                mutex_unlock(total);
            }
        }
#endif
    }
}

extern int main_kernel1(void);
int main(void) {
    if (DPU_INPUT_ARGUMENTS.kernel != astar_kernel1) {
        return 1;
    }
    return main_kernel1();
}

int main_kernel1(void) {
    uint32_t tasklet_id = me();
    uint32_t group = tasklet_id / TASKLETS_PER_GROUP;
    uint32_t leader = (tasklet_id % TASKLETS_PER_GROUP) == 0u;
    search_state_t *st = &groups[group];
    mram_u32_cache_t stream;
    stream.tag = UINT32_MAX;

    if (group >= DPU_INPUT_ARGUMENTS.num_syndromes) {
        return 0;
    }
    if (leader) {
        residual_words = (DPU_INPUT_ARGUMENTS.num_detectors + 31u) >> 5;
        blocked_words = (DPU_INPUT_ARGUMENTS.num_errors + 31u) >> 5;
        if (initialize_search(st, group)) {
            prepare_expansion(st, &stream);
        }
    }
    group_barrier(group);

    while (st->phase != PHASE_DONE) {
        evaluate_parent(st, &stream);
        group_barrier(group);
        evaluate_siblings(st, &stream);
        group_barrier(group);
        if (leader) {
            push_results(st, &stream);
            if (st->phase != PHASE_DONE) {
                prepare_expansion(st, &stream);
            }
        }
        group_barrier(group);
    }
    return 0;
}
