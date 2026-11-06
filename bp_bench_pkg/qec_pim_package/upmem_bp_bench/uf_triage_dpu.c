/* uf_triage_dpu.c — Union-Find Triage Decoder for 3D Surface Code
 *
 * Tier 1 (PIM): UF decoder on DPU, WRAM-only for d <= 17
 * Runs weighted union-find on 3D detector graph.
 * Classifies shots as "easy" (PIM handles) or "hard" (CPU MWPM needed).
 *
 * UF Algorithm (Kruskal-style, conservative):
 *   1. Initialize parent[i]=i, mark event nodes active
 *   2. Process edges from low to high weight
 *   3. For edge (u,v): union if BOTH clusters are active
 *   4. After all edges: count odd-parity clusters
 *   5. Triage: odd-parity interior cluster with no boundary => hard
 *
 * MRAM layout:
 *   [events_off]    events[num_events]   uint32 node indices
 *   [result_off]    uf_result_t          output
 */

#include <stdint.h>
#include <stdbool.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <defs.h>

#define ALIGN8(x)  (((x) + 7) & ~7u)
#define MAX_NODES  5120
#define MAX_EVENTS 256

static inline uint32_t w_sp(uint32_t r, uint32_t c, uint32_t t) {
    return 1000u + ((r * 7 + c * 13 + t * 3) & 0xFFu);
}
static inline uint32_t w_tm(uint32_t r, uint32_t c, uint32_t t) {
    return 500u + ((r * 3 + c * 7 + t * 11) & 0x7Fu);
}

typedef struct {
    uint32_t d, T, total_nodes;
    uint32_t num_events;
    uint32_t events_off;
    uint32_t syndrome_off;
    uint32_t result_off;
    uint32_t kernel;
} uf_args_t;

typedef struct {
    uint64_t cycle_count;
    uint32_t num_clusters;
    uint32_t num_merges;
    uint32_t easy_case;
    uint32_t correction_size;
} uf_result_t;

__host uf_args_t DPU_INPUT_ARGUMENTS;

/* WRAM arrays -- all UF computation stays here.
 * parent[]: 20 KB, rank_arr[]: 5 KB, active[]: 5 KB,
 * parity[]: 5 KB, boundary[]: 5 KB  => ~40 KB < 64 KB */
uint32_t parent[MAX_NODES];
uint8_t  rank_arr[MAX_NODES];
uint8_t  active[MAX_NODES];
uint8_t  parity[MAX_NODES];
uint8_t  boundary[MAX_NODES];

__dma_aligned uint32_t ev_buf[MAX_EVENTS];
__dma_aligned uf_result_t res_buf;

static uint32_t find(uint32_t x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

static uint32_t unite(uint32_t a, uint32_t b) {
    a = find(a); b = find(b);
    if (a == b) return 0;
    if (rank_arr[a] < rank_arr[b]) { uint32_t t = a; a = b; b = t; }
    parent[b] = a;
    if (rank_arr[a] == rank_arr[b]) rank_arr[a]++;
    if (active[b]) active[a] = 1;
    parity[a] ^= parity[b];
    if (boundary[b]) boundary[a] = 1;
    return 1;
}

int main(void) {
    uf_args_t a = DPU_INPUT_ARGUMENTS;
    uint32_t d = a.d, T = a.T, nn = a.total_nodes;
    uint32_t num_ev = a.num_events;

    __mram_ptr uint8_t *heap = (__mram_ptr uint8_t *)DPU_MRAM_HEAP_POINTER;

    uint32_t dd = d * d;

    /* Step 1: Init WRAM arrays */
    for (uint32_t i = 0; i < nn; i++) {
        parent[i] = i;
        rank_arr[i] = 0;
        active[i] = 0;
        parity[i] = 0;
        /* Boundary: first/last layer of each dimension */
        uint32_t it = i / dd;
        uint32_t ir = (i % dd) / d;
        uint32_t ic = i % d;
        boundary[i] = (ir == 0 || ir == d - 1 ||
                       ic == 0 || ic == d - 1 ||
                       it == 0 || it == T - 1) ? 1 : 0;
    }

    /* Step 2: Load events, mark active + parity */
    if (num_ev > 0) {
        uint32_t ev_bytes = ALIGN8(num_ev * 4);
        if (ev_bytes < 8) ev_bytes = 8;
        if (ev_bytes > sizeof(ev_buf)) ev_bytes = sizeof(ev_buf);
        mram_read((__mram_ptr void *)(heap + a.events_off), ev_buf, ev_bytes);
        for (uint32_t i = 0; i < num_ev; i++) {
            if (ev_buf[i] < nn) {
                active[ev_buf[i]] = 1;
                parity[ev_buf[i]] ^= 1;
            }
        }
    }

    perfcounter_config(COUNT_CYCLES, true);

    uint32_t merges = 0;

    /* Step 3: Kruskal-style UF -- process edges by weight
     *
     * STRICT CONSERVATIVE: only union if BOTH endpoints' clusters
     * contain detection events. No growth pass.
     * This keeps clusters tight around event locations.
     */

    /* Phase 1: Temporal edges (weight ~500-627, lighter) */
    if (T > 1) {
        for (uint32_t t = 0; t < T - 1; t++)
            for (uint32_t r = 0; r < d; r++)
                for (uint32_t c = 0; c < d; c++) {
                    uint32_t u = t * dd + r * d + c;
                    uint32_t v = (t + 1) * dd + r * d + c;
                    uint32_t ru = find(u), rv = find(v);
                    if (ru != rv && active[ru] && active[rv])
                        merges += unite(u, v);
                }
    }

    /* Phase 2: Spatial-right (weight ~1000-1255) */
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t r = 0; r < d; r++)
            for (uint32_t c = 0; c < d - 1; c++) {
                uint32_t u = t * dd + r * d + c;
                uint32_t v = t * dd + r * d + (c + 1);
                uint32_t ru = find(u), rv = find(v);
                if (ru != rv && active[ru] && active[rv])
                    merges += unite(u, v);
            }

    /* Phase 3: Spatial-down (weight ~1000-1255) */
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t r = 0; r < d - 1; r++)
            for (uint32_t c = 0; c < d; c++) {
                uint32_t u = t * dd + r * d + c;
                uint32_t v = t * dd + (r + 1) * d + c;
                uint32_t ru = find(u), rv = find(v);
                if (ru != rv && active[ru] && active[rv])
                    merges += unite(u, v);
            }

    /* Step 4: Triage
     *
     * Count clusters with odd parity that are NOT touching boundary.
     * - 0 such clusters: easy (all satisfied)
     * - Even number: easy (can pair them in matching)
     * - Odd number: HARD (unpaired interior defect, needs MWPM)
     */
    uint32_t num_active = 0;
    uint32_t num_odd_interior = 0;

    for (uint32_t i = 0; i < nn; i++) {
        if (find(i) == i && active[i]) {
            num_active++;
            if (parity[i] && !boundary[i])
                num_odd_interior++;
        }
    }

    uint32_t easy = (num_odd_interior == 0 || (num_odd_interior & 1) == 0) ? 1 : 0;
    uint32_t correction_edges = easy ? merges : 0;

    uint64_t cyc = perfcounter_get();

    res_buf.cycle_count = cyc;
    res_buf.num_clusters = num_active;
    res_buf.num_merges = merges;
    res_buf.easy_case = easy;
    res_buf.correction_size = correction_edges;

    uint32_t res_bytes = ALIGN8(sizeof(uf_result_t));
    if (res_bytes < 8) res_bytes = 8;
    mram_write(&res_buf, (__mram_ptr void *)(heap + a.result_off), res_bytes);

    return 0;
}
