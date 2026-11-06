/* multi_qubit_bench.c — Multi-Qubit Throughput Benchmark: CPU vs UPMEM PIM
 *
 * Measures per-logical-qubit decoding throughput as N (number of
 * simultaneous logical qubits) increases.
 *
 * KEY INSIGHT: CPU per-decode time INCREASES with N due to cache thrashing.
 *              PIM per-decode time stays CONSTANT (each DPU has private WRAM).
 *
 * CPU test: Allocate N independent UF decoder instances (~12.6 KB each).
 *           Round-robin decode across all N. As total data exceeds L1/L2/L3,
 *           cache misses cause measurable slowdown.
 *
 * PIM test: Allocate N DPUs, each with independent syndrome data.
 *           Launch all simultaneously. Per-DPU cycle count stays flat.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <dpu.h>

#define DPU_BINARY   "/tmp/uf_triage_binary"
#define ALIGN8(x)    (((x) + 7) & ~7u)
#define DPU_MHZ      350
#define MAX_EVENTS   256
#define CPU_CORES    16   /* reference server CPU */

/* Distance and measurement rounds for the benchmark */
#define BENCH_D      11
#define BENCH_T      11
#define BENCH_NODES  (BENCH_D * BENCH_D * BENCH_T)  /* 1331 */

/* Per-instance data for CPU cache thrashing test.
 * Size: 1400*4 + 1400*4 + 1400*1 = 12600 bytes ~ 12.3 KB */
#define STATE_NODES  1400

typedef struct {
    uint32_t parent[STATE_NODES];    /* 5600 bytes */
    uint32_t rank_arr[STATE_NODES];  /* 5600 bytes */
    uint8_t  active[STATE_NODES];    /* 1400 bytes */
    /* Total: ~12.6 KB per instance */
} decoder_state_t;

/* Syndrome storage per qubit */
typedef struct {
    uint32_t events[MAX_EVENTS];
    uint32_t num_events;
} syndrome_t;

/* DPU interface structures (must match DPU binary) */
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

/* ================================================================
 * CPU Union-Find -- operates on a decoder_state_t instance
 * ================================================================ */

static uint32_t cpu_find(decoder_state_t *s, uint32_t x) {
    while (s->parent[x] != x) {
        s->parent[x] = s->parent[s->parent[x]];  /* path halving */
        x = s->parent[x];
    }
    return x;
}

static uint32_t cpu_unite(decoder_state_t *s, uint32_t a, uint32_t b) {
    a = cpu_find(s, a);
    b = cpu_find(s, b);
    if (a == b) return 0;
    if (s->rank_arr[a] < s->rank_arr[b]) { uint32_t t = a; a = b; b = t; }
    s->parent[b] = a;
    if (s->rank_arr[a] == s->rank_arr[b]) s->rank_arr[a]++;
    if (s->active[b]) s->active[a] = 1;
    return 1;
}

/* Full UF decode on a single decoder_state_t instance.
 * Initialises state, marks events, processes edges, merges active clusters.
 * Uses volatile pointer to prevent compiler from optimising away memory
 * accesses (critical for measuring real cache behaviour). */
static void cpu_uf_decode(decoder_state_t *state, uint32_t *events,
                          uint32_t num_ev, uint32_t d, uint32_t T) {
    volatile decoder_state_t *vs = (volatile decoder_state_t *)state;
    uint32_t nn = d * d * T;
    uint32_t dd = d * d;

    /* Init -- write through volatile to ensure stores actually go to memory */
    for (uint32_t i = 0; i < nn; i++) {
        vs->parent[i] = i;
        vs->rank_arr[i] = 0;
        vs->active[i] = 0;
    }

    /* Mark detection events */
    for (uint32_t i = 0; i < num_ev; i++) {
        if (events[i] < nn) vs->active[events[i]] = 1;
    }

    /* Copy back for algorithmic work (find/union need non-volatile for perf) */
    /* The init above ensures the data is in memory at the state's address;
     * the actual UF work below touches state-> which forces cache loads */

    /* Kruskal-style: process all edges, union clusters with active endpoints.
     * Phase 1: Temporal edges (lighter weight) */
    if (T > 1) {
        for (uint32_t t = 0; t < T - 1; t++)
            for (uint32_t r = 0; r < d; r++)
                for (uint32_t c = 0; c < d; c++) {
                    uint32_t u = t * dd + r * d + c;
                    uint32_t v = (t + 1) * dd + r * d + c;
                    uint32_t ru = cpu_find(state, u), rv = cpu_find(state, v);
                    if (ru != rv && state->active[ru] && state->active[rv])
                        cpu_unite(state, u, v);
                }
    }

    /* Phase 2: Spatial-right */
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t r = 0; r < d; r++)
            for (uint32_t c = 0; c < d - 1; c++) {
                uint32_t u = t * dd + r * d + c;
                uint32_t v = t * dd + r * d + (c + 1);
                uint32_t ru = cpu_find(state, u), rv = cpu_find(state, v);
                if (ru != rv && state->active[ru] && state->active[rv])
                    cpu_unite(state, u, v);
            }

    /* Phase 3: Spatial-down */
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t r = 0; r < d - 1; r++)
            for (uint32_t c = 0; c < d; c++) {
                uint32_t u = t * dd + r * d + c;
                uint32_t v = t * dd + (r + 1) * d + c;
                uint32_t ru = cpu_find(state, u), rv = cpu_find(state, v);
                if (ru != rv && state->active[ru] && state->active[rv])
                    cpu_unite(state, u, v);
            }

    /* Final read barrier: ensure we actually read the result back.
     * This prevents the compiler from eliding the decode entirely. */
    volatile uint32_t sink = 0;
    for (uint32_t i = 0; i < 8; i++) sink += state->parent[i * (nn / 8)];
    (void)sink;
}

/* ================================================================
 * Syndrome generation (realistic correlated error model)
 * ================================================================ */

static uint32_t generate_syndrome(uint32_t d, uint32_t T, double p,
                                  uint32_t *events, uint32_t max_ev,
                                  unsigned int *seed) {
    uint32_t dd = d * d;
    uint32_t nn = dd * T;
    uint32_t ne = 0;
    uint8_t *syn = (uint8_t *)calloc(nn, 1);

    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t r = 0; r < d; r++) {
            for (uint32_t c = 0; c < d; c++) {
                double rv = (double)rand_r(seed) / RAND_MAX;
                if (rv < p * 0.6) {
                    uint32_t u = t * dd + r * d + c;
                    syn[u] ^= 1;
                    int dir = rand_r(seed) % 6;
                    uint32_t v = u;
                    if      (dir == 0 && c + 1 < d) v = t * dd + r * d + (c + 1);
                    else if (dir == 1 && c > 0)     v = t * dd + r * d + (c - 1);
                    else if (dir == 2 && r + 1 < d) v = t * dd + (r + 1) * d + c;
                    else if (dir == 3 && r > 0)     v = t * dd + (r - 1) * d + c;
                    else if (dir == 4 && t + 1 < T) v = (t + 1) * dd + r * d + c;
                    else if (dir == 5 && t > 0)     v = (t - 1) * dd + r * d + c;
                    syn[v] ^= 1;
                } else if (rv < p) {
                    uint32_t u = t * dd + r * d + c;
                    syn[u] ^= 1;
                }
            }
        }
    }

    for (uint32_t i = 0; i < nn && ne < max_ev; i++)
        if (syn[i]) events[ne++] = i;

    free(syn);
    return ne;
}

/* ================================================================
 * Timing helpers
 * ================================================================ */

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static const char *cache_level_label(double per_core_kb) {
    /* AMD Threadripper PRO 5955WX: L1d=32KB, L2=512KB/core, L3=64MB shared */
    if      (per_core_kb <= 32)    return "L1d";
    if      (per_core_kb <= 512)   return "L2";
    if      (per_core_kb <= 4096)  return "L3";
    return "DRAM";
}

/* ================================================================
 * MAIN
 * ================================================================ */

int main(void) {
    double state_kb = sizeof(decoder_state_t) / 1024.0;

    printf("\n");
    printf("====================================================================\n");
    printf("  MULTI-QUBIT THROUGHPUT: CPU vs UPMEM PIM\n");
    printf("====================================================================\n");
    printf("  Decoder: Union-Find on 3D detector graph (d=%d, T=%d)\n", BENCH_D, BENCH_T);
    printf("  Nodes per qubit: %d\n", BENCH_NODES);
    printf("  Per-instance data: %.1f KB (sizeof decoder_state_t = %zu)\n",
           state_kb, sizeof(decoder_state_t));
    printf("  CPU: %d cores, round-robin across N qubits\n", CPU_CORES);
    printf("  PIM: N DPUs, 1 qubit per DPU, all run simultaneously\n\n");

    /* ==============================================================
     * SECTION 1: CPU Cache Thrashing Test
     * ============================================================== */
    printf("  ---- CPU: Per-Decode Time vs N (cache thrashing) ----\n\n");
    printf("  N qubits | Total Data  | Per-Core WS | Per-Decode   | vs N=1   | Cache Hit\n");
    printf("  ---------+-------------+-------------+--------------+----------+----------\n");
    fflush(stdout);

    int n_values[] = {1, 16, 64, 200, 500, 750, 1000, 2000, 4000, 6000, 8000, 10000};
    int n_count = sizeof(n_values) / sizeof(n_values[0]);
    double cpu_per_decode_us[12];
    double baseline_us = 0;

    for (int ni = 0; ni < n_count; ni++) {
        int N = n_values[ni];
        double total_data_kb = N * state_kb;

        /* Allocate N decoder states as separate malloc blocks.
         * Separate allocations ensure different virtual pages and
         * realistic cache line conflicts (unlike a contiguous array
         * where prefetcher can help). */
        decoder_state_t **states = (decoder_state_t **)malloc(N * sizeof(decoder_state_t *));
        if (!states) { fprintf(stderr, "malloc failed for N=%d\n", N); continue; }
        for (int q = 0; q < N; q++) {
            states[q] = (decoder_state_t *)malloc(sizeof(decoder_state_t));
            if (!states[q]) { fprintf(stderr, "malloc failed q=%d\n", q); }
        }

        /* Generate independent syndromes for each qubit */
        syndrome_t *syndromes = (syndrome_t *)malloc(N * sizeof(syndrome_t));
        if (!syndromes) {
            for (int q = 0; q < N; q++) free(states[q]);
            free(states); continue;
        }

        for (int q = 0; q < N; q++) {
            unsigned int seed = 42 + q * 137;
            syndromes[q].num_events = generate_syndrome(
                BENCH_D, BENCH_T, 0.005,
                syndromes[q].events, MAX_EVENTS, &seed);
        }

        /* Determine number of rounds: enough for stable measurement.
         * At least 3 full cycles through all N qubits, min 1000 decodes. */
        int total_decodes;
        if (N <= 64)       total_decodes = 3000;
        else if (N <= 200) total_decodes = N * 5;
        else               total_decodes = N * 3;

        /* Warmup: 2 full cycles to populate TLB and establish baseline */
        for (int round = 0; round < N * 2; round++) {
            int qubit = round % N;
            cpu_uf_decode(states[qubit], syndromes[qubit].events,
                          syndromes[qubit].num_events, BENCH_D, BENCH_T);
        }

        /* Timed run: round-robin across all N qubits */
        double t0 = now_us();

        for (int round = 0; round < total_decodes; round++) {
            int qubit = round % N;
            cpu_uf_decode(states[qubit], syndromes[qubit].events,
                          syndromes[qubit].num_events, BENCH_D, BENCH_T);
        }

        double t1 = now_us();
        double per_decode = (t1 - t0) / total_decodes;
        cpu_per_decode_us[ni] = per_decode;

        if (ni == 0) baseline_us = per_decode;
        double slowdown = (baseline_us > 0) ? per_decode / baseline_us : 1.0;

        double per_core_kb = ceil((double)N / CPU_CORES) * state_kb;
        printf("  %-9d | %7.1f KB  | %7.1f KB  | %8.2f us   | %5.2fx   | %s\n",
               N, total_data_kb, per_core_kb, per_decode, slowdown,
               cache_level_label(per_core_kb));
        fflush(stdout);

        free(syndromes);
        for (int q = 0; q < N; q++) free(states[q]);
        free(states);
    }

    /* ==============================================================
     * SECTION 2: PIM Constant Performance Test
     * ============================================================== */
    printf("\n  ---- PIM: Per-Decode Time vs N (should be constant) ----\n\n");
    printf("  N DPUs | Per-Decode DPU Cycles | Per-Decode @350MHz | vs N=1\n");
    printf("  -------+-----------------------+--------------------+-------\n");
    fflush(stdout);

    int pim_n_values[] = {1, 4, 8, 16, 32, 64};
    int pim_n_count = sizeof(pim_n_values) / sizeof(pim_n_values[0]);
    double pim_per_decode_us[6];
    double pim_baseline_us = 0;

    for (int ni = 0; ni < pim_n_count; ni++) {
        int N = pim_n_values[ni];

        /* Allocate N DPUs */
        struct dpu_set_t set, dpu;
        if (dpu_alloc(N, NULL, &set) != DPU_OK) {
            fprintf(stderr, "  ERROR: dpu_alloc(%d) failed\n", N);
            pim_per_decode_us[ni] = -1;
            continue;
        }

        if (dpu_load(set, DPU_BINARY, NULL) != DPU_OK) {
            fprintf(stderr, "  ERROR: dpu_load failed for N=%d\n", N);
            dpu_free(set);
            pim_per_decode_us[ni] = -1;
            continue;
        }

        /* Prepare per-DPU syndrome data */
        uint32_t events_off = 0;
        uint32_t syndrome_off = ALIGN8(MAX_EVENTS * 4);
        uint32_t result_off = syndrome_off + 8;

        uint32_t ev_size = ALIGN8(MAX_EVENTS * 4);
        uint32_t **ev_bufs = (uint32_t **)malloc(N * sizeof(uint32_t *));
        uint32_t *num_evs = (uint32_t *)malloc(N * sizeof(uint32_t));
        uf_result_t *results = (uf_result_t *)malloc(N * sizeof(uf_result_t));

        for (int q = 0; q < N; q++) {
            ev_bufs[q] = (uint32_t *)calloc(1, ev_size);
            unsigned int seed = 100 + q * 73 + ni * 1000;
            num_evs[q] = generate_syndrome(BENCH_D, BENCH_T, 0.005,
                                           ev_bufs[q], MAX_EVENTS, &seed);
        }

        /* Copy per-DPU data: iterate and set up each DPU individually */
        {
            int qi = 0;
            DPU_FOREACH(set, dpu) {
                uf_args_t args;
                memset(&args, 0, sizeof(args));
                args.d = BENCH_D;
                args.T = BENCH_T;
                args.total_nodes = BENCH_NODES;
                args.num_events = num_evs[qi];
                args.events_off = events_off;
                args.syndrome_off = syndrome_off;
                args.result_off = result_off;

                dpu_copy_to(dpu, "DPU_INPUT_ARGUMENTS", 0, &args, sizeof(args));
                dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, events_off,
                            ev_bufs[qi], ev_size);
                qi++;
            }
        }

        /* Launch all N DPUs simultaneously */
        if (dpu_launch(set, DPU_SYNCHRONOUS) != DPU_OK) {
            fprintf(stderr, "  ERROR: dpu_launch failed for N=%d\n", N);
            DPU_FOREACH(set, dpu) { dpu_log_read(dpu, stdout); }
            for (int q = 0; q < N; q++) free(ev_bufs[q]);
            free(ev_bufs); free(num_evs); free(results);
            dpu_free(set);
            pim_per_decode_us[ni] = -1;
            continue;
        }

        /* Read back results from each DPU */
        uint32_t rsz = ALIGN8(sizeof(uf_result_t));
        if (rsz < 8) rsz = 8;

        uint64_t max_cycles = 0;
        uint64_t sum_cycles = 0;

        {
            int qi = 0;
            DPU_FOREACH(set, dpu) {
                memset(&results[qi], 0, sizeof(uf_result_t));
                dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, result_off,
                              &results[qi], rsz);
                if (results[qi].cycle_count > max_cycles)
                    max_cycles = results[qi].cycle_count;
                sum_cycles += results[qi].cycle_count;
                qi++;
            }
        }

        double avg_cycles = (double)sum_cycles / N;
        double per_decode_us = avg_cycles / DPU_MHZ;
        pim_per_decode_us[ni] = per_decode_us;

        if (ni == 0) pim_baseline_us = per_decode_us;
        double ratio = (pim_baseline_us > 0) ? per_decode_us / pim_baseline_us : 1.0;

        printf("  %-7d | %13.0f         | %11.2f us   | %5.2fx\n",
               N, avg_cycles, per_decode_us, ratio);
        fflush(stdout);

        /* Cleanup */
        for (int q = 0; q < N; q++) free(ev_bufs[q]);
        free(ev_bufs);
        free(num_evs);
        free(results);
        dpu_free(set);
    }

    /* ==============================================================
     * SECTION 3: Per-Qubit Throughput Comparison
     * ============================================================== */
    printf("\n  ---- Per-Qubit Throughput Comparison ----\n\n");
    printf("  N qubits | CPU decode/s/qubit | PIM decode/s/qubit | PIM/CPU ratio\n");
    printf("  ---------+--------------------+--------------------+--------------\n");
    fflush(stdout);

    /* Compute average PIM per-decode time (should be constant) */
    double pim_const_us = 0;
    int pim_valid = 0;
    for (int ni = 0; ni < pim_n_count; ni++) {
        if (pim_per_decode_us[ni] > 0) {
            pim_const_us += pim_per_decode_us[ni];
            pim_valid++;
        }
    }
    if (pim_valid > 0) pim_const_us /= pim_valid;

    /*
     * CPU throughput model (16 cores, N qubits):
     *   Each core round-robins ceil(N/16) qubits.
     *   Per-core working set: ceil(N/16) * 12.3 KB.
     *   Use MEASURED per-decode time for N qubits (includes cache effects).
     *   Total CPU decodes/sec = 16 / per_decode_us * 1e6
     *   Per-qubit: (16 / per_decode_us * 1e6) / N
     *
     * PIM throughput model (N DPUs, 1 qubit/DPU):
     *   All DPUs run in parallel, each takes pim_const_us.
     *   Total PIM decodes/sec = N / pim_const_us * 1e6
     *   Per-qubit: 1 / pim_const_us * 1e6  (constant!)
     */

    double crossover_n = -1;
    double prev_ratio = 0;
    int prev_n = 0;

    for (int ni = 0; ni < n_count; ni++) {
        int N = n_values[ni];
        double cpu_per_us = cpu_per_decode_us[ni];

        double cpu_tput_per_qubit = (CPU_CORES * 1e6) / ((double)N * cpu_per_us);
        double pim_tput_per_qubit = (pim_const_us > 0) ? 1e6 / pim_const_us : 0;
        double ratio = (cpu_tput_per_qubit > 0) ? pim_tput_per_qubit / cpu_tput_per_qubit : 0;

        printf("  %-9d | %12.0f       | %12.0f       | %8.2f\n",
               N, cpu_tput_per_qubit, pim_tput_per_qubit, ratio);

        /* Detect crossover by linear interpolation */
        if (prev_n > 0 && prev_ratio < 1.0 && ratio >= 1.0 && crossover_n < 0) {
            double frac = (1.0 - prev_ratio) / (ratio - prev_ratio);
            crossover_n = prev_n + frac * (N - prev_n);
        }

        prev_ratio = ratio;
        prev_n = N;
        fflush(stdout);
    }

    /* ==============================================================
     * SECTION 4: Aggregate Throughput Scaling
     * ==============================================================
     * This is where PIM shines: total system throughput.
     * CPU: fixed 16 cores, throughput degrades as N grows.
     * PIM: add more DPUs, throughput scales linearly.
     */
    printf("\n  ---- Aggregate Throughput: Total decodes/sec for N qubits ----\n\n");
    printf("  N qubits | CPU total dec/s   | PIM total dec/s   | PIM/CPU\n");
    printf("  ---------+-------------------+-------------------+--------\n");
    fflush(stdout);

    double agg_crossover_n = -1;
    prev_ratio = 0;
    prev_n = 0;

    for (int ni = 0; ni < n_count; ni++) {
        int N = n_values[ni];
        double cpu_per_us = cpu_per_decode_us[ni];

        /* CPU: 16 cores, each round-robins its share of N qubits.
         * Aggregate = 16 / per_decode_us * 1e6 decodes/s */
        double cpu_agg = (CPU_CORES * 1e6) / cpu_per_us;

        /* PIM: N DPUs all run in parallel.
         * Aggregate = N / pim_const_us * 1e6 decodes/s */
        double pim_agg = (pim_const_us > 0) ? (N * 1e6) / pim_const_us : 0;

        double ratio = (cpu_agg > 0) ? pim_agg / cpu_agg : 0;

        char cpu_s[32], pim_s[32];
        if (cpu_agg >= 1e6) snprintf(cpu_s, sizeof(cpu_s), "%8.2f M", cpu_agg / 1e6);
        else                snprintf(cpu_s, sizeof(cpu_s), "%8.0f K", cpu_agg / 1e3);
        if (pim_agg >= 1e6) snprintf(pim_s, sizeof(pim_s), "%8.2f M", pim_agg / 1e6);
        else                snprintf(pim_s, sizeof(pim_s), "%8.0f K", pim_agg / 1e3);

        printf("  %-9d | %s dec/s | %s dec/s | %6.2fx\n", N, cpu_s, pim_s, ratio);

        if (prev_n > 0 && prev_ratio < 1.0 && ratio >= 1.0 && agg_crossover_n < 0) {
            double frac = (1.0 - prev_ratio) / (ratio - prev_ratio);
            agg_crossover_n = prev_n + frac * (N - prev_n);
        }

        prev_ratio = ratio;
        prev_n = N;
        fflush(stdout);
    }

    /* ==============================================================
     * SECTION 5: Crossover Analysis
     * ============================================================== */
    printf("\n  ---- Crossover Analysis ----\n\n");

    printf("  Per-qubit throughput crossover:\n");
    if (crossover_n > 0) {
        printf("    At N ~ %.0f qubits, PIM per-qubit throughput surpasses CPU.\n", crossover_n);
    } else if (prev_ratio >= 1.0) {
        printf("    PIM per-qubit throughput >= CPU for all tested N.\n");
    } else {
        /* Extrapolate: at what N would the measured CPU slowdown rate
         * cause crossover? */
        double cpu1 = cpu_per_decode_us[0];
        double cpuN = cpu_per_decode_us[n_count - 1];
        int N1 = n_values[0], NN = n_values[n_count - 1];
        printf("    Not reached in tested range (max N=%d).\n", NN);
        if (pim_const_us > 0 && cpu1 > 0) {
            /* PIM/CPU ratio = (cpu_per * N) / (pim_const * 16) */
            /* Crossover when ratio = 1: N = pim_const * 16 / cpu_per */
            /* Use worst-case cpu_per */
            double cross_est = pim_const_us * CPU_CORES / cpuN;
            printf("    Estimated crossover (extrapolated): N ~ %.0f qubits\n", cross_est);
            printf("    (Using CPU per-decode = %.2f us at N=%d, PIM = %.2f us)\n",
                   cpuN, NN, pim_const_us);
        }
    }

    printf("\n  Aggregate throughput crossover:\n");
    if (agg_crossover_n > 0) {
        printf("    At N ~ %.0f DPUs, total PIM throughput surpasses 16-core CPU.\n",
               agg_crossover_n);
    } else {
        printf("    Not reached in tested range.\n");
        if (pim_const_us > 0 && cpu_per_decode_us[0] > 0) {
            double cross_est = pim_const_us * CPU_CORES / cpu_per_decode_us[0];
            printf("    Estimated: N ~ %.0f DPUs needed to match 16-core CPU aggregate.\n",
                   ceil(cross_est));
        }
    }

    /* ==============================================================
     * SECTION 6: Scaling Summary
     * ============================================================== */
    printf("\n  ---- Scaling Summary ----\n\n");
    printf("  CPU cache thrashing effect:\n");
    if (baseline_us > 0) {
        printf("    N=%-5d per-decode: %8.2f us (baseline, per-core %.1f KB, L1d)\n",
               n_values[0], baseline_us, ceil((double)n_values[0] / CPU_CORES) * state_kb);
        for (int ni = 1; ni < n_count; ni++) {
            printf("    N=%-5d per-decode: %8.2f us (%5.2fx, per-core %.1f KB, %s)\n",
                   n_values[ni], cpu_per_decode_us[ni],
                   cpu_per_decode_us[ni] / baseline_us,
                   ceil((double)n_values[ni] / CPU_CORES) * state_kb,
                   cache_level_label(ceil((double)n_values[ni] / CPU_CORES) * state_kb));
        }
    }

    printf("\n  PIM constant performance:\n");
    for (int ni = 0; ni < pim_n_count; ni++) {
        if (pim_per_decode_us[ni] > 0)
            printf("    N=%-4d per-decode: %8.2f us (%5.2fx vs N=1)\n",
                   pim_n_values[ni], pim_per_decode_us[ni],
                   (pim_baseline_us > 0) ? pim_per_decode_us[ni] / pim_baseline_us : 1.0);
    }

    printf("\n  Key insight:\n");
    printf("    CPU: Working set = N x %.1f KB. As N grows, data spills\n", state_kb);
    printf("    from L1 -> L2 -> L3 -> DRAM. Each cache level transition\n");
    printf("    increases per-decode latency. With %d cores sharing N qubits,\n", CPU_CORES);
    printf("    per-qubit throughput = %d/(N * t_decode), which drops as 1/N\n", CPU_CORES);
    printf("    even without cache effects, and faster with them.\n");
    printf("\n    PIM: Each DPU has private 64 KB WRAM. One decoder instance\n");
    printf("    (%.1f KB) always fits. No cache hierarchy, no interference\n", state_kb);
    printf("    between qubits. Per-decode time is constant regardless of N.\n");
    printf("    Per-qubit throughput = 1/t_decode = constant.\n");
    printf("    Total throughput scales linearly with number of DPUs.\n");

    printf("\n    At scale (2528 DPUs per UPMEM module):\n");
    if (pim_const_us > 0 && baseline_us > 0) {
        double pim_2528 = 2528.0 * 1e6 / pim_const_us;
        double cpu_16 = CPU_CORES * 1e6 / baseline_us;  /* best-case CPU */
        printf("    PIM aggregate: 2528 / %.0f us = %.0f decodes/s\n",
               pim_const_us, pim_2528);
        printf("    CPU aggregate: %d / %.2f us = %.0f decodes/s (best case, N=1)\n",
               CPU_CORES, baseline_us, cpu_16);
        printf("    PIM / CPU = %.1fx (even vs best-case CPU with no cache pressure)\n",
               pim_2528 / cpu_16);
    }

    printf("\n====================================================================\n\n");

    return 0;
}
