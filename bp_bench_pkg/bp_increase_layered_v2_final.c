#define _GNU_SOURCE
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bb144_pcm.h"
#include "bb144_colors.h"

#define FP_MAX 32000
#define MAX_ENTRIES 100
typedef int16_t fp_t;
static volatile uint64_t benchmark_sink;

static inline fp_t sat(int32_t x) {
    return (fp_t)(x > FP_MAX ? FP_MAX : (x < -FP_MAX ? -FP_MAX : x));
}
static inline int32_t iabs32(int32_t x) { return x < 0 ? -x : x; }

static void minima(const int32_t mag[BB_CHK_DEG], int32_t *mn1,
                   int32_t *mn2, uint32_t *min_pos) {
    *mn1 = mag[0]; *mn2 = FP_MAX; *min_pos = 0;
    for (uint32_t j = 1; j < BB_CHK_DEG; j++) {
        if (mag[j] <= *mn1) {
            *mn2 = *mn1; *mn1 = mag[j]; *min_pos = j;
        } else if (mag[j] < *mn2) {
            *mn2 = mag[j];
        }
    }
}

static void update_first(uint32_t chk, const uint8_t *syn, fp_t *c2v,
                         fp_t *post, int final_iter) {
    uint32_t b = chk * BB_CHK_DEG;
    int32_t q[BB_CHK_DEG], mag[BB_CHK_DEG];
    int sign = syn[chk] ? 1 : 0;
    for (uint32_t j = 0; j < BB_CHK_DEG; j++) {
        q[j] = post[bb_chk_vars[chk][j]];
        if (q[j] < 0) sign ^= 1;
        mag[j] = iabs32(q[j]);
    }
    int32_t mn1, mn2; uint32_t min_pos;
    minima(mag, &mn1, &mn2, &min_pos);
    for (uint32_t j = 0; j < BB_CHK_DEG; j++) {
        int out_sign = sign ^ (q[j] < 0 ? 1 : 0);
        fp_t raw = (fp_t)(j == min_pos ? mn2 : mn1);
        if (out_sign) raw = -raw;
        fp_t cv = (fp_t)(raw >> 1);
        if (!final_iter) c2v[b + j] = cv;
        post[bb_chk_vars[chk][j]] = sat(q[j] + cv);
    }
}

static void update_check(uint32_t chk, const uint8_t *syn, fp_t *c2v,
                         fp_t *post, int final_iter) {
    uint32_t b = chk * BB_CHK_DEG;
    int32_t q[BB_CHK_DEG], mag[BB_CHK_DEG];
    int sign = syn[chk] ? 1 : 0;
    for (uint32_t j = 0; j < BB_CHK_DEG; j++) {
        q[j] = sat((int32_t)post[bb_chk_vars[chk][j]] - c2v[b + j]);
        if (q[j] < 0) sign ^= 1;
        mag[j] = iabs32(q[j]);
    }
    int32_t mn1, mn2; uint32_t min_pos;
    minima(mag, &mn1, &mn2, &min_pos);
    for (uint32_t j = 0; j < BB_CHK_DEG; j++) {
        int out_sign = sign ^ (q[j] < 0 ? 1 : 0);
        fp_t raw = (fp_t)(j == min_pos ? mn2 : mn1);
        if (out_sign) raw = -raw;
        int32_t old = c2v[b + j];
        fp_t cv = sat(old + (((int32_t)raw - old) >> 1));
        if (!final_iter) c2v[b + j] = cv;
        post[bb_chk_vars[chk][j]] = sat(q[j] + cv);
    }
}

/* Exact sequential equivalent of task_layered_v2.c: color order 1,2,0,
 * alpha=1, half damping, fixed iteration count. */
static void decode_colored(const uint8_t *syn, fp_t prior, int iterations,
                           fp_t *post) {
    fp_t c2v[BB_NNZ];
    for (uint32_t v = 0; v < BB_NV; v++) post[v] = prior;
    int first_final = iterations == 1;
    for (uint32_t cp = 0; cp < BB_NUM_COLORS; cp++) {
        uint32_t color = bb_color_order[cp];
        for (uint32_t pos = 0; pos < BB_CHECKS_PER_COLOR; pos++)
            update_first(bb_check_colors[color][pos], syn, c2v, post,
                         first_final);
    }
    for (int iter = 1; iter < iterations; iter++) {
        int final_iter = iter + 1 == iterations;
        for (uint32_t cp = 0; cp < BB_NUM_COLORS; cp++) {
            uint32_t color = bb_color_order[cp];
            for (uint32_t pos = 0; pos < BB_CHECKS_PER_COLOR; pos++)
                update_check(bb_check_colors[color][pos], syn, c2v, post,
                             final_iter);
        }
    }
}

static void syndrome_from_error(const uint8_t *error, uint8_t *syn) {
    for (uint32_t chk = 0; chk < BB_NC; chk++) {
        uint8_t s = 0;
        for (uint32_t j = 0; j < BB_CHK_DEG; j++)
            s ^= error[bb_chk_vars[chk][j]];
        syn[chk] = s;
    }
}

static uint8_t hz[BB_NC][BB_NV];
static void build_hz(void) {
    memset(hz, 0, sizeof(hz));
    for (uint32_t c = 0; c < BB_NC; c++)
        for (uint32_t j = 0; j < BB_CHK_DEG; j++)
            hz[c][bb_chk_vars[c][j]] = 1;
}
static int check_conv(const uint8_t *syn, const fp_t *post) {
    for (uint32_t chk = 0; chk < BB_NC; chk++) {
        uint8_t s = 0;
        for (uint32_t j = 0; j < BB_CHK_DEG; j++)
            if (post[bb_chk_vars[chk][j]] <= 0) s ^= 1;
        if (s != syn[chk]) return 0;
    }
    return 1;
}
static int is_stabilizer(const uint8_t *res) {
    uint8_t mat[BB_NC][BB_NV], tgt[BB_NV];
    memcpy(mat, hz, sizeof(mat)); memcpy(tgt, res, sizeof(tgt));
    int pivot_row = 0;
    for (int col = 0; col < BB_NV && pivot_row < BB_NC; col++) {
        int found = -1;
        for (int r = pivot_row; r < BB_NC; r++)
            if (mat[r][col]) { found = r; break; }
        if (found < 0) continue;
        if (found != pivot_row)
            for (int c = 0; c < BB_NV; c++) {
                uint8_t t = mat[pivot_row][c];
                mat[pivot_row][c] = mat[found][c]; mat[found][c] = t;
            }
        if (tgt[col])
            for (int c = 0; c < BB_NV; c++) tgt[c] ^= mat[pivot_row][c];
        for (int r = 0; r < BB_NC; r++)
            if (r != pivot_row && mat[r][col])
                for (int c = 0; c < BB_NV; c++)
                    mat[r][c] ^= mat[pivot_row][c];
        pivot_row++;
    }
    for (int i = 0; i < BB_NV; i++) if (tgt[i]) return 0;
    return 1;
}

static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static void fmt(double value, char *buffer, int size) {
    if (value >= 1e6) snprintf(buffer, size, "%.2f M/s", value / 1e6);
    else if (value >= 1e3) snprintf(buffer, size, "%.1f K/s", value / 1e3);
    else snprintf(buffer, size, "%.0f /s", value);
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s aggregate seed total_dpus timing_trials ler_trials\n", argv[0]);
        return 1;
    }
    const char *input = argv[1];
    unsigned int seed_base = (unsigned int)strtoul(argv[2], NULL, 10);
    double total_dpus = atof(argv[3]);
    int n_timing = atoi(argv[4]), n_ler = atoi(argv[5]);
    if (n_timing <= 0 || n_ler <= 0) return 1;
    build_hz();

    double p[MAX_ENTRIES], cycles[MAX_ENTRIES];
    int prior[MAX_ENTRIES], iterations[MAX_ENTRIES], n = 0;
    FILE *file = fopen(input, "r");
    if (!file) { perror(input); return 1; }
    while (n < MAX_ENTRIES &&
           fscanf(file, "%lf %d %d %lf", &p[n], &prior[n],
                  &iterations[n], &cycles[n]) == 4) n++;
    fclose(file);
    if (n != 50) {
        fprintf(stderr, "expected 50 (p,iter) aggregate rows, got %d\n", n);
        return 1;
    }

    int n_cores = omp_get_num_procs();
    char cpu_model[256] = "Unknown";
    FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
    if (cpuinfo) {
        char line[512];
        while (fgets(line, sizeof(line), cpuinfo)) {
            if (!strncmp(line, "model name", 10)) {
                char *colon = strchr(line, ':');
                if (colon) { colon += 2; colon[strcspn(colon, "\n")] = 0;
                    strncpy(cpu_model, colon, sizeof(cpu_model)-1); }
                break;
            }
        }
        fclose(cpuinfo);
    }

    double cpu_us[MAX_ENTRIES], e2e_us[MAX_ENTRIES], ratio[MAX_ENTRIES];
    char cpu1[MAX_ENTRIES][16], cpuall[MAX_ENTRIES][16];
    char dpu1[MAX_ENTRIES][16], dputotal[MAX_ENTRIES][16];
    char ler_text[MAX_ENTRIES][24], ci_text[MAX_ENTRIES][32];

    printf("\n[Phase A] CPU throughput measurement (%d trials each, X+Z)...\n", n_timing);
    for (int row = 0; row < n; row++) {
        fp_t post[BB_NV]; uint8_t syn[BB_NC], error[BB_NV];
        for (int w = 0; w < 2000; w++) {
            unsigned int seed = seed_base + 77777 + w;
            memset(error, 0, sizeof(error));
            for (int v = 0; v < BB_NV; v++)
                if ((double)rand_r(&seed) / RAND_MAX < p[row]) error[v] = 1;
            syndrome_from_error(error, syn);
            decode_colored(syn, (fp_t)prior[row], iterations[row], post);
        }
        double start_us = now_us();
        uint64_t checksum = 0;
        for (int t = 0; t < n_timing; t++) {
            unsigned int seed = seed_base + t * 7;
            memset(error, 0, sizeof(error));
            for (int v = 0; v < BB_NV; v++)
                if ((double)rand_r(&seed) / RAND_MAX < p[row]) error[v] = 1;
            syndrome_from_error(error, syn);
            decode_colored(syn, (fp_t)prior[row], iterations[row], post);
            checksum += (uint16_t)post[t % BB_NV];
        }
        double component_us = (now_us() - start_us) / n_timing;
        benchmark_sink ^= checksum;
        cpu_us[row] = 2.0 * component_us;
        fmt(1e6 / cpu_us[row], cpu1[row], 16);

        double start_parallel = now_sec();
        uint64_t parallel_checksum = 0;
#pragma omp parallel for schedule(static) num_threads(n_cores) reduction(+:parallel_checksum)
        for (int t = 0; t < n_timing; t++) {
            fp_t local_post[BB_NV]; uint8_t local_syn[BB_NC], local_error[BB_NV];
            unsigned int seed = seed_base + t * 137;
            memset(local_error, 0, sizeof(local_error));
            for (int v = 0; v < BB_NV; v++)
                if ((double)rand_r(&seed) / RAND_MAX < p[row]) local_error[v] = 1;
            syndrome_from_error(local_error, local_syn);
            decode_colored(local_syn, (fp_t)prior[row], iterations[row], local_post);
            parallel_checksum += (uint16_t)local_post[t % BB_NV];
        }
        benchmark_sink ^= parallel_checksum;
        double cpu_all = n_timing / (2.0 * (now_sec() - start_parallel));
        fmt(cpu_all, cpuall[row], 16);

        e2e_us[row] = 2.0 * cycles[row] / 350.0;
        double one_dpu = 1e6 / e2e_us[row];
        double all_dpu = one_dpu * total_dpus;
        ratio[row] = all_dpu / cpu_all;
        fmt(one_dpu, dpu1[row], 16); fmt(all_dpu, dputotal[row], 16);
    }

    printf("\n[Phase B] LER measurement (%d trials each, X+Z)...\n", n_ler);
    for (int row = 0; row < n; row++) {
        long long failures = 0;
#pragma omp parallel for schedule(static) num_threads(n_cores) reduction(+:failures)
        for (int t = 0; t < n_ler; t++) {
            int trial_failed = 0;
            for (int component = 0; component < 2; component++) {
                fp_t post[BB_NV]; uint8_t syn[BB_NC], error[BB_NV], res[BB_NV];
                unsigned int seed = seed_base + (component ? 7777777 : 9999999) + t * 137;
                memset(error, 0, sizeof(error));
                for (int v = 0; v < BB_NV; v++)
                    if ((double)rand_r(&seed) / RAND_MAX < p[row]) error[v] = 1;
                syndrome_from_error(error, syn);
                decode_colored(syn, (fp_t)prior[row], iterations[row], post);
                int residual_weight = 0;
                for (int v = 0; v < BB_NV; v++) {
                    res[v] = error[v] ^ (post[v] <= 0 ? 1 : 0);
                    residual_weight += res[v];
                }
                if (residual_weight &&
                    (!check_conv(syn, post) || !is_stabilizer(res)))
                    trial_failed = 1;
            }
            failures += trial_failed;
        }
        double z = 1.96, kd = failures, nd = n_ler;
        double estimate = kd / nd;
        double center = (kd + z*z/2.0) / (nd + z*z);
        double half = z * sqrt(kd*(nd-kd)/nd + z*z/4.0) / (nd + z*z);
        double lo = fmax(0.0, center-half) * 100.0, hi = (center+half) * 100.0;
        if (failures == 0) {
            snprintf(ler_text[row], 24, "< %.2e", 3.0/nd*100.0);
            snprintf(ci_text[row], 32, "[0, %.2e]", 3.0/nd*100.0);
        } else {
            snprintf(ler_text[row], 24, "%.4e", estimate*100.0);
            snprintf(ci_text[row], 32, "[%.2e, %.2e]", lo, hi);
        }
        printf("  [%d/%d] p=%.3f iter=%d LER=%s%%\n",
               row+1, n, p[row], iterations[row], ler_text[row]);
    }

    char dpu_label[24], cpu_label[24];
    snprintf(dpu_label, sizeof(dpu_label), "%.0f DPU(proj)", total_dpus);
    snprintf(cpu_label, sizeof(cpu_label), "CPU %d-core", n_cores);
    #define SEP1 "  -------+------+--------+------------+------------+------------+--------------+--------------+--------------+----------------+---------+--------------\n"
    printf("\n================================================================================\n");
    printf("  Throughput + LER Summary (X+Z) — COOPERATIVE V2\n");
    printf("  CPU: %s (%d logical cores)\n", cpu_model, n_cores);
    printf("  DPU cycles: mean of verified phase1 samples for each p\n");
    printf("================================================================================\n\n");
    printf("  %6s | %4s | %6s | %10s | %10s | %10s | %12s | %12s | %12s | %14s | %7s | %12s\n",
           "p","iter","prior","DPU cycles","CPU us/dec","E2E us/dec",
           "CPU 1-core",cpu_label,"1 DPU",dpu_label,"ratio","LER(%%)");
    printf(SEP1);
    for (int row = 0; row < n; row++)
        printf("  %6.3f | %4d | %6d | %10.0f | %10.3f | %10.3f | %12s | %12s | %12s | %14s | %6.1fx | %12s\n",
               p[row], iterations[row], prior[row], cycles[row], cpu_us[row],
               e2e_us[row], cpu1[row], cpuall[row], dpu1[row], dputotal[row],
               ratio[row], ler_text[row]);
    printf(SEP1);

    #define SEP2 "  -------+------+------------+------------------------+-----------\n"
    printf("\n================================================================================\n");
    printf("  Logical Error Rate detail (X+Z, Wilson 95%% CI) — COOPERATIVE V2\n");
    printf("================================================================================\n\n");
    printf("  %6s | %4s | %10s | %22s | %9s\n",
           "p","iter","LER(%%)","95%% CI [lo, hi]","n_trials");
    printf(SEP2);
    for (int row = 0; row < n; row++)
        printf("  %6.3f | %4d | %10s | %22s | %9d\n",
               p[row], iterations[row], ler_text[row], ci_text[row], n_ler);
    printf(SEP2);
    return benchmark_sink == UINT64_C(0xdeadbeef) ? 1 : 0;
}
