#!/bin/bash
# run_bp_increase_layered_phase2.sh — CPU benchmark + LER + combine into the
# final table ONLY (layered schedule). No uPIMulator, no DPU simulation here.
#
# Takes phase1's two output files (layered_cycles_raw.txt, layered_meta.txt)
# as input -- these can come from a completely different machine than the one
# running this script, since phase1's cycle counts are host-independent. This
# script's own CPU throughput numbers ARE host-dependent (real wall-clock
# timing), so run this on the machine you actually want "CPU throughput" to
# represent (e.g. the EC2 instance).
#
# 실행 방법:
#   ./run_bp_increase_layered_phase2.sh [cycles_file] [meta_file]
#   기본값: ./layered_cycles_raw.txt, ./layered_meta.txt (phase1과 같은 디렉터리에서
#   바로 이어 돌릴 때) -- 다른 기계에서 phase1을 돌렸다면 그 두 파일을 scp로
#   옮겨온 뒤 경로를 인자로 넘기면 됩니다.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CYCLES_FILE="${1:-$SCRIPT_DIR/layered_cycles_raw.txt}"
META_FILE="${2:-$SCRIPT_DIR/layered_meta.txt}"
SUPPORT="$SCRIPT_DIR/uPIMulator/benchmark/BPMS/support"
SEED_BASE=${SHARED_SEED:-$(date +%s)}

if [ ! -f "$CYCLES_FILE" ]; then
    echo "cycles file not found: $CYCLES_FILE" >&2
    echo "run run_bp_increase_layered_phase1.sh first (on any machine) and copy its output here." >&2
    exit 1
fi

TOTAL_DPUS=2560
if [ -f "$META_FILE" ]; then
    # shellcheck disable=SC1090
    source "$META_FILE"
else
    echo "WARNING: meta file not found ($META_FILE), assuming TOTAL_DPUS=$TOTAL_DPUS" >&2
fi

echo ""
echo "================================================================================"
echo "  [Phase 2 only] BP CPU benchmark + LER + combine (LAYERED / Gauss-Seidel)"
echo "  cycles file: $CYCLES_FILE"
echo "  TOTAL_DPUS : $TOTAL_DPUS (from phase1's simulation, not a linear projection)"
echo "================================================================================"
echo ""

cat > /tmp/bp_increase_layered_final.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "bb144_pcm.h"

#define FP_MAX 32000
#define MAX_ENTRIES 100
typedef int16_t fp_t;
static volatile uint64_t benchmark_sink;

static const uint8_t check_colors[3][24] = {
    {1,4,6,9,14,17,19,22,24,27,32,35,37,40,42,45,50,53,55,58,60,63,68,71},
    {0,3,8,11,13,16,18,21,26,29,31,34,36,39,44,47,49,52,54,57,62,65,67,70},
    {2,5,7,10,12,15,20,23,25,28,30,33,38,41,43,46,48,51,56,59,61,64,66,69}
};
static const uint8_t color_order[3] = {1,2,0};

/* bb144_pcm.h is the single syndrome-PCM source used by both CPU and DPU.
 * For this BB code H=[A^T|B^T]; the same-Pauli stabilizer partner is
 * H_partner=[B|A]. */
static uint8_t syndrome_pcm[BB_NC][BB_NV];
static uint8_t stabilizer_pcm[BB_NC][BB_NV];
static void build_css_matrices(void) {
    memset(syndrome_pcm, 0, sizeof(syndrome_pcm));
    memset(stabilizer_pcm, 0, sizeof(stabilizer_pcm));
    for (int c = 0; c < BB_NC; c++)
        for (int j = 0; j < BB_CHK_DEG; j++)
            syndrome_pcm[c][bb_chk_vars[c][j]] = 1;
    for (int r = 0; r < BB_NC; r++) {
        for (int c = 0; c < BB_NC; c++) {
            stabilizer_pcm[r][c] = syndrome_pcm[c][BB_NC + r];
            stabilizer_pcm[r][BB_NC + c] = syndrome_pcm[c][r];
        }
    }
}
static int is_stabilizer(const uint8_t *res) {
    uint8_t mat[BB_NC][BB_NV];
    memcpy(mat, stabilizer_pcm, sizeof(mat));
    uint8_t tgt[BB_NV]; memcpy(tgt, res, BB_NV); int pr = 0;
    for (int col = 0; col < BB_NV && pr < BB_NC; col++) {
        int f = -1;
        for (int r = pr; r < BB_NC; r++) if (mat[r][col]) { f = r; break; }
        if (f < 0) continue;
        if (f != pr)
            for (int c = 0; c < BB_NV; c++) { uint8_t t=mat[pr][c]; mat[pr][c]=mat[f][c]; mat[f][c]=t; }
        if (tgt[col])
            for (int c = 0; c < BB_NV; c++) tgt[c] ^= mat[pr][c];
        for (int r = 0; r < BB_NC; r++)
            if (r != pr && mat[r][col])
                for (int c = 0; c < BB_NV; c++) mat[r][c] ^= mat[pr][c];
        pr++;
    }
    for (int i = 0; i < BB_NV; i++) if (tgt[i]) return 0;
    return 1;
}
static int check_conv(const uint8_t *syn, const fp_t *post) {
    for (int chk = 0; chk < BB_NC; chk++) {
        uint8_t s = 0;
        for (int j = 0; j < BB_CHK_DEG; j++) if (post[bb_chk_vars[chk][j]] <= 0) s ^= 1;
        if (s != syn[chk]) return 0;
    }
    return 1;
}

/* Cooperative v1-wide task_layered.c: three-color order 1,2,0, int32
   c2v/lambda, unsaturated internal state, alpha=6/8=0.75, and no damping. */
static inline int32_t ascale32(int32_t x){ return ((x<<1)+x) >> 2; }
static inline int32_t iabs32(int32_t x){ return x<0?-x:x; }
static inline int32_t sat32(int32_t x){ return x>FP_MAX?FP_MAX:(x<-FP_MAX?-FP_MAX:x); }

static void bp_decode_layered(const uint8_t *syn, fp_t prior_in, int n_iter, fp_t *post) {
    int32_t prior = prior_in;
    int32_t c2v[BB_NNZ];
    int32_t lambda[BB_NV];
    for (int v = 0; v < BB_NV; v++) lambda[v] = prior;
    for (int j = 0; j < BB_NNZ; j++) c2v[j] = 0;

    for (int iter = 0; iter < n_iter; iter++) {
        for (int cp = 0; cp < 3; cp++) {
          int color = color_order[cp];
          for (int pos = 0; pos < 24; pos++) {
            int chk = check_colors[color][pos];
            int b = chk*BB_CHK_DEG;
            int32_t q[BB_CHK_DEG], a[BB_CHK_DEG];
            int sign = syn[chk] ? 1 : 0;
            for (int j = 0; j < BB_CHK_DEG; j++) {
                uint16_t v = bb_chk_vars[chk][j];
                int32_t qv = lambda[v] - c2v[b+j];
                q[j] = qv;
                if (qv < 0) sign ^= 1;
                a[j] = iabs32(qv);
            }
            int32_t mn1 = a[0], mn2 = a[1]; int mp = 0;
            if (mn2 <= mn1) {
                int32_t tmp = mn1;
                mn1 = mn2;
                mn2 = tmp;
                mp = 1;
            }
            for (int j = 2; j < BB_CHK_DEG; j++) {
                if (a[j] <= mn1) { mn2=mn1; mn1=a[j]; mp=j; }
                else if (a[j] < mn2) mn2 = a[j];
            }
            for (int j = 0; j < BB_CHK_DEG; j++) {
                uint16_t v = bb_chk_vars[chk][j];
                int s = sign ^ (q[j] < 0 ? 1 : 0);
                int32_t mn = (j==mp) ? mn2 : mn1;
                int32_t cv = ascale32(mn);
                if (s) cv = -cv;
                c2v[b+j] = cv;
                lambda[v] = q[j] + cv;
            }
          }
        }
    }
    for (int v = 0; v < BB_NV; v++) post[v] = (fp_t)sat32(lambda[v]);
}

static double now_us(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e6 + ts.tv_nsec/1e3;
}
static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}
static void fmt(double v, char *b, int sz) {
    if (v >= 1e6) snprintf(b, sz, "%.2f M/s", v/1e6);
    else if (v >= 1e3) snprintf(b, sz, "%.1f K/s", v/1e3);
    else snprintf(b, sz, "%.0f /s", v);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "Usage: %s <cycles_file> <seed_base> <total_dpus>\n", argv[0]); return 1; }
    build_css_matrices();
    unsigned int seed_base = (unsigned int)atol(argv[2]);
    double total_dpus = atof(argv[3]);

    FILE *cf = fopen(argv[1], "r");
    if (!cf) { perror("cycles"); return 1; }
    double  p_arr[MAX_ENTRIES];
    int     prior_arr[MAX_ENTRIES], iter_arr[MAX_ENTRIES];
    double  cyc_arr[MAX_ENTRIES];
    char    status[MAX_ENTRIES][8];
    int n_entries = 0;
    while (n_entries < MAX_ENTRIES &&
           fscanf(cf, "%lf %d %d %lf %7s",
                  &p_arr[n_entries], &prior_arr[n_entries],
                  &iter_arr[n_entries], &cyc_arr[n_entries],
                  status[n_entries]) == 5)
        n_entries++;
    fclose(cf);

    /* CPU 정보 */
    FILE *fi = fopen("/proc/cpuinfo", "r");
    char cpu_model[256] = "Unknown"; int n_phys = 0, n_logical = 0;
    if (fi) {
        char line[512];
        while (fgets(line, sizeof(line), fi)) {
            if (strncmp(line, "model name", 10) == 0 && cpu_model[0] == 'U') {
                char *p = strchr(line, ':');
                if (p) { p += 2; p[strlen(p)-1] = 0; strncpy(cpu_model, p, 255); }
            }
            if (strncmp(line, "processor", 9) == 0) n_logical++;
            if (strncmp(line, "cpu cores", 9) == 0) {
                char *p = strchr(line, ':'); if (p) n_phys = atoi(p+2);
            }
        }
        fclose(fi);
    }
    int n_cores = n_phys > 0 ? n_phys : n_logical;
    int K = 1; int tasklets_per_decoder = 11; double DPU_MHZ = 350.0;
    int n_timing  = 50000;   /* throughput 측정: 고정 횟수 */
    int n_warmup  = 2000;
    int n_ler      = 10000000; /* LER 측정 고정 시행 횟수 (10M) */

    printf("\n================================================================================\n");
    printf("  BP Throughput Benchmark: CPU vs DPU — iter 1~10 sweep (LAYERED / Gauss-Seidel)\n");
    printf("================================================================================\n\n");
    printf("  Code         : [[144,12,12]] Bivariate Bicycle QLDPC\n");
    printf("  Decoder      : Horizontal layered v1-wide NMS BP (int32 internal state, final int16 clamp)\n");
    printf("  Alpha scaling: shift+add: ((x<<1)+x)>>2 = x*0.75\n");
    printf("  BP iterations: fixed (1~10, no early termination)\n");
    printf("  Throughput   : %d trials per (p,iter)\n", n_timing);
    printf("  LER          : fixed %d trials per (p,iter), Wilson 95%% CI\n", n_ler);
    printf("  CI           : Wilson 95%% confidence interval\n");
    printf("  Verified     : DPU output = CPU output (byte-exact, %d tasklets -> 1 syndrome) -- checked in phase1\n", tasklets_per_decoder);
    printf("\n");
    printf("  CPU          : %s (%d cores)  [measured on THIS machine, running phase2]\n", cpu_model, n_cores);
    printf("  DPU          : UPMEM PIM @ %.0f MHz, %d tasklets/decoder, K=%d decoder/DPU\n", DPU_MHZ, tasklets_per_decoder, K);
    printf("  DPU cycles   : from phase1's run (may have run on a different machine --\n");
    printf("                 cycle counts are host-independent, see script header)\n");
    printf("  '%.0f DPU'    : %.0f DPUs actually simulated in phase1 (not a linear projection)\n", total_dpus, total_dpus);
    /* ========== 저장 배열 ========== */
    double cpu_us_arr[MAX_ENTRIES], e2e_arr[MAX_ENTRIES], ratio_arr[MAX_ENTRIES];
    char sc1_arr[MAX_ENTRIES][16], sc16_arr[MAX_ENTRIES][16];
    char sd1_arr[MAX_ENTRIES][16], sd2560_arr[MAX_ENTRIES][16];
    char ler_str_arr[MAX_ENTRIES][24], ci_str_arr[MAX_ENTRIES][32];
    int  valid[MAX_ENTRIES];

    /* ========== Phase A: Throughput 측정 (결과는 배열에만 저장) ========== */
    for (int pi = 0; pi < n_entries; pi++) {
        valid[pi] = (strcmp(status[pi], "PASS") == 0);
        cpu_us_arr[pi] = 0; e2e_arr[pi] = 0; ratio_arr[pi] = 0;
        strcpy(sc1_arr[pi],"-"); strcpy(sc16_arr[pi],"-");
        strcpy(sd1_arr[pi],"-"); strcpy(sd2560_arr[pi],"-");
        strcpy(ler_str_arr[pi],"-"); strcpy(ci_str_arr[pi],"-");
        if (!valid[pi]) continue;

        double p = p_arr[pi];
        fp_t prior = (fp_t)prior_arr[pi];
        int n_iter = iter_arr[pi];
        double cpu_us, cpu1;
        {fp_t lpost[BB_NV]; uint8_t lsyn[BB_NC],lerr[BB_NV];
        for (int w = 0; w < n_warmup; w++) {
            unsigned int seed = seed_base + 77777 + w;
            memset(lerr, 0, BB_NV);
            for (int i = 0; i < BB_NV; i++)
                if ((double)rand_r(&seed)/RAND_MAX < p) lerr[i] = 1;
            for (int chk = 0; chk < BB_NC; chk++) {
                uint8_t s = 0;
                for (int j = 0; j < BB_CHK_DEG; j++) s ^= lerr[bb_chk_vars[chk][j]];
                lsyn[chk] = s;
            }
            bp_decode_layered(lsyn, prior, n_iter, lpost);
        }
        double t1c = now_us();
        uint64_t checksum = 0;
        for (int t = 0; t < n_timing; t++) {
            unsigned int seed = seed_base + t*7;
            memset(lerr, 0, BB_NV);
            for (int i = 0; i < BB_NV; i++)
                if ((double)rand_r(&seed)/RAND_MAX < p) lerr[i] = 1;
            for (int chk = 0; chk < BB_NC; chk++) {
                uint8_t s = 0;
                for (int j = 0; j < BB_CHK_DEG; j++) s ^= lerr[bb_chk_vars[chk][j]];
                lsyn[chk] = s;
            }
            bp_decode_layered(lsyn, prior, n_iter, lpost);
            checksum += (uint16_t)lpost[t % BB_NV];
        }
        cpu_us = (now_us()-t1c) / n_timing * 2.0;
        benchmark_sink ^= checksum;
        cpu1   = 1e6 / cpu_us;}
        double ts_mp = now_sec();
        uint64_t parallel_checksum = 0;
        #pragma omp parallel num_threads(n_cores) reduction(+:parallel_checksum)
        {fp_t post[BB_NV]; uint8_t syn[BB_NC],error[BB_NV];
        #pragma omp for schedule(static)
        for (int t = 0; t < n_timing; t++) {
            unsigned int seed = seed_base + t*137;
            memset(error, 0, BB_NV);
            for (int i = 0; i < BB_NV; i++)
                if ((double)rand_r(&seed)/RAND_MAX < p) error[i] = 1;
            for (int chk = 0; chk < BB_NC; chk++) {
                uint8_t s = 0;
                for (int j = 0; j < BB_CHK_DEG; j++) s ^= error[bb_chk_vars[chk][j]];
                syn[chk] = s;
            }
            bp_decode_layered(syn, prior, n_iter, post);
            parallel_checksum += (uint16_t)post[t % BB_NV];
        }}
        double parallel_elapsed = now_sec() - ts_mp;
        benchmark_sink ^= parallel_checksum;
        /* X and Z are serialized, matching CPU us/dec and DPU t_batch. */
        double cpu_all    = (double)n_timing / (2.0 * parallel_elapsed);
        double dpu_us     = cyc_arr[pi] / DPU_MHZ;
        double t_batch    = 2.0 * dpu_us;
        double e2e_per_us = t_batch / K;
        double dpu1       = K * 1e6 / t_batch;
        double dpu_total  = dpu1 * total_dpus;
        double ratio      = dpu_total / cpu_all;

        cpu_us_arr[pi] = cpu_us; e2e_arr[pi] = e2e_per_us; ratio_arr[pi] = ratio;
        fmt(cpu1,    sc1_arr[pi],    16); fmt(cpu_all, sc16_arr[pi],   16);
        fmt(dpu1,    sd1_arr[pi],    16); fmt(dpu_total, sd2560_arr[pi], 16);
    }

    /* ========== Phase B: LER 측정 (결과는 배열에만 저장) ========== */
    printf("\n[Phase B] LER measurement (%d trials each, X+Z)...\n", n_ler);
    fflush(stdout);
    for (int pi = 0; pi < n_entries; pi++) {
        if (!valid[pi]) continue;
        double p = p_arr[pi];
        fp_t prior = (fp_t)prior_arr[pi];
        int n_iter = iter_arr[pi];
        printf("  [%2d/%2d] p=%.3f iter=%2d ... ", pi+1, n_entries, p, n_iter);
        fflush(stdout);
        int n_fail = 0;
        #pragma omp parallel num_threads(n_cores) reduction(+:n_fail)
        {fp_t post[BB_NV]; uint8_t syn[BB_NC],error[BB_NV];
        #pragma omp for schedule(static)
        for (int t = 0; t < n_ler; t++) {
            unsigned int seed_x = seed_base + 9999999 + t*137;
            memset(error, 0, BB_NV);
            for (int i = 0; i < BB_NV; i++)
                if ((double)rand_r(&seed_x)/RAND_MAX < p) error[i] = 1;
            for (int chk = 0; chk < BB_NC; chk++) {
                uint8_t s = 0;
                for (int j = 0; j < BB_CHK_DEG; j++) s ^= error[bb_chk_vars[chk][j]];
                syn[chk] = s;
            }
            bp_decode_layered(syn, prior, n_iter, post);
            uint8_t corr[BB_NV], res[BB_NV]; int rw = 0;
            for (int i = 0; i < BB_NV; i++) {
                corr[i] = (post[i] <= 0) ? 1 : 0;
                res[i] = error[i] ^ corr[i]; rw += res[i];
            }
            int x_fail = 0;
            if (rw > 0) {
                if (!check_conv(syn, post)) x_fail = 1;
                else if (!is_stabilizer(res)) x_fail = 1;
            }
            unsigned int seed_z = seed_base + 7777777 + t*137;
            memset(error, 0, BB_NV);
            for (int i = 0; i < BB_NV; i++)
                if ((double)rand_r(&seed_z)/RAND_MAX < p) error[i] = 1;
            for (int chk = 0; chk < BB_NC; chk++) {
                uint8_t s = 0;
                for (int j = 0; j < BB_CHK_DEG; j++) s ^= error[bb_chk_vars[chk][j]];
                syn[chk] = s;
            }
            bp_decode_layered(syn, prior, n_iter, post);
            rw = 0;
            for (int i = 0; i < BB_NV; i++) {
                corr[i] = (post[i] <= 0) ? 1 : 0;
                res[i] = error[i] ^ corr[i]; rw += res[i];
            }
            int z_fail = 0;
            if (rw > 0) {
                if (!check_conv(syn, post)) z_fail = 1;
                else if (!is_stabilizer(res)) z_fail = 1;
            }
            if (x_fail || z_fail) n_fail++;
        }}

        printf("LER=%.4e\n", (double)n_fail/n_ler*100.0);
        fflush(stdout);

        double zz = 1.96, kd = (double)n_fail, nd = (double)n_ler;
        double p_hat = kd / nd;
        double center = (kd + zz*zz/2.0) / (nd + zz*zz);
        double half   = zz * sqrt(kd*(nd-kd)/nd + zz*zz/4.0) / (nd + zz*zz);
        double ci_lo  = (center - half) * 100.0;
        double ci_hi  = (center + half) * 100.0;
        if (ci_lo < 0.0) ci_lo = 0.0;
        if (n_fail == 0) {
            snprintf(ler_str_arr[pi], 24, "< %.2e", 3.0/nd*100.0);
            snprintf(ci_str_arr[pi],  32, "[0, %.2e]", 3.0/nd*100.0);
        } else {
            snprintf(ler_str_arr[pi], 24, "%.4e", p_hat*100.0);
            snprintf(ci_str_arr[pi],  32, "[%.2e, %.2e]", ci_lo, ci_hi);
        }
    }

    /* ========== 통합 표 ========== */
    fflush(stdout);
    #define SEP1 "  -------+------+--------+------------+------------+------------+--------------+--------------+--------------+--------------+---------+--------------\n"
    printf("\n================================================================================\n");
    printf("  Throughput + LER Summary (X+Z, 10M trials) — LAYERED\n");
    printf("================================================================================\n\n");
    char dpu_col_label[16];
    snprintf(dpu_col_label, sizeof(dpu_col_label), "%.0f DPU(real)", total_dpus);
    printf("  %6s | %4s | %6s | %10s | %10s | %10s | %12s | %12s | %12s | %12s | %7s | %12s\n",
           "p","iter","prior","DPU cycles","CPU us/dec","E2E us/dec",
           "CPU 1-core","CPU 16-core","1 DPU",dpu_col_label,"ratio","LER(%%)");
    printf(SEP1);
    int prev_iter = -1;
    for (int pi = 0; pi < n_entries; pi++) {
        if (prev_iter != -1 && iter_arr[pi] != prev_iter) printf(SEP1);
        prev_iter = iter_arr[pi];
        if (!valid[pi])
            printf("  %6.3f | %4d | %6d | %10s | %10s | %10s | %12s | %12s | %12s | %12s | %7s | %12s\n",
                   p_arr[pi],iter_arr[pi],prior_arr[pi],"FAIL","-","-","-","-","-","-","-","-");
        else
            printf("  %6.3f | %4d | %6d | %10.0f | %10.3f | %10.3f | %12s | %12s | %12s | %12s | %6.1fx | %12s\n",
                   p_arr[pi],iter_arr[pi],prior_arr[pi],cyc_arr[pi],
                   cpu_us_arr[pi],e2e_arr[pi],
                   sc1_arr[pi],sc16_arr[pi],sd1_arr[pi],sd2560_arr[pi],
                   ratio_arr[pi],ler_str_arr[pi]);
    }
    printf(SEP1);

    /* ========== 상세 LER 표 (Wilson 95%% CI) ========== */
    #define SEP2 "  -------+------+------------+------------------------+-----------\n"
    printf("\n================================================================================\n");
    printf("  Logical Error Rate detail (X+Z, fixed 10M trials, Wilson 95%% CI) — LAYERED\n");
    printf("================================================================================\n\n");
    printf("  %6s | %4s | %10s | %22s | %9s\n",
           "p","iter","LER(%%)","95%% CI [lo, hi]","n_trials");
    printf(SEP2);
    prev_iter = -1;
    for (int pi = 0; pi < n_entries; pi++) {
        if (prev_iter != -1 && iter_arr[pi] != prev_iter) printf(SEP2);
        prev_iter = iter_arr[pi];
        if (!valid[pi])
            printf("  %6.3f | %4d | %10s | %22s | %9s\n",
                   p_arr[pi],iter_arr[pi],"FAIL","-","-");
        else
            printf("  %6.3f | %4d | %10s | %22s | %9d\n",
                   p_arr[pi],iter_arr[pi],ler_str_arr[pi],ci_str_arr[pi],n_ler);
    }
    printf(SEP2);
    printf("\n================================================================================\n\n");
    return benchmark_sink == UINT64_C(0xdeadbeef) ? 1 : 0;
}
CEOF

gcc -O2 -fopenmp -I"$SUPPORT" -o /tmp/bp_increase_layered_final /tmp/bp_increase_layered_final.c -lm
/tmp/bp_increase_layered_final "$CYCLES_FILE" "$SEED_BASE" "$TOTAL_DPUS"
