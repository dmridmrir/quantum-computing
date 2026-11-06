/* task_noopt.c — Non-optimized BP
 * - Convergence-based iteration (max 20, early termination)
 * - Alpha scaling via multiplication: (x * 160) >> 8
 * - PCM precomputed (DPU hardware constraint, not BP optimization)
 */
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>
#include "../support/common.h"
#include "../support/bb144_pcm.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
BARRIER_INIT(my_barrier, NR_TASKLETS);
typedef int16_t fp_t;

static fp_t v2c[NR_TASKLETS][BB_NNZ];
static fp_t c2v[NR_TASKLETS][BB_NNZ];
static fp_t post_arr[NR_TASKLETS][BB_NV];
static uint8_t syn_arr[NR_TASKLETS][BB_NC];

static inline fp_t alpha_scale(fp_t x) {
    return (fp_t)(((int32_t)x * FP_ALPHA) >> 8);
}
static inline fp_t fp_abs(fp_t x) { return x < 0 ? -x : x; }
static inline fp_t fp_sat(int32_t x) {
    return (fp_t)(x > FP_MAX ? FP_MAX : (x < -FP_MAX ? -FP_MAX : x));
}

#define MAX_ITER 20

static void decode(uint32_t tid, fp_t prior) {
    uint32_t chk, var, iter, j;
    for (j = 0; j < BB_NNZ; j++) { v2c[tid][j] = prior; c2v[tid][j] = 0; }
    for (iter = 0; iter < MAX_ITER; iter++) {
        for (chk = 0; chk < BB_NC; chk++) {
            uint32_t b = chk * BB_CHK_DEG;
            fp_t m[BB_CHK_DEG];
            for (j = 0; j < BB_CHK_DEG; j++) m[j] = v2c[tid][b + j];
            int sign = syn_arr[tid][chk] ? 1 : 0;
            for (j = 0; j < BB_CHK_DEG; j++) if (m[j] < 0) sign ^= 1;
            fp_t a[BB_CHK_DEG];
            for (j = 0; j < BB_CHK_DEG; j++) a[j] = fp_abs(m[j]);
            fp_t mn1 = FP_MAX, mn2 = FP_MAX; uint32_t mp = 0;
            for (j = 0; j < BB_CHK_DEG; j++) {
                if (a[j] <= mn1) { mn2 = mn1; mn1 = a[j]; mp = j; }
                else if (a[j] < mn2) mn2 = a[j];
            }
            for (j = 0; j < BB_CHK_DEG; j++) {
                int s = sign ^ (m[j] < 0 ? 1 : 0);
                fp_t mn = (j == mp) ? mn2 : mn1;
                fp_t cv = alpha_scale(mn);
                if (s) cv = -cv;
                c2v[tid][b + j] = cv;
            }
        }
        for (var = 0; var < BB_NV; var++) {
            uint16_t i0 = bb_var_map[var][0], i1 = bb_var_map[var][1], i2 = bb_var_map[var][2];
            fp_t c0 = c2v[tid][i0], c1 = c2v[tid][i1], c2_ = c2v[tid][i2];
            post_arr[tid][var] = fp_sat((int32_t)prior + c0 + c1 + c2_);
            v2c[tid][i0] = fp_sat((int32_t)prior + c1 + c2_);
            v2c[tid][i1] = fp_sat((int32_t)prior + c0 + c2_);
            v2c[tid][i2] = fp_sat((int32_t)prior + c0 + c1);
        }
        uint8_t converged = 1;
        for (chk = 0; chk < BB_NC; chk++) {
            uint8_t s = 0;
            for (j = 0; j < BB_CHK_DEG; j++)
                if (post_arr[tid][bb_chk_vars[chk][j]] <= 0) s ^= 1;
            if (s != syn_arr[tid][chk]) { converged = 0; break; }
        }
        if (converged) break;
    }
}

extern int main_kernel1(void);
int (*kernels[nr_kernels])(void) = {main_kernel1};
int main(void) { return kernels[DPU_INPUT_ARGUMENTS.kernel](); }

int main_kernel1(void) {
    uint32_t tid = me();
    if (tid == 0) mem_reset();
    barrier_wait(&my_barrier);
    __mram_ptr uint8_t *heap = (__mram_ptr uint8_t *)DPU_MRAM_HEAP_POINTER;
    uint32_t slot_size = ((BB_NC + 7) / 8) * 8;
    mram_read(heap + tid * slot_size, syn_arr[tid], slot_size);
    decode(tid, 889);
    uint32_t input_area = NR_TASKLETS * slot_size;
    __mram_ptr uint8_t *out_ptr = heap + input_area + tid * (BB_NV * 2);
    mram_write(post_arr[tid], out_ptr, BB_NV * 2);
    barrier_wait(&my_barrier);
    return 0;
}
