/* Cooperative v1 three-color horizontal-layered Min-Sum BP.
 *
 * V1-wide arithmetic: int32 C2V/lambda, unsaturated q, alpha=3/4, and no
 * damping.  Internal messages stay wide; only the exported int16 posterior
 * is clamped.  This avoids the per-check +/-FP_MAX saturation limit cycle.
 * Eleven tasklets cooperate on one syndrome. Checks of the same color have
 * disjoint variables, so they can update the shared posterior without races;
 * a barrier separates colors in the fixed order 1,2,0.
 */
#include <stdint.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include "../support/common.h"
#include "../support/bb144_pcm.h"
#include "../support/bb144_colors.h"

#if NR_TASKLETS != 11
#error "cooperative v1 requires NR_TASKLETS=11"
#endif

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
BARRIER_INIT(all_tasklets, NR_TASKLETS);
typedef int16_t fp_t;

__dma_aligned int32_t shared_c2v[BB_NNZ];
__dma_aligned int32_t shared_lambda[BB_NV];
__dma_aligned fp_t shared_post[BB_NV];
__dma_aligned uint8_t shared_syn[BB_NC];

static inline int32_t alpha_scale(int32_t x) {
    return ((x << 1) + x) >> 2;
}
static inline int32_t i32_abs(int32_t x) { return x < 0 ? -x : x; }
static inline int32_t sat32(int32_t x) {
    return x > FP_MAX ? FP_MAX : (x < -FP_MAX ? -FP_MAX : x);
}

static inline void update_check(uint32_t chk) {
    uint32_t b = chk * BB_CHK_DEG;
    int32_t q[BB_CHK_DEG], magnitude[BB_CHK_DEG];
    int sign = shared_syn[chk] ? 1 : 0;

    for (uint32_t j = 0; j < BB_CHK_DEG; j++) {
        uint16_t v = bb_chk_vars[chk][j];
        int32_t qv = shared_lambda[v] - shared_c2v[b + j];
        q[j] = qv;
        if (qv < 0) sign ^= 1;
        magnitude[j] = i32_abs(qv);
    }

    int32_t mn1 = magnitude[0], mn2 = magnitude[1];
    uint32_t min_pos = 0;
    if (mn2 <= mn1) {
        int32_t tmp = mn1;
        mn1 = mn2;
        mn2 = tmp;
        min_pos = 1;
    }
    for (uint32_t j = 2; j < BB_CHK_DEG; j++) {
        if (magnitude[j] <= mn1) {
            mn2 = mn1;
            mn1 = magnitude[j];
            min_pos = j;
        } else if (magnitude[j] < mn2) {
            mn2 = magnitude[j];
        }
    }

    for (uint32_t j = 0; j < BB_CHK_DEG; j++) {
        uint16_t v = bb_chk_vars[chk][j];
        int out_sign = sign ^ (q[j] < 0 ? 1 : 0);
        int32_t magnitude_out = j == min_pos ? mn2 : mn1;
        int32_t cv = alpha_scale(magnitude_out);
        if (out_sign) cv = -cv;
        shared_c2v[b + j] = cv;
        shared_lambda[v] = q[j] + cv;
    }
}

static void decode(uint32_t tid, fp_t prior, uint32_t n_iter) {
    for (uint32_t v = tid; v < BB_NV; v += NR_TASKLETS)
        shared_lambda[v] = prior;
    for (uint32_t edge = tid; edge < BB_NNZ; edge += NR_TASKLETS)
        shared_c2v[edge] = 0;
    barrier_wait(&all_tasklets);

    for (uint32_t iter = 0; iter < n_iter; iter++) {
        for (uint32_t color_position = 0; color_position < BB_NUM_COLORS;
             color_position++) {
            uint32_t color = bb_color_order[color_position];
            for (uint32_t pos = tid; pos < BB_CHECKS_PER_COLOR;
                 pos += NR_TASKLETS)
                update_check(bb_check_colors[color][pos]);
            barrier_wait(&all_tasklets);
        }
    }

    for (uint32_t v = tid; v < BB_NV; v += NR_TASKLETS)
        shared_post[v] = (fp_t)sat32(shared_lambda[v]);
    barrier_wait(&all_tasklets);
}

extern int main_kernel1(void);
int (*kernels[nr_kernels])(void) = {main_kernel1};
int main(void) { return kernels[DPU_INPUT_ARGUMENTS.kernel](); }

int main_kernel1(void) {
    uint32_t tid = me();
    if (tid == 0) mem_reset();
    barrier_wait(&all_tasklets);

    __mram_ptr uint8_t *heap =
        (__mram_ptr uint8_t *)DPU_MRAM_HEAP_POINTER;
    uint32_t slot_size = ((BB_NC + 7) / 8) * 8;
    if (tid == 0) mram_read(heap, shared_syn, slot_size);
    barrier_wait(&all_tasklets);

    decode(tid, 889, 2);

    if (tid == 0)
        mram_write(shared_post, heap + slot_size, BB_NV * 2);
    barrier_wait(&all_tasklets);
    return 0;
}
