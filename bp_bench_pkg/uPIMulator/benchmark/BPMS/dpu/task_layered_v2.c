/* Cooperative three-color horizontal-layered Q8 Min-Sum BP. */
#include <stdint.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <atomic_bit.h>
#include "../support/common.h"
#include "../support/bb144_pcm.h"
#include "../support/bb144_colors.h"

#ifndef BP_GROUP_SIZE
#error "BP_GROUP_SIZE must be defined"
#endif
#if NR_TASKLETS % BP_GROUP_SIZE != 0
#error "BP_GROUP_SIZE must divide NR_TASKLETS"
#endif

#define BP_NUM_GROUPS (NR_TASKLETS / BP_GROUP_SIZE)

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
BARRIER_INIT(all_tasklets, NR_TASKLETS);
#if BP_NUM_GROUPS > 1
ATOMIC_BIT_INIT(group_barrier_locks)[BP_NUM_GROUPS];
barrier_t group_barriers[BP_NUM_GROUPS];
#endif

typedef int16_t fp_t;
typedef int32_t check_t;

/* One shared decoder state per concurrent syndrome. */
__dma_aligned fp_t shared_c2v[BP_NUM_GROUPS][BB_NNZ];
__dma_aligned fp_t shared_post[BP_NUM_GROUPS][BB_NV];
__dma_aligned uint8_t shared_syn[BP_NUM_GROUPS][BB_NC];

static inline check_t int_abs(check_t x) { return x < 0 ? -x : x; }
static inline fp_t fp_sat(int32_t x) {
    return (fp_t)(x > FP_MAX ? FP_MAX : (x < -FP_MAX ? -FP_MAX : x));
}

/* Selected CPU-sweep point: alpha=1 and damping=1/2.
 * raw Min-Sum therefore needs no normalization multiply. */
static inline fp_t damp_half(fp_t old, fp_t raw) {
    return fp_sat((int32_t)old + (((int32_t)raw - (int32_t)old) >> 1));
}

static inline void group_wait(uint32_t group) {
#if BP_NUM_GROUPS == 1
    (void)group;
    barrier_wait(&all_tasklets);
#else
    barrier_wait(&group_barriers[group]);
#endif
}

static inline void update_check(uint32_t chk, fp_t *c2v, fp_t *post,
                                const uint8_t *syn, int final_iter) {
    uint32_t b = chk * BB_CHK_DEG;
    check_t q[BB_CHK_DEG];
    check_t magnitude[BB_CHK_DEG];
    int sign = syn[chk] ? 1 : 0;

    for (uint32_t j = 0; j < BB_CHK_DEG; j++) {
        uint32_t v = bb_chk_vars[chk][j];
        q[j] = fp_sat((int32_t)post[v] - (int32_t)c2v[b + j]);
        if (q[j] < 0) sign ^= 1;
        magnitude[j] = int_abs(q[j]);
    }

    check_t mn1 = magnitude[0], mn2 = FP_MAX;
    uint32_t min_pos = 0;
    for (uint32_t j = 1; j < BB_CHK_DEG; j++) {
        if (magnitude[j] <= mn1) {
            mn2 = mn1;
            mn1 = magnitude[j];
            min_pos = j;
        } else if (magnitude[j] < mn2) {
            mn2 = magnitude[j];
        }
    }

    for (uint32_t j = 0; j < BB_CHK_DEG; j++) {
        int out_sign = sign ^ (q[j] < 0 ? 1 : 0);
        fp_t raw = (fp_t)((j == min_pos) ? mn2 : mn1);
        if (out_sign) raw = -raw;
        fp_t cv = damp_half(c2v[b + j], raw);
        uint32_t v = bb_chk_vars[chk][j];
        if (!final_iter) c2v[b + j] = cv;
        post[v] = fp_sat((int32_t)q[j] + (int32_t)cv);
    }
}

/* Exact first-sweep specialization: old C2V is zero, hence q=posterior and
 * half damping is raw>>1. The sweep writes every C2V edge before it can be
 * read by a later sweep, so a separate 432-entry zero initialization is dead. */
static inline void update_check_first(uint32_t chk, fp_t *c2v, fp_t *post,
                                      const uint8_t *syn, int final_iter) {
    uint32_t b = chk * BB_CHK_DEG;
    check_t q[BB_CHK_DEG];
    check_t magnitude[BB_CHK_DEG];
    int sign = syn[chk] ? 1 : 0;

    for (uint32_t j = 0; j < BB_CHK_DEG; j++) {
        uint32_t v = bb_chk_vars[chk][j];
        q[j] = post[v];
        if (q[j] < 0) sign ^= 1;
        magnitude[j] = int_abs(q[j]);
    }

    check_t mn1 = magnitude[0], mn2 = FP_MAX;
    uint32_t min_pos = 0;
    for (uint32_t j = 1; j < BB_CHK_DEG; j++) {
        if (magnitude[j] <= mn1) {
            mn2 = mn1;
            mn1 = magnitude[j];
            min_pos = j;
        } else if (magnitude[j] < mn2) {
            mn2 = magnitude[j];
        }
    }

    for (uint32_t j = 0; j < BB_CHK_DEG; j++) {
        int out_sign = sign ^ (q[j] < 0 ? 1 : 0);
        fp_t raw = (fp_t)((j == min_pos) ? mn2 : mn1);
        if (out_sign) raw = -raw;
        fp_t cv = (fp_t)(raw >> 1);
        uint32_t v = bb_chk_vars[chk][j];
        if (!final_iter) c2v[b + j] = cv;
        post[v] = fp_sat((int32_t)q[j] + (int32_t)cv);
    }
}

extern int main_kernel1(void);
int (*kernels[nr_kernels])(void) = {main_kernel1};
int main(void) { return kernels[DPU_INPUT_ARGUMENTS.kernel](); }

int main_kernel1(void) {
    uint32_t tid = me();
#if BP_NUM_GROUPS == 1
    uint32_t group = 0;
    uint32_t local_tid = tid;
#else
    uint32_t group = tid / BP_GROUP_SIZE;
    uint32_t local_tid = tid % BP_GROUP_SIZE;
#endif

    if (tid == 0) {
        mem_reset();
#if BP_NUM_GROUPS > 1
        for (uint32_t g = 0; g < BP_NUM_GROUPS; g++) {
            group_barriers[g].wait_queue = 0xff;
            group_barriers[g].count = BP_GROUP_SIZE;
            group_barriers[g].initial_count = BP_GROUP_SIZE;
            group_barriers[g].lock =
                (uint8_t)&ATOMIC_BIT_GET(group_barrier_locks)[g];
        }
#endif
    }
    barrier_wait(&all_tasklets);

    fp_t *c2v = shared_c2v[group];
    fp_t *post = shared_post[group];
    uint8_t *syn = shared_syn[group];

    for (uint32_t v = local_tid; v < BB_NV; v += BP_GROUP_SIZE)
        post[v] = (fp_t)DPU_INPUT_ARGUMENTS.L;

    uint32_t slot_size = ((BB_NC + 7) / 8) * 8;
    __mram_ptr uint8_t *heap = (__mram_ptr uint8_t *)DPU_MRAM_HEAP_POINTER;
    if (local_tid == 0)
        mram_read(heap + group * slot_size, syn, slot_size);
    group_wait(group);

    uint32_t n_iter = DPU_INPUT_ARGUMENTS.num_events;
    int first_is_final = (n_iter == 1);
    for (uint32_t color_position = 0; color_position < BB_NUM_COLORS;
         color_position++) {
        uint32_t color = bb_color_order[color_position];
        for (uint32_t pos = local_tid; pos < BB_CHECKS_PER_COLOR;
             pos += BP_GROUP_SIZE) {
            update_check_first(bb_check_colors[color][pos], c2v, post, syn,
                               first_is_final);
        }
        group_wait(group);
    }

    for (uint32_t iter = 1; iter < n_iter; iter++) {
        int final_iter = (iter + 1 == n_iter);
        for (uint32_t color_position = 0; color_position < BB_NUM_COLORS;
             color_position++) {
            uint32_t color = bb_color_order[color_position];
            for (uint32_t pos = local_tid; pos < BB_CHECKS_PER_COLOR;
                 pos += BP_GROUP_SIZE) {
                update_check(bb_check_colors[color][pos], c2v, post, syn,
                             final_iter);
            }
            group_wait(group);
        }
    }

    if (local_tid == 0) {
        uint32_t input_area = BP_NUM_GROUPS * slot_size;
        __mram_ptr uint8_t *out_ptr =
            heap + input_area + group * (BB_NV * 2);
        mram_write(post, out_ptr, BB_NV * 2);
    }
    barrier_wait(&all_tasklets);
    return 0;
}
