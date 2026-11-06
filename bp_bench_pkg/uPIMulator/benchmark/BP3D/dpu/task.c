/* BP3D DPU Kernel for uPIMulator — Min-Sum BP on 3D Surface Code
 *
 * Single-tasklet: 1 DPU = 1 logical qubit BP decoder
 * WRAM: node arrays (bel, sp, m1, m2, llr)
 * MRAM: edge messages (v2c, c2v), channel LLR, result
 *
 * Memory access pattern: MRAM block DMA reads/writes every pass
 * → heavy memory traffic, ideal for demonstrating PIM advantage
 */

#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>

#include "../support/common.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

BARRIER_INIT(my_barrier, NR_TASKLETS);

int32_t w_bel[MAX_NODES];
int32_t w_sp[MAX_NODES];
int32_t w_m1[MAX_NODES];
int32_t w_m2[MAX_NODES];
int32_t w_llr[MAX_NODES];

__dma_aligned int32_t buf_a[BLK];
__dma_aligned int32_t buf_b[BLK];
__dma_aligned bp3d_result_t res_buf;

static uint32_t blk_rd(__mram_ptr uint8_t *base, uint32_t bi, uint32_t N, int32_t *buf) {
    uint32_t s = bi * BLK;
    uint32_t n = (s + BLK <= N) ? BLK : (N - s);
    uint32_t b = ALIGN8(n * 4);
    if (b < 8) b = 8; if (b > BLK_B) b = BLK_B;
    mram_read((__mram_ptr void *)(base + s * 4), buf, b);
    return n;
}
static void blk_wr(__mram_ptr uint8_t *base, uint32_t bi, uint32_t n, int32_t *buf) {
    uint32_t s = bi * BLK;
    uint32_t b = ALIGN8(n * 4);
    if (b < 8) b = 8; if (b > BLK_B) b = BLK_B;
    mram_write(buf, (__mram_ptr void *)(base + s * 4), b);
}

static inline int32_t apply_alpha(int32_t sign_val, int32_t mag) {
    int32_t scaled = (mag * ALPHA_NUM) >> ALPHA_SHIFT;
    return sign_val * scaled;
}

extern int main_kernel1(void);
int (*kernels[1])(void) = {main_kernel1};

int main(void) {
    return kernels[DPU_INPUT_ARGUMENTS.kernel]();
}

int main_kernel1(void) {
    unsigned int tasklet_id = me();
    if (tasklet_id == 0) {
        mem_reset();
    }
    barrier_wait(&my_barrier);

    /* Only tasklet 0 does the BP work (single-tasklet BP) */
    if (tasklet_id != 0) {
        barrier_wait(&my_barrier);
        return 0;
    }

    dpu_arguments_t a = DPU_INPUT_ARGUMENTS;
    uint32_t d = a.d, T = a.T;
    uint32_t ne = a.num_edges, nn = a.num_nodes;
    uint32_t max_iter = a.max_iter;
    uint32_t nr = a.n_right, ndn = a.n_down, nf = a.n_future;
    uint32_t dd = d * d, dm1 = d - 1;

    __mram_ptr uint8_t *heap = (__mram_ptr uint8_t *)DPU_MRAM_HEAP_POINTER;
    __mram_ptr uint8_t *m_v2c = heap + OFF_V2C;
    __mram_ptr uint8_t *m_c2v = heap + OFF_C2V(ne);

    /* Load LLR from MRAM */
    uint32_t llr_off = OFF_RES(ne) + ALIGN8(sizeof(bp3d_result_t));
    {
        uint32_t nb = (nn + BLK - 1) / BLK;
        for (uint32_t bi = 0; bi < nb; bi++) {
            uint32_t cnt = blk_rd(heap + llr_off, bi, nn, buf_a);
            uint32_t s = bi * BLK;
            for (uint32_t i = 0; i < cnt; i++) w_llr[s+i] = buf_a[i];
        }
    }

    uint32_t total_msgs = 0;
    uint32_t nb_r = (nr + BLK - 1) / BLK;
    uint32_t nb_d = (ndn + BLK - 1) / BLK;
    uint32_t nb_f = (nf > 0) ? ((nf + BLK - 1) / BLK) : 0;

    for (uint32_t iter = 0; iter < max_iter; iter++) {

        /* PASS 1: belief = llr + c2v */
        for (uint32_t i = 0; i < nn; i++) w_bel[i] = w_llr[i];

        { uint32_t t_e=0,r_e=0,c_e=0;
          for (uint32_t bi=0; bi<nb_r; bi++) {
            uint32_t cnt = blk_rd(m_c2v, bi, nr, buf_a);
            for (uint32_t li=0; li<cnt; li++) {
              uint32_t na = t_e*dd + r_e*d + c_e;
              w_bel[na] += buf_a[li]; w_bel[na+1] += buf_a[li];
              if (++c_e >= dm1) { c_e=0; if (++r_e >= d) { r_e=0; t_e++; } }
            }
          }
        }
        { uint32_t t_e=0,r_e=0,c_e=0;
          __mram_ptr uint8_t *mc = m_c2v + nr*4;
          for (uint32_t bi=0; bi<nb_d; bi++) {
            uint32_t cnt = blk_rd(mc, bi, ndn, buf_a);
            for (uint32_t li=0; li<cnt; li++) {
              uint32_t na = t_e*dd + r_e*d + c_e;
              w_bel[na] += buf_a[li]; w_bel[na+d] += buf_a[li];
              if (++c_e >= d) { c_e=0; if (++r_e >= dm1) { r_e=0; t_e++; } }
            }
          }
        }
        if (nf > 0) {
          uint32_t t_e=0,r_e=0,c_e=0;
          __mram_ptr uint8_t *mc = m_c2v + (nr+ndn)*4;
          for (uint32_t bi=0; bi<nb_f; bi++) {
            uint32_t cnt = blk_rd(mc, bi, nf, buf_a);
            for (uint32_t li=0; li<cnt; li++) {
              uint32_t na = t_e*dd + r_e*d + c_e;
              w_bel[na] += buf_a[li]; w_bel[na+dd] += buf_a[li];
              if (++c_e >= d) { c_e=0; if (++r_e >= d) { r_e=0; t_e++; } }
            }
          }
        }

        /* PASS 2: v2c = bel - c2v, accumulate min-sum */
        for (uint32_t i = 0; i < nn; i++) { w_sp[i]=1; w_m1[i]=LLR_INF; w_m2[i]=LLR_INF; }

        { uint32_t t_e=0,r_e=0,c_e=0;
          for (uint32_t bi=0; bi<nb_r; bi++) {
            uint32_t cnt = blk_rd(m_c2v, bi, nr, buf_a);
            for (uint32_t li=0; li<cnt; li++) {
              uint32_t na = t_e*dd + r_e*d + c_e, nb2 = na+1;
              int32_t v2c = w_bel[na] - buf_a[li];
              if (v2c > LLR_INF) v2c = LLR_INF;
              if (v2c < -LLR_INF) v2c = -LLR_INF;
              buf_b[li] = v2c;
              int32_t av = v2c; if (av < 0) av = -av;
              int32_t sv = (v2c >= 0) ? 1 : -1;
              w_sp[na] *= sv;
              if (av < w_m1[na]) { w_m2[na] = w_m1[na]; w_m1[na] = av; }
              else if (av < w_m2[na]) w_m2[na] = av;
              w_sp[nb2] *= sv;
              if (av < w_m1[nb2]) { w_m2[nb2] = w_m1[nb2]; w_m1[nb2] = av; }
              else if (av < w_m2[nb2]) w_m2[nb2] = av;
              if (++c_e >= dm1) { c_e=0; if (++r_e >= d) { r_e=0; t_e++; } }
            }
            blk_wr(m_v2c, bi, cnt, buf_b);
            total_msgs += cnt;
          }
        }
        { uint32_t t_e=0,r_e=0,c_e=0;
          __mram_ptr uint8_t *mc = m_c2v + nr*4;
          __mram_ptr uint8_t *mv = m_v2c + nr*4;
          for (uint32_t bi=0; bi<nb_d; bi++) {
            uint32_t cnt = blk_rd(mc, bi, ndn, buf_a);
            for (uint32_t li=0; li<cnt; li++) {
              uint32_t na = t_e*dd + r_e*d + c_e, nb2 = na+d;
              int32_t v2c = w_bel[na] - buf_a[li];
              if (v2c > LLR_INF) v2c = LLR_INF;
              if (v2c < -LLR_INF) v2c = -LLR_INF;
              buf_b[li] = v2c;
              int32_t av = v2c; if (av < 0) av = -av;
              int32_t sv = (v2c >= 0) ? 1 : -1;
              w_sp[na] *= sv;
              if (av < w_m1[na]) { w_m2[na] = w_m1[na]; w_m1[na] = av; }
              else if (av < w_m2[na]) w_m2[na] = av;
              w_sp[nb2] *= sv;
              if (av < w_m1[nb2]) { w_m2[nb2] = w_m1[nb2]; w_m1[nb2] = av; }
              else if (av < w_m2[nb2]) w_m2[nb2] = av;
              if (++c_e >= d) { c_e=0; if (++r_e >= dm1) { r_e=0; t_e++; } }
            }
            blk_wr(mv, bi, cnt, buf_b);
            total_msgs += cnt;
          }
        }
        if (nf > 0) {
          uint32_t t_e=0,r_e=0,c_e=0;
          __mram_ptr uint8_t *mc = m_c2v + (nr+ndn)*4;
          __mram_ptr uint8_t *mv = m_v2c + (nr+ndn)*4;
          for (uint32_t bi=0; bi<nb_f; bi++) {
            uint32_t cnt = blk_rd(mc, bi, nf, buf_a);
            for (uint32_t li=0; li<cnt; li++) {
              uint32_t na = t_e*dd + r_e*d + c_e, nb2 = na+dd;
              int32_t v2c = w_bel[na] - buf_a[li];
              if (v2c > LLR_INF) v2c = LLR_INF;
              if (v2c < -LLR_INF) v2c = -LLR_INF;
              buf_b[li] = v2c;
              int32_t av = v2c; if (av < 0) av = -av;
              int32_t sv = (v2c >= 0) ? 1 : -1;
              w_sp[na] *= sv;
              if (av < w_m1[na]) { w_m2[na] = w_m1[na]; w_m1[na] = av; }
              else if (av < w_m2[na]) w_m2[na] = av;
              w_sp[nb2] *= sv;
              if (av < w_m1[nb2]) { w_m2[nb2] = w_m1[nb2]; w_m1[nb2] = av; }
              else if (av < w_m2[nb2]) w_m2[nb2] = av;
              if (++c_e >= d) { c_e=0; if (++r_e >= d) { r_e=0; t_e++; } }
            }
            blk_wr(mv, bi, cnt, buf_b);
            total_msgs += cnt;
          }
        }

        /* PASS 3: c2v = alpha * sign * min */
        { uint32_t t_e=0,r_e=0,c_e=0;
          for (uint32_t bi=0; bi<nb_r; bi++) {
            uint32_t cnt = blk_rd(m_v2c, bi, nr, buf_a);
            for (uint32_t li=0; li<cnt; li++) {
              uint32_t na = t_e*dd + r_e*d + c_e;
              int32_t v = buf_a[li], av = v; if (av < 0) av = -av;
              int32_t sv = (v >= 0) ? 1 : -1;
              int32_t mag = (av <= w_m1[na]) ? w_m2[na] : w_m1[na];
              buf_b[li] = apply_alpha(w_sp[na] * sv, mag);
              if (++c_e >= dm1) { c_e=0; if (++r_e >= d) { r_e=0; t_e++; } }
            }
            blk_wr(m_c2v, bi, cnt, buf_b);
            total_msgs += cnt;
          }
        }
        { uint32_t t_e=0,r_e=0,c_e=0;
          __mram_ptr uint8_t *mv = m_v2c + nr*4;
          __mram_ptr uint8_t *mc = m_c2v + nr*4;
          for (uint32_t bi=0; bi<nb_d; bi++) {
            uint32_t cnt = blk_rd(mv, bi, ndn, buf_a);
            for (uint32_t li=0; li<cnt; li++) {
              uint32_t na = t_e*dd + r_e*d + c_e;
              int32_t v = buf_a[li], av = v; if (av < 0) av = -av;
              int32_t sv = (v >= 0) ? 1 : -1;
              int32_t mag = (av <= w_m1[na]) ? w_m2[na] : w_m1[na];
              buf_b[li] = apply_alpha(w_sp[na] * sv, mag);
              if (++c_e >= d) { c_e=0; if (++r_e >= dm1) { r_e=0; t_e++; } }
            }
            blk_wr(mc, bi, cnt, buf_b);
            total_msgs += cnt;
          }
        }
        if (nf > 0) {
          uint32_t t_e=0,r_e=0,c_e=0;
          __mram_ptr uint8_t *mv = m_v2c + (nr+ndn)*4;
          __mram_ptr uint8_t *mc = m_c2v + (nr+ndn)*4;
          for (uint32_t bi=0; bi<nb_f; bi++) {
            uint32_t cnt = blk_rd(mv, bi, nf, buf_a);
            for (uint32_t li=0; li<cnt; li++) {
              uint32_t na = t_e*dd + r_e*d + c_e;
              int32_t v = buf_a[li], av = v; if (av < 0) av = -av;
              int32_t sv = (v >= 0) ? 1 : -1;
              int32_t mag = (av <= w_m1[na]) ? w_m2[na] : w_m1[na];
              buf_b[li] = apply_alpha(w_sp[na] * sv, mag);
              if (++c_e >= d) { c_e=0; if (++r_e >= d) { r_e=0; t_e++; } }
            }
            blk_wr(mc, bi, cnt, buf_b);
            total_msgs += cnt;
          }
        }
    }

    res_buf.cycle_count = 0; /* perfcounter not available in uPIMulator */
    res_buf.iterations_run = max_iter;
    res_buf.total_msgs_processed = total_msgs;
    mram_write(&res_buf, (__mram_ptr void *)(heap + OFF_RES(ne)), ALIGN8(sizeof(bp3d_result_t)));

    barrier_wait(&my_barrier);
    return 0;
}
