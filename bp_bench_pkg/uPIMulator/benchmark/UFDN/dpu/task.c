/* task.c — D-N UF Decoder (PIM-optimized) for Toric Code on DPU
 *
 * Same algorithm as D-N: per-root growth, support counters, peeling.
 * Data structures: fixed arrays, bitmap border, border compaction.
 * No hash maps, no dynamic allocation.
 *
 * Each tasklet decodes one independent syndrome from MRAM.
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

/* ---- Lattice ---- */
static uint32_t L_val;

static inline uint32_t vid(int r, int c) {
    return ((uint32_t)((r+(int)L_val)%(int)L_val))*L_val +
           ((uint32_t)((c+(int)L_val)%(int)L_val));
}
static void get_nbrs(uint32_t v, uint32_t out[4]) {
    uint32_t r=v/L_val, c=v%L_val;
    out[0]=vid((int)r-1,(int)c); out[1]=vid((int)r+1,(int)c);
    out[2]=vid((int)r,(int)c-1); out[3]=vid((int)r,(int)c+1);
}
static uint32_t eidx(uint32_t u, uint32_t v) {
    if(u>v){uint32_t t=u;u=v;v=t;}
    uint32_t r1=u/L_val,c1=u%L_val,r2=v/L_val,c2=v%L_val;
    if(c1==c2){uint32_t ur=r1;if((r2+1)%L_val==r1)ur=r2;return ur*L_val+c1;}
    else{uint32_t lc=c1;if((c2+1)%L_val==c1)lc=c2;return L_val*L_val+r1*L_val+lc;}
}

/* ---- Per-tasklet state ---- */
static uint32_t uf_par[NR_TASKLETS][MAX_V_DPU];
static uint16_t uf_sz[NR_TASKLETS][MAX_V_DPU];
static uint8_t  uf_parity[NR_TASKLETS][MAX_V_DPU];
static uint8_t  support[NR_TASKLETS][MAX_E_DPU];
static uint8_t  is_border[NR_TASKLETS][MAX_V_DPU];
static uint16_t border_list[NR_TASKLETS][MAX_BORDER];
static uint32_t n_border[NR_TASKLETS];
static uint32_t syn[NR_TASKLETS][MAX_V_DPU];

/* Fuse queue (per root, reused) */
typedef struct { uint16_t u, v; } e16_t;
static e16_t fuse_q[NR_TASKLETS][MAX_FUSE];
static uint32_t fq_len[NR_TASKLETS];

/* Spanning forest for peeling */
static e16_t peel_e[NR_TASKLETS][MAX_PEEL];
static uint32_t n_peel[NR_TASKLETS];

/* Error buffer */
static uint8_t err_buf[NR_TASKLETS][MAX_E_DPU];

/* ---- UF ---- */
static uint32_t uf_find(uint32_t tid, uint32_t x) {
    while (uf_par[tid][x] != x) {
        uf_par[tid][x] = uf_par[tid][uf_par[tid][x]];
        x = uf_par[tid][x];
    }
    return x;
}

/* ---- Border compaction ---- */
static void compact_border(uint32_t tid) {
    uint32_t NV = L_val * L_val;
    uint32_t nn = 0;
    for (uint32_t i = 0; i < NV; i++) is_border[tid][i] = 0;
    for (uint32_t i = 0; i < n_border[tid]; i++) {
        uint32_t v = border_list[tid][i];
        if (is_border[tid][v]) continue;
        uint32_t rv = uf_find(tid, v);
        uint32_t nbrs[4]; get_nbrs(v, nbrs);
        int on_border = 0;
        for (int j = 0; j < 4; j++)
            if (uf_find(tid, nbrs[j]) != rv) { on_border = 1; break; }
        if (on_border) {
            border_list[tid][nn] = v;
            is_border[tid][v] = 1;
            nn++;
        }
    }
    n_border[tid] = nn;
}

/* ---- Decode ---- */
static void decode(uint32_t tid) {
    uint32_t NV = L_val * L_val;
    uint32_t NE = 2 * NV;

    /* Init */
    for (uint32_t i = 0; i < NV; i++) {
        uf_par[tid][i] = i; uf_sz[tid][i] = 1;
        uf_parity[tid][i] = 0; is_border[tid][i] = 0;
    }
    for (uint32_t i = 0; i < NE; i++) support[tid][i] = 0;
    n_border[tid] = 0; n_peel[tid] = 0;

    /* Syndrome */
    for (uint32_t v = 0; v < NV; v++) {
        uint32_t nbrs[4]; get_nbrs(v, nbrs);
        uint32_t s = 0;
        for (int j = 0; j < 4; j++) s ^= err_buf[tid][eidx(v, nbrs[j])];
        syn[tid][v] = s;
    }

    /* Init border from syndrome */
    for (uint32_t i = 0; i < NV; i++) {
        if (syn[tid][i] & 1) {
            uf_parity[tid][i] = 1;
            border_list[tid][n_border[tid]] = (uint16_t)i;
            is_border[tid][i] = 1;
            n_border[tid]++;
        }
    }

    /* Growth loop */
    uint32_t max_steps = 2 * L_val;
    for (uint32_t step = 0; step < max_steps; step++) {
        /* Find odd roots */
        uint16_t odds[MAX_V_DPU]; uint32_t n_odd = 0;
        for (uint32_t i = 0; i < NV; i++) {
            uint32_t ri = uf_find(tid, i);
            if (ri == i && (uf_parity[tid][ri] & 1))
                odds[n_odd++] = (uint16_t)i;
        }
        if (n_odd == 0) break;

        /* Per-root growth + fusion */
        for (uint32_t oi = 0; oi < n_odd; oi++) {
            uint32_t root = odds[oi];
            if (!(uf_parity[tid][uf_find(tid, root)] & 1)) continue;

            fq_len[tid] = 0;

            /* Grow: scan border for this root */
            for (uint32_t bi = 0; bi < n_border[tid]; bi++) {
                uint32_t bv = border_list[tid][bi];
                if (uf_find(tid, bv) != uf_find(tid, root)) continue;

                uint32_t nbrs[4]; get_nbrs(bv, nbrs);
                for (int j = 0; j < 4; j++) {
                    uint32_t ei = eidx(bv, nbrs[j]);
                    if (support[tid][ei] >= 2) continue;
                    if (++support[tid][ei] == 2 && fq_len[tid] < MAX_FUSE) {
                        fuse_q[tid][fq_len[tid]].u = (uint16_t)bv;
                        fuse_q[tid][fq_len[tid]].v = (uint16_t)nbrs[j];
                        fq_len[tid]++;
                    }
                }
            }

            /* Fuse for this root */
            for (uint32_t fi = 0; fi < fq_len[tid]; fi++) {
                uint32_t r1 = uf_find(tid, fuse_q[tid][fi].u);
                uint32_t r2 = uf_find(tid, fuse_q[tid][fi].v);
                if (r1 == r2) continue;

                peel_e[tid][n_peel[tid]++] = fuse_q[tid][fi];

                if (uf_sz[tid][r1] < uf_sz[tid][r2]) { uint32_t t=r1; r1=r2; r2=t; }
                uf_par[tid][r2] = r1;
                uf_sz[tid][r1] += uf_sz[tid][r2];
                uf_parity[tid][r1] += uf_parity[tid][r2];

                /* Add endpoints + neighbors to border */
                uint16_t verts[2] = {fuse_q[tid][fi].u, fuse_q[tid][fi].v};
                for (int vi = 0; vi < 2; vi++) {
                    uint32_t w = verts[vi];
                    if (!is_border[tid][w] && n_border[tid] < MAX_BORDER) {
                        border_list[tid][n_border[tid]++] = (uint16_t)w;
                        is_border[tid][w] = 1;
                    }
                    uint32_t nbrs[4]; get_nbrs(w, nbrs);
                    for (int j = 0; j < 4; j++) {
                        uint32_t nb = nbrs[j];
                        if (!is_border[tid][nb] && n_border[tid] < MAX_BORDER) {
                            border_list[tid][n_border[tid]++] = (uint16_t)nb;
                            is_border[tid][nb] = 1;
                        }
                    }
                }
            }
        }

        compact_border(tid);
    }

    /* Peeling */
    uint8_t deg[MAX_V_DPU]; for (uint32_t i = 0; i < NV; i++) deg[i] = 0;
    uint8_t used[MAX_PEEL]; for (uint32_t i = 0; i < n_peel[tid]; i++) used[i] = 0;
    for (uint32_t i = 0; i < n_peel[tid]; i++) {
        deg[peel_e[tid][i].u]++; deg[peel_e[tid][i].v]++;
    }

    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = (int)n_peel[tid] - 1; i >= 0; i--) {
            if (used[i]) continue;
            uint16_t u = peel_e[tid][i].u, v = peel_e[tid][i].v;
            uint16_t leaf, other;
            if (deg[u] == 1) { leaf = u; other = v; }
            else if (deg[v] == 1) { leaf = v; other = u; }
            else continue;
            used[i] = 1; deg[leaf]--; deg[other]--;
            if (syn[tid][leaf] & 1) { syn[tid][leaf] ^= 1; syn[tid][other] ^= 1; }
            changed = 1;
        }
    }
}

/* ---- Main ---- */
extern int main_kernel1(void);
int (*kernels[nr_kernels])(void) = {main_kernel1};
int main(void) { return kernels[DPU_INPUT_ARGUMENTS.kernel](); }

int main_kernel1(void) {
    uint32_t tid = me();
    dpu_arguments_t args = DPU_INPUT_ARGUMENTS;
    L_val = args.L;
    uint32_t NE = args.num_edges;

    /* Read errors from MRAM */
    __mram_ptr uint8_t *heap = (__mram_ptr uint8_t *)DPU_MRAM_HEAP_POINTER;
    uint32_t slot_size = ((NE + 7) / 8) * 8;
    mram_read(heap + tid * slot_size, err_buf[tid], slot_size);

    barrier_wait(&my_barrier);

    decode(tid);

    volatile uint32_t sink = syn[tid][0];
    (void)sink;

    barrier_wait(&my_barrier);
    return 0;
}
