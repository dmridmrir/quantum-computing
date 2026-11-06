/*
 * task_alg3_var.c — BF-UF Algorithm 3 variant for [[144,12,12]] BB-QLDPC
 *
 * Optimization over task.c (Algorithm 3):
 *   Explicit cluster membership lists (linked lists per root):
 *     - chk_head/chk_tail/chk_next : check nodes in each cluster
 *     - var_head/var_tail/var_next  : variable nodes in each cluster
 *   Maintained in O(1) on every uf_unite_keep_a call.
 *   Member collection for GE: O(cluster_size) instead of O(NC+NV)=O(216).
 *   Early stopping behavior identical to task.c (no delayed GE).
 *
 * Ref: Löbl et al., arXiv:2407.15988 (2024)
 *
 * WRAM budget per tasklet:
 *   UF arrays (par + rnk + parity):         216×3  =  648 B
 *   Membership lists (head/tail/next):       216×4 + 72 + 144 = 1080 B
 *   BFS queue + visited + root_done:         216×3  =  648 B
 *   cl_var_idx + cl_checks + cl_vars:        144+72+144 = 360 B
 *   GF2 mat (72 rows × 5 uint32):                   = 1440 B
 *   gf2_pcol + e_cl:                         72+144 =  216 B
 *   syndrome + correction buffers:           72+144 =  216 B
 *   Total per tasklet:                              ~4608 B
 *   × 11 tasklets:                                 ~51 KB  (< 64 KB)
 */

#include <stdint.h>
#include <string.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>

#include "../support/common.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
BARRIER_INIT(my_barrier, NR_TASKLETS);

/* ------------------------------------------------------------------ */
/* Tanner graph tables                                                  */
/* ------------------------------------------------------------------ */
static const uint16_t bb_chk_vars[BB_NC][BB_CHK_DEG] = {
  {3,6,12,73,74,90},   {4,7,13,74,75,91},   {5,8,14,75,76,92},
  {0,9,15,76,77,93},   {1,10,16,72,77,94},  {2,11,17,72,73,95},
  {9,12,18,79,80,96},  {10,13,19,80,81,97}, {11,14,20,81,82,98},
  {6,15,21,82,83,99},  {7,16,22,78,83,100}, {8,17,23,78,79,101},
  {15,18,24,85,86,102},{16,19,25,86,87,103},{17,20,26,87,88,104},
  {12,21,27,88,89,105},{13,22,28,84,89,106},{14,23,29,84,85,107},
  {21,24,30,91,92,108},{22,25,31,92,93,109},{23,26,32,93,94,110},
  {18,27,33,94,95,111},{19,28,34,90,95,112},{20,29,35,90,91,113},
  {27,30,36,97,98,114},{28,31,37,98,99,115},{29,32,38,99,100,116},
  {24,33,39,100,101,117},{25,34,40,96,101,118},{26,35,41,96,97,119},
  {33,36,42,103,104,120},{34,37,43,104,105,121},{35,38,44,105,106,122},
  {30,39,45,106,107,123},{31,40,46,102,107,124},{32,41,47,102,103,125},
  {39,42,48,109,110,126},{40,43,49,110,111,127},{41,44,50,111,112,128},
  {36,45,51,112,113,129},{37,46,52,108,113,130},{38,47,53,108,109,131},
  {45,48,54,115,116,132},{46,49,55,116,117,133},{47,50,56,117,118,134},
  {42,51,57,118,119,135},{43,52,58,114,119,136},{44,53,59,114,115,137},
  {51,54,60,121,122,138},{52,55,61,122,123,139},{53,56,62,123,124,140},
  {48,57,63,124,125,141},{49,58,64,120,125,142},{50,59,65,120,121,143},
  {57,60,66,72,127,128},{58,61,67,73,128,129},{59,62,68,74,129,130},
  {54,63,69,75,130,131},{55,64,70,76,126,131},{56,65,71,77,126,127},
  {0,63,66,78,133,134}, {1,64,67,79,134,135}, {2,65,68,80,135,136},
  {3,60,69,81,136,137}, {4,61,70,82,132,137}, {5,62,71,83,132,133},
  {0,6,69,84,139,140},  {1,7,70,85,140,141},  {2,8,71,86,141,142},
  {3,9,66,87,142,143},  {4,10,67,88,138,143}, {5,11,68,89,138,139},
};

static uint8_t bb_var_chks[BB_NV][BB_VAR_DEG];
static uint8_t bb_var_chks_ready = 0;

/* ------------------------------------------------------------------ */
/* Per-tasklet WRAM: Union-Find                                        */
/* ------------------------------------------------------------------ */
static uint8_t uf_par    [NR_TASKLETS][BFUF_TOTAL];
static uint8_t uf_rnk    [NR_TASKLETS][BFUF_TOTAL];
static uint8_t uf_parity [NR_TASKLETS][BFUF_TOTAL];  /* 1 = cluster invalid */

/* ------------------------------------------------------------------ */
/* Per-tasklet WRAM: Explicit cluster membership (linked lists)        */
/*                                                                      */
/* chk_head[tid][root] : first check in cluster, 0xFF = empty          */
/* chk_tail[tid][root] : last  check in cluster, 0xFF = empty          */
/* chk_next[tid][c]    : next check after c in list, 0xFF = end        */
/* var_head[tid][root] : first var  in cluster, 0xFF = empty           */
/* var_tail[tid][root] : last  var  in cluster, 0xFF = empty           */
/* var_next[tid][v]    : next var  after v in list, 0xFF = end         */
/* ------------------------------------------------------------------ */
static uint8_t chk_head  [NR_TASKLETS][BFUF_TOTAL];  /* indexed by root */
static uint8_t chk_tail  [NR_TASKLETS][BFUF_TOTAL];  /* indexed by root */
static uint8_t chk_next  [NR_TASKLETS][BB_NC];       /* indexed by check */
static uint8_t var_head  [NR_TASKLETS][BFUF_TOTAL];  /* indexed by root */
static uint8_t var_tail  [NR_TASKLETS][BFUF_TOTAL];  /* indexed by root */
static uint8_t var_next  [NR_TASKLETS][BB_NV];       /* indexed by var  */

/* ------------------------------------------------------------------ */
/* Per-tasklet WRAM: BFS and GE scratch                               */
/* ------------------------------------------------------------------ */
static uint8_t  visited   [NR_TASKLETS][BFUF_TOTAL];
static uint8_t  bfs_q     [NR_TASKLETS][BB_NC];      /* check nodes only */
static uint8_t  root_done [NR_TASKLETS][BFUF_TOTAL];
static uint8_t  cl_var_idx[NR_TASKLETS][BB_NV];
static uint8_t  cl_checks [NR_TASKLETS][BB_NC];
static uint8_t  cl_vars   [NR_TASKLETS][BB_NV];
static uint8_t  syn_arr   [NR_TASKLETS][BB_NC];
static uint8_t  corr_arr  [NR_TASKLETS][BB_NV];

static uint32_t gf2_mat   [NR_TASKLETS][BB_NC][GF2_WORDS];
static int8_t   gf2_pcol  [NR_TASKLETS][BB_NC];
static uint8_t  e_cl      [NR_TASKLETS][BB_NV];

/* ------------------------------------------------------------------ */
/* Union-Find (path-halving + uniteKeepA + O(1) list concat)          */
/* ------------------------------------------------------------------ */
static uint8_t uf_find(uint32_t tid, uint8_t x) {
    while (uf_par[tid][x] != x) {
        uf_par[tid][x] = uf_par[tid][uf_par[tid][x]];
        x = uf_par[tid][x];
    }
    return x;
}

static void uf_unite_keep_a(uint32_t tid, uint8_t a, uint8_t b) {
    uint8_t ra = uf_find(tid, a);
    uint8_t rb = uf_find(tid, b);
    if (ra == rb) return;

    /* append rb's check list to ra's check list — O(1) */
    if (chk_head[tid][rb] != 0xFF) {
        if (chk_tail[tid][ra] != 0xFF)
            chk_next[tid][chk_tail[tid][ra]] = chk_head[tid][rb];
        else
            chk_head[tid][ra] = chk_head[tid][rb];
        chk_tail[tid][ra] = chk_tail[tid][rb];
    }

    /* append rb's var list to ra's var list — O(1) */
    if (var_head[tid][rb] != 0xFF) {
        if (var_tail[tid][ra] != 0xFF)
            var_next[tid][var_tail[tid][ra]] = var_head[tid][rb];
        else
            var_head[tid][ra] = var_head[tid][rb];
        var_tail[tid][ra] = var_tail[tid][rb];
    }

    /* UF union: keep ra as root */
    uf_par[tid][rb] = ra;
    if (uf_rnk[tid][ra] <= uf_rnk[tid][rb])
        uf_rnk[tid][ra] = uf_rnk[tid][rb] + 1;
}

/* ------------------------------------------------------------------ */
/* GF(2) Gaussian elimination (identical to task.c)                   */
/* ------------------------------------------------------------------ */
static int gf2_solve(uint32_t tid, int nc, int nv) {
    int prow = 0;
    int8_t *pcol = gf2_pcol[tid];
    uint32_t (*mat)[GF2_WORDS] = gf2_mat[tid];

    for (int i = 0; i < nc; i++) pcol[i] = -1;

    for (int col = 0; col < nv && prow < nc; col++) {
        uint32_t mask = 1u << (col & 31);
        int      word = col >> 5;

        int pivot = -1;
        for (int r = prow; r < nc; r++)
            if (mat[r][word] & mask) { pivot = r; break; }
        if (pivot < 0) continue;

        if (pivot != prow) {
            for (int w = 0; w < GF2_WORDS; w++) {
                uint32_t t = mat[prow][w];
                mat[prow][w] = mat[pivot][w];
                mat[pivot][w] = t;
            }
        }
        pcol[prow] = (int8_t)col;

        for (int r = 0; r < nc; r++)
            if (r != prow && (mat[r][word] & mask))
                for (int w = 0; w < GF2_WORDS; w++)
                    mat[r][w] ^= mat[prow][w];
        prow++;
    }

    int aug_word = nv >> 5;
    uint32_t aug_mask = 1u << (nv & 31);
    for (int r = prow; r < nc; r++)
        if (mat[r][aug_word] & aug_mask) return 0;

    uint8_t *e = e_cl[tid];
    for (int v = 0; v < nv; v++) e[v] = 0;
    for (int r = 0; r < prow; r++)
        if (pcol[r] >= 0)
            e[(uint8_t)pcol[r]] = (mat[r][aug_word] & aug_mask) ? 1 : 0;
    return 1;
}

/* ------------------------------------------------------------------ */
/* Cluster solvability check using linked-list member collection       */
/* O(nc_cl + nv_cl) traversal instead of O(NC+NV) full scan          */
/* ------------------------------------------------------------------ */
static int cluster_solvable_var(uint32_t tid, const uint8_t *syn, uint8_t root) {
    int nc_cl = 0, nv_cl = 0;
    for (int i = 0; i < BB_NV; i++) cl_var_idx[tid][i] = 0xFF;

    /* collect checks via linked list — O(nc_cl) */
    for (uint8_t c = chk_head[tid][root]; c != 0xFF; c = chk_next[tid][c])
        cl_checks[tid][nc_cl++] = c;

    /* collect vars via linked list — O(nv_cl) */
    for (uint8_t v = var_head[tid][root]; v != 0xFF; v = var_next[tid][v]) {
        cl_var_idx[tid][v] = (uint8_t)nv_cl;
        cl_vars[tid][nv_cl++] = v;
    }

    if (!nc_cl || !nv_cl) return 0;

    for (int r = 0; r < nc_cl; r++) {
        for (int w = 0; w < GF2_WORDS; w++) gf2_mat[tid][r][w] = 0;
        uint8_t c = cl_checks[tid][r];
        if (syn[c]) gf2_mat[tid][r][nv_cl >> 5] |= 1u << (nv_cl & 31);
        for (int j = 0; j < BB_CHK_DEG; j++) {
            uint8_t v   = (uint8_t)bb_chk_vars[c][j];
            uint8_t col = cl_var_idx[tid][v];
            if (col != 0xFF)
                gf2_mat[tid][r][col >> 5] |= 1u << (col & 31);
        }
    }
    return gf2_solve(tid, nc_cl, nv_cl);
}

/* ------------------------------------------------------------------ */
/* BF-UF decode — Algorithm 3 with O(cluster_size) member collection  */
/* ------------------------------------------------------------------ */
static void decode(uint32_t tid) {
    const uint8_t *syn  = syn_arr[tid];
    uint8_t       *corr = corr_arr[tid];

    /* init UF, membership lists, visited */
    for (uint8_t i = 0; i < BFUF_TOTAL; i++) {
        uf_par[tid][i]    = i;
        uf_rnk[tid][i]    = 0;
        uf_parity[tid][i] = 0;
        visited[tid][i]   = 0;
        if (i < BB_NC) {
            /* check node: starts with itself in check list, empty var list */
            chk_head[tid][i] = i;
            chk_tail[tid][i] = i;
            chk_next[tid][i] = 0xFF;
            var_head[tid][i] = 0xFF;
            var_tail[tid][i] = 0xFF;
        } else {
            /* var node: empty check list, itself in var list */
            uint8_t v = i - BB_NC;
            chk_head[tid][i] = 0xFF;
            chk_tail[tid][i] = 0xFF;
            var_head[tid][i] = v;
            var_tail[tid][i] = v;
            var_next[tid][v] = 0xFF;
        }
    }
    for (uint8_t i = 0; i < BB_NV; i++) corr[i] = 0;

    /* seed BFS from unsatisfied check nodes */
    int qh = 0, qt = 0, num_invalid = 0;
    for (uint8_t c = 0; c < BB_NC; c++) {
        if (syn[c]) {
            visited[tid][c]   = 1;
            uf_parity[tid][c] = 1;
            num_invalid++;
            bfs_q[tid][qt++] = c;
        }
    }

    /* BFS: check nodes only, double-step check→qubit→check */
    while (num_invalid > 0 && qh < qt) {
        uint8_t node = bfs_q[tid][qh++];
        uint8_t root = uf_find(tid, node);
        if (!uf_parity[tid][root]) continue;

        for (int j = 0; j < BB_CHK_DEG; j++) {
            uint8_t v  = (uint8_t)bb_chk_vars[node][j];
            uint8_t vn = BFUF_VAR_NODE(v);

            uint8_t rvn = uf_find(tid, vn);
            uint8_t rc  = uf_find(tid, node);
            if (rvn != rc) {
                if (uf_parity[tid][rvn]) { num_invalid--; uf_parity[tid][rvn] = 0; }
                uf_unite_keep_a(tid, node, vn);  /* adds v to root's var list */
            }
            visited[tid][vn] = 1;

            for (int k = 0; k < BB_VAR_DEG; k++) {
                uint8_t cn  = bb_var_chks[v][k];
                uint8_t rcn = uf_find(tid, cn);
                rc = uf_find(tid, node);
                if (rcn != rc) {
                    if (uf_parity[tid][rcn]) { num_invalid--; uf_parity[tid][rcn] = 0; }
                    uf_unite_keep_a(tid, node, cn);  /* adds cn to root's check list */
                }
                if (!visited[tid][cn]) {
                    visited[tid][cn] = 1;
                    bfs_q[tid][qt++] = cn;
                }
            }
        }

        /* update_cluster_validity: GE on expanded cluster */
        root = uf_find(tid, node);
        if (cluster_solvable_var(tid, syn, root))
            if (uf_parity[tid][root]) { uf_parity[tid][root] = 0; num_invalid--; }
    }

    /* ---- final GE per cluster (also uses linked lists) ---- */
    for (int i = 0; i < BFUF_TOTAL; i++) root_done[tid][i] = 0;

    for (int qi = 0; qi < qt; qi++) {
        uint8_t node = bfs_q[tid][qi];
        uint8_t root = uf_find(tid, node);
        if (root_done[tid][root]) continue;
        root_done[tid][root] = 1;

        int nc_cl = 0, nv_cl = 0;
        for (int i = 0; i < BB_NV; i++) cl_var_idx[tid][i] = 0xFF;

        for (uint8_t c = chk_head[tid][root]; c != 0xFF; c = chk_next[tid][c])
            cl_checks[tid][nc_cl++] = c;
        for (uint8_t v = var_head[tid][root]; v != 0xFF; v = var_next[tid][v]) {
            cl_var_idx[tid][v] = (uint8_t)nv_cl;
            cl_vars[tid][nv_cl++] = v;
        }

        if (!nc_cl || !nv_cl) continue;

        for (int r = 0; r < nc_cl; r++) {
            for (int w = 0; w < GF2_WORDS; w++) gf2_mat[tid][r][w] = 0;
            uint8_t c = cl_checks[tid][r];
            if (syn[c]) gf2_mat[tid][r][nv_cl >> 5] |= 1u << (nv_cl & 31);
            for (int j = 0; j < BB_CHK_DEG; j++) {
                uint8_t v   = (uint8_t)bb_chk_vars[c][j];
                uint8_t col = cl_var_idx[tid][v];
                if (col != 0xFF)
                    gf2_mat[tid][r][col >> 5] |= 1u << (col & 31);
            }
        }
        if (gf2_solve(tid, nc_cl, nv_cl))
            for (int col = 0; col < nv_cl; col++)
                corr[cl_vars[tid][col]] ^= e_cl[tid][col];
    }
}

/* ------------------------------------------------------------------ */
/* DPU main kernel                                                     */
/* ------------------------------------------------------------------ */
extern int main_kernel1(void);
int (*kernels[nr_kernels])(void) = {main_kernel1};
int main(void) { return kernels[DPU_INPUT_ARGUMENTS.kernel](); }

int main_kernel1(void) {
    uint32_t tid = me();

    if (tid == 0) {
        mem_reset();
        uint8_t cnt[BB_NV];
        for (int i = 0; i < BB_NV; i++) cnt[i] = 0;
        for (uint8_t c = 0; c < BB_NC; c++)
            for (int j = 0; j < BB_CHK_DEG; j++) {
                uint8_t v = bb_chk_vars[c][j];
                bb_var_chks[v][cnt[v]++] = c;
            }
        bb_var_chks_ready = 1;
    }
    barrier_wait(&my_barrier);

    __mram_ptr uint8_t *heap = (__mram_ptr uint8_t *)DPU_MRAM_HEAP_POINTER;
    mram_read(heap + tid * SYN_SLOT_SIZE, syn_arr[tid], SYN_SLOT_SIZE);

    decode(tid);

    uint32_t out_off = NR_TASKLETS * SYN_SLOT_SIZE + tid * CORR_SLOT_SIZE;
    mram_write(corr_arr[tid], heap + out_off, CORR_SLOT_SIZE);

    barrier_wait(&my_barrier);
    return 0;
}
