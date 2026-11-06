/*
 * bfuf_cpu.c - CPU reference: BF-UF decoder for [[144,12,12]] BB-QLDPC
 *
 * Ref: Löbl et al., "Breadth-first graph traversal union-find decoder",
 *      arXiv:2407.15988 (2024)
 *
 * Algorithm:
 *   Phase 1 — BFS from unsatisfied check nodes on Tanner graph,
 *              merging clusters via Union-Find as edges are traversed.
 *   Phase 2 — For each cluster, solve H_cl * e = s_cl over GF(2)
 *              via Gaussian elimination and apply the correction.
 *
 * Compile: gcc -O2 -o bfuf_cpu bfuf_cpu.c
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* [[144,12,12]] Bivariate Bicycle QLDPC — parity check matrix        */
/* ------------------------------------------------------------------ */
#define BB_NC      72    /* check nodes  (rows of H)   */
#define BB_NV     144    /* variable nodes (cols of H) */
#define BB_CHK_DEG  6    /* checks per variable node   */
#define BB_VAR_DEG  3    /* variables per check node   */
#define BB_NNZ    432    /* non-zeros in H             */

/* bb_chk_vars[c][j] = j-th variable index (0..143) of check c */
static const uint8_t bb_chk_vars[BB_NC][BB_CHK_DEG] = {
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

/*
 * bb_var_map[v][j] = edge index (0..431) for variable v's j-th check.
 * check_node = bb_var_map[v][j] / BB_CHK_DEG
 */
static const uint16_t bb_var_map[BB_NV][BB_VAR_DEG] = {
  {18,360,396},{24,366,402},{30,372,408},{0,378,414},{6,384,420},
  {12,390,426},{1,54,397},{7,60,403},{13,66,409},{19,36,415},
  {25,42,421},{31,48,427},{2,37,90},{8,43,96},{14,49,102},
  {20,55,72},{26,61,78},{32,67,84},{38,73,126},{44,79,132},
  {50,85,138},{56,91,108},{62,97,114},{68,103,120},{74,109,162},
  {80,115,168},{86,121,174},{92,127,144},{98,133,150},{104,139,156},
  {110,145,198},{116,151,204},{122,157,210},{128,163,180},{134,169,186},
  {140,175,192},{146,181,234},{152,187,240},{158,193,246},{164,199,216},
  {170,205,222},{176,211,228},{182,217,270},{188,223,276},{194,229,282},
  {200,235,252},{206,241,258},{212,247,264},{218,253,306},{224,259,312},
  {230,265,318},{236,271,288},{242,277,294},{248,283,300},{254,289,342},
  {260,295,348},{266,301,354},{272,307,324},{278,313,330},{284,319,336},
  {290,325,379},{296,331,385},{302,337,391},{308,343,361},{314,349,367},
  {320,355,373},{326,362,416},{332,368,422},{338,374,428},{344,380,398},
  {350,386,404},{356,392,410},{27,33,327},{3,34,333},{4,9,339},
  {10,15,345},{16,21,351},{22,28,357},{63,69,363},{39,70,369},
  {40,45,375},{46,51,381},{52,57,387},{58,64,393},{99,105,399},
  {75,106,405},{76,81,411},{82,87,417},{88,93,423},{94,100,429},
  {5,135,141},{11,111,142},{17,112,117},{23,118,123},{29,124,129},
  {35,130,136},{41,171,177},{47,147,178},{53,148,153},{59,154,159},
  {65,160,165},{71,166,172},{77,207,213},{83,183,214},{89,184,189},
  {95,190,195},{101,196,201},{107,202,208},{113,243,249},{119,219,250},
  {125,220,225},{131,226,231},{137,232,237},{143,238,244},{149,279,285},
  {155,255,286},{161,256,261},{167,262,267},{173,268,273},{179,274,280},
  {185,315,321},{191,291,322},{197,292,297},{203,298,303},{209,304,309},
  {215,310,316},{221,352,358},{227,328,359},{233,329,334},{239,335,340},
  {245,341,346},{251,347,353},{257,388,394},{263,364,395},{269,365,370},
  {275,371,376},{281,377,382},{287,383,389},{293,424,430},{299,400,431},
  {305,401,406},{311,407,412},{317,413,418},{323,419,425},
};

/* ------------------------------------------------------------------ */
/* Node encoding: 0..71 = check nodes, 72..215 = variable nodes       */
/* ------------------------------------------------------------------ */
#define TOTAL_NODES  (BB_NC + BB_NV)  /* 216 */
#define VAR_NODE(v)  (BB_NC + (v))
#define GET_VAR(n)   ((n) - BB_NC)

/* variable→check adjacency derived from bb_var_map */
static uint8_t var_chks[BB_NV][BB_VAR_DEG];

static void build_var_chks(void) {
    for (int v = 0; v < BB_NV; v++)
        for (int j = 0; j < BB_VAR_DEG; j++)
            var_chks[v][j] = (uint8_t)(bb_var_map[v][j] / BB_CHK_DEG);
}

/* ------------------------------------------------------------------ */
/* Union-Find with path halving + union by rank                        */
/* ------------------------------------------------------------------ */
static int uf_par[TOTAL_NODES];
static int uf_rnk[TOTAL_NODES];

static int uf_find(int x) {
    while (uf_par[x] != x) { uf_par[x] = uf_par[uf_par[x]]; x = uf_par[x]; }
    return x;
}

static void uf_unite(int a, int b) {
    a = uf_find(a); b = uf_find(b);
    if (a == b) return;
    if (uf_rnk[a] < uf_rnk[b]) { int t = a; a = b; b = t; }
    uf_par[b] = a;
    if (uf_rnk[a] == uf_rnk[b]) uf_rnk[a]++;
}

/* ------------------------------------------------------------------ */
/* GF(2) Gaussian elimination                                          */
/* Solve H * e = s (mod 2), H is nc×nv, rows bit-packed into uint64_t */
/* Returns 1 if a solution exists, 0 otherwise.                       */
/* Free variables are left as 0 (minimum-weight bias not guaranteed). */
/* ------------------------------------------------------------------ */
#define GF2_WORDS 3   /* ceil((BB_NV+1)/64) = ceil(145/64) = 3 */

static int gf2_solve(uint8_t (*H)[BB_NV], uint8_t *s,
                     uint8_t *e, int nc, int nv) {
    uint64_t mat[BB_NC][GF2_WORDS];
    memset(mat, 0, sizeof(mat));

    for (int r = 0; r < nc; r++) {
        for (int c = 0; c < nv; c++)
            if (H[r][c]) mat[r][c >> 6] |= 1ULL << (c & 63);
        /* augmented column at position nv */
        if (s[r]) mat[r][nv >> 6] |= 1ULL << (nv & 63);
    }

    int prow = 0;
    int pcol_of[BB_NC];
    memset(pcol_of, -1, sizeof(pcol_of));

    for (int col = 0; col < nv && prow < nc; col++) {
        /* find pivot row */
        int pivot = -1;
        for (int r = prow; r < nc; r++)
            if ((mat[r][col >> 6] >> (col & 63)) & 1) { pivot = r; break; }
        if (pivot < 0) continue;

        /* swap pivot row into position */
        if (pivot != prow)
            for (int w = 0; w < GF2_WORDS; w++) {
                uint64_t t = mat[prow][w];
                mat[prow][w] = mat[pivot][w];
                mat[pivot][w] = t;
            }

        pcol_of[prow] = col;

        /* eliminate column in all other rows */
        for (int r = 0; r < nc; r++)
            if (r != prow && (mat[r][col >> 6] >> (col & 63)) & 1)
                for (int w = 0; w < GF2_WORDS; w++)
                    mat[r][w] ^= mat[prow][w];
        prow++;
    }

    /* consistency check: zero rows with non-zero augmented column → no solution */
    for (int r = prow; r < nc; r++)
        if ((mat[r][nv >> 6] >> (nv & 63)) & 1) return 0;

    /* extract solution (free variables stay 0) */
    memset(e, 0, nv);
    for (int r = 0; r < prow; r++)
        if (pcol_of[r] >= 0)
            e[pcol_of[r]] = (mat[r][nv >> 6] >> (nv & 63)) & 1;
    return 1;
}

/* ------------------------------------------------------------------ */
/* BF-UF decoder                                                       */
/* syn[0..71]  : syndrome (1 = unsatisfied check)                     */
/* corr[0..143]: output correction vector                             */
/* Returns 1 (always; caller checks H*corr=syn for success)           */
/* ------------------------------------------------------------------ */

/* static workspace avoids repeated stack allocation */
static uint8_t  visited[TOTAL_NODES];
static int      bfs_q[TOTAL_NODES];
static uint8_t  root_done[TOTAL_NODES];
static uint8_t  cl_var_idx[BB_NV];   /* global var idx → local col in cluster */
static int      cl_checks[BB_NC];
static int      cl_vars[BB_NV];
static uint8_t  H_cl[BB_NC][BB_NV];
static uint8_t  s_cl[BB_NC];
static uint8_t  e_cl[BB_NV];

int bfuf_decode(const uint8_t syn[BB_NC], uint8_t corr[BB_NV]) {
    /* --- init Union-Find and BFS state --- */
    for (int i = 0; i < TOTAL_NODES; i++) { uf_par[i] = i; uf_rnk[i] = 0; }
    memset(visited, 0, sizeof(visited));
    int qh = 0, qt = 0;

    /* seed BFS from every unsatisfied check */
    for (int c = 0; c < BB_NC; c++) {
        if (syn[c]) { visited[c] = 1; bfs_q[qt++] = c; }
    }

    /* --- BFS on Tanner graph, merging clusters via UF --- */
    while (qh < qt) {
        int node = bfs_q[qh++];

        if (node < BB_NC) {                        /* check node */
            int c = node;
            for (int j = 0; j < BB_CHK_DEG; j++) {
                int vn = VAR_NODE(bb_chk_vars[c][j]);
                uf_unite(node, vn);
                if (!visited[vn]) { visited[vn] = 1; bfs_q[qt++] = vn; }
            }
        } else {                                   /* variable node */
            int v = GET_VAR(node);
            for (int j = 0; j < BB_VAR_DEG; j++) {
                int cn = var_chks[v][j];
                uf_unite(node, cn);
                if (!visited[cn]) { visited[cn] = 1; bfs_q[qt++] = cn; }
            }
        }
    }

    /* --- decode each cluster --- */
    memset(corr, 0, BB_NV);
    memset(root_done, 0, sizeof(root_done));

    for (int qi = 0; qi < qt; qi++) {
        int node = bfs_q[qi];
        if (node >= BB_NC) continue;   /* iterate cluster roots via check nodes only */

        int root = uf_find(node);
        if (root_done[root]) continue;
        root_done[root] = 1;

        /* collect cluster members */
        int nc_cl = 0, nv_cl = 0;
        memset(cl_var_idx, 0xFF, BB_NV);   /* 0xFF = not in cluster */

        for (int c = 0; c < BB_NC; c++)
            if (visited[c] && uf_find(c) == root)
                cl_checks[nc_cl++] = c;
        for (int v = 0; v < BB_NV; v++)
            if (visited[VAR_NODE(v)] && uf_find(VAR_NODE(v)) == root) {
                cl_var_idx[v] = (uint8_t)nv_cl;
                cl_vars[nv_cl++] = v;
            }

        if (!nc_cl || !nv_cl) continue;

        /* build H_cl (nc_cl × nv_cl) and s_cl */
        memset(H_cl, 0, sizeof(H_cl));
        for (int r = 0; r < nc_cl; r++) {
            int c = cl_checks[r];
            s_cl[r] = syn[c];
            for (int j = 0; j < BB_CHK_DEG; j++) {
                int v = bb_chk_vars[c][j];
                if (cl_var_idx[v] != 0xFF)
                    H_cl[r][cl_var_idx[v]] = 1;
            }
        }

        /* solve H_cl * e = s_cl and apply correction */
        if (gf2_solve(H_cl, s_cl, e_cl, nc_cl, nv_cl))
            for (int col = 0; col < nv_cl; col++)
                corr[cl_vars[col]] ^= e_cl[col];
        /* if unsolvable: leave cluster bits as 0 (decoding failure) */
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* Syndrome computation helper                                         */
/* ------------------------------------------------------------------ */
static void compute_syndrome(const uint8_t bits[BB_NV], uint8_t syn[BB_NC]) {
    for (int c = 0; c < BB_NC; c++) {
        uint8_t s = 0;
        for (int j = 0; j < BB_CHK_DEG; j++)
            s ^= bits[bb_chk_vars[c][j]];
        syn[c] = s;
    }
}

/* ------------------------------------------------------------------ */
/* Monte Carlo test harness                                            */
/* ------------------------------------------------------------------ */
int main(void) {
    srand((unsigned)time(NULL));
    build_var_chks();

    const int   N_TRIALS = 20000;
    const double rates[] = {0.01, 0.03, 0.05, 0.07, 0.10};
    const int   N_RATES  = (int)(sizeof(rates)/sizeof(rates[0]));

    uint8_t error[BB_NV], syn[BB_NC], corr[BB_NV], syn_check[BB_NC];

    printf("BF-UF decoder — [[144,12,12]] BB-QLDPC (BSC, %d trials)\n\n",
           N_TRIALS);
    printf("%-8s  %-12s  %-12s\n", "p", "FER", "fail/total");
    printf("%-8s  %-12s  %-12s\n", "---", "---", "---");

    for (int ri = 0; ri < N_RATES; ri++) {
        double p = rates[ri];
        int    fail = 0;

        for (int t = 0; t < N_TRIALS; t++) {
            /* random error pattern */
            for (int v = 0; v < BB_NV; v++)
                error[v] = ((double)rand() / ((double)RAND_MAX + 1.0) < p) ? 1 : 0;

            compute_syndrome(error, syn);
            bfuf_decode(syn, corr);

            /* verify: H * corr should equal syn */
            compute_syndrome(corr, syn_check);
            for (int c = 0; c < BB_NC; c++) {
                if (syn_check[c] != syn[c]) { fail++; break; }
            }
        }

        printf("%-8.3f  %-12.4f  %d/%d\n",
               p, (double)fail / N_TRIALS, fail, N_TRIALS);
    }
    return 0;
}
