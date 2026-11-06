/* bp_layered_ler_check.c
 *
 * Standalone validation: compares task_opt.c (flooding) against task_layered.c
 * (layered / Gauss-Seidel) using the project's real LER metric (check_conv +
 * is_stabilizer, same as run_bp_increase.sh's embedded CPU reference), not just
 * the BP-convergence proxy used in bp_schedule_compare.py.
 *
 * bp_decode_layered here is a line-for-line match of task_layered.c / the Go
 * bpDecodeLayered in bpms_layered_verify.go: int32 c2v/lambda, q NOT
 * pre-saturated, no first/final-iteration branching (this project's original
 * structure -- BP-PIM's own branchy/double-saturating int16 version measured
 * slower on real DPU cycles at iter=5, see task_layered.c's header comment).
 * alpha=6/8=0.75 is kept from BP-PIM's tuning (see
 * BP-PIM/verify/BP_LAYERED2_OPTIMIZATION_2026-07-23.md) -- that was tuned at
 * BSC p=0.03, not this project's p=0.001..0.009, so treat the exact iteration
 * counts below as a starting point to re-check, not a final answer.
 *
 * Build (needs gcc, not available in the machine this was written on):
 *   gcc -O2 -fopenmp -o bp_layered_ler_check bp_layered_ler_check.c -lm
 *   ./bp_layered_ler_check
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define BB_NC 72
#define BB_NV 144
#define BB_NNZ 432
#define BB_CHK_DEG 6
#define BB_VAR_DEG 3
#define FP_MAX 32000
typedef int16_t fp_t;

static uint16_t chk_vars[BB_NC][BB_CHK_DEG];
static uint16_t var_map[BB_NV][BB_VAR_DEG];

static void build_pcm(void) {
    int l=12, m=6, lm=72;
    uint8_t Hz[BB_NC][BB_NV]; memset(Hz, 0, sizeof(Hz));
    for (int i=0;i<l;i++) for (int j=0;j<m;j++) {
        int col=i*m+j;
        Hz[i*m+((j+3)%m)][col]    ^=1; Hz[((i+1)%l)*m+j][col]  ^=1;
        Hz[((i+2)%l)*m+j][col]    ^=1; Hz[((i+3)%l)*m+j][lm+col]^=1;
        Hz[i*m+((j+1)%m)][lm+col] ^=1; Hz[i*m+((j+2)%m)][lm+col]^=1;
    }
    for (int c=0;c<BB_NC;c++) {
        int pos=0;
        for (int v=0;v<BB_NV;v++) if (Hz[c][v]) chk_vars[c][pos++]=v;
    }
    uint8_t sl[BB_NV]; memset(sl,0,BB_NV);
    for (int c=0;c<BB_NC;c++)
        for (int p=0;p<BB_CHK_DEG;p++) {
            int v=chk_vars[c][p]; var_map[v][sl[v]++]=c*BB_CHK_DEG+p;
        }
}

static uint8_t Hz_mat[BB_NC][BB_NV];
static void build_hz(void) {
    memset(Hz_mat, 0, sizeof(Hz_mat));
    for (int c = 0; c < BB_NC; c++)
        for (int j = 0; j < BB_CHK_DEG; j++) Hz_mat[c][chk_vars[c][j]] = 1;
}
static int is_stabilizer(const uint8_t *res) {
    uint8_t mat[BB_NC][BB_NV]; memcpy(mat, Hz_mat, sizeof(mat));
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
        for (int j = 0; j < BB_CHK_DEG; j++) if (post[chk_vars[chk][j]] <= 0) s ^= 1;
        if (s != syn[chk]) return 0;
    }
    return 1;
}

/* ---- flooding (task_opt.c), ground truth / baseline ---- */
static inline fp_t ascale_opt(fp_t x){return(fp_t)((((int32_t)x<<2)+(int32_t)x)>>3);}
static inline fp_t fab16(fp_t x){return x<0?-x:x;}
static inline fp_t fp_sat16(int32_t x){return(fp_t)(x>FP_MAX?FP_MAX:(x<-FP_MAX?-FP_MAX:x));}

static void bp_decode_flooding(const uint8_t *syn, int32_t prior_in, int n_iter, fp_t *post) {
    fp_t prior = (fp_t)prior_in;
    fp_t v2c[BB_NNZ], c2v[BB_NNZ];
    for (int j=0;j<BB_NNZ;j++){v2c[j]=prior;c2v[j]=0;}
    for (int iter=0;iter<n_iter;iter++) {
        for (int chk=0;chk<BB_NC;chk++) {
            int b=chk*BB_CHK_DEG;
            fp_t m[BB_CHK_DEG];
            for (int j=0;j<BB_CHK_DEG;j++) m[j]=v2c[b+j];
            int sign=syn[chk]?1:0;
            for (int j=0;j<BB_CHK_DEG;j++) if(m[j]<0) sign^=1;
            fp_t a[BB_CHK_DEG];
            for (int j=0;j<BB_CHK_DEG;j++) a[j]=fab16(m[j]);
            fp_t mn1=FP_MAX,mn2=FP_MAX; int mp=0;
            for (int j=0;j<BB_CHK_DEG;j++){
                if(a[j]<=mn1){mn2=mn1;mn1=a[j];mp=j;}else if(a[j]<mn2)mn2=a[j];
            }
            for (int j=0;j<BB_CHK_DEG;j++){
                int s=sign^(m[j]<0?1:0);
                fp_t mn=(j==mp)?mn2:mn1;
                fp_t cv=ascale_opt(mn); if(s)cv=-cv; c2v[b+j]=cv;
            }
        }
        for (int var=0;var<BB_NV;var++){
            uint16_t i0=var_map[var][0],i1=var_map[var][1],i2=var_map[var][2];
            fp_t c0=c2v[i0],c1=c2v[i1],c2=c2v[i2];
            post[var]=fp_sat16((int32_t)prior+c0+c1+c2);
            v2c[i0]=fp_sat16((int32_t)prior+c1+c2);
            v2c[i1]=fp_sat16((int32_t)prior+c0+c2);
            v2c[i2]=fp_sat16((int32_t)prior+c0+c1);
        }
    }
}

/* ---- layered (task_layered.c): int32 c2v/lambda, q NOT pre-saturated, no
   first/final-iteration branching -- see task_layered.c's header comment for
   why (the branchy/double-saturating BP-PIM-style version measured SLOWER on
   real DPU cycles at iter=5 than this). alpha = 6/8 = 0.75 (the actual
   validated win from BP-PIM's tuning). ---- */
static inline int32_t ascale32(int32_t x){ return ((x<<1)+x) >> 2; }
static inline int32_t iabs32(int32_t x){ return x<0?-x:x; }
static inline int32_t sat32(int32_t x){ return x>FP_MAX?FP_MAX:(x<-FP_MAX?-FP_MAX:x); }

static void bp_decode_layered(const uint8_t *syn, int32_t prior, int n_iter, fp_t *post) {
    int32_t c2v[BB_NNZ];
    int32_t lambda[BB_NV];
    for (int v=0; v<BB_NV; v++) lambda[v] = prior;
    for (int j=0; j<BB_NNZ; j++) c2v[j] = 0;

    for (int iter=0; iter<n_iter; iter++) {
        for (int chk=0; chk<BB_NC; chk++) {
            int b = chk*BB_CHK_DEG;
            int32_t q[BB_CHK_DEG], a[BB_CHK_DEG];
            int sign = syn[chk] ? 1 : 0;
            for (int j=0;j<BB_CHK_DEG;j++) {
                uint16_t v = chk_vars[chk][j];
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
            for (int j=2;j<BB_CHK_DEG;j++) {
                if (a[j] <= mn1) { mn2=mn1; mn1=a[j]; mp=j; }
                else if (a[j] < mn2) mn2 = a[j];
            }
            for (int j=0;j<BB_CHK_DEG;j++) {
                uint16_t v = chk_vars[chk][j];
                int s = sign ^ (q[j] < 0 ? 1 : 0);
                int32_t mn = (j==mp) ? mn2 : mn1;
                int32_t cv = ascale32(mn);
                if (s) cv = -cv;
                c2v[b+j] = cv;
                lambda[v] = sat32(q[j] + cv);
            }
        }
    }
    for (int v=0; v<BB_NV; v++) post[v] = (fp_t)lambda[v];
}

typedef void (*decode_fn)(const uint8_t*, int32_t, int, fp_t*);

static void run_ler(const char *label, decode_fn fn, double p, fp_t prior, int n_iter,
                     long n_trials, unsigned int seed_base, int n_cores) {
    long n_fail = 0;
    #pragma omp parallel num_threads(n_cores) reduction(+:n_fail)
    {
        fp_t post[BB_NV]; uint8_t syn[BB_NC], error[BB_NV];
        #pragma omp for schedule(static)
        for (long t = 0; t < n_trials; t++) {
            unsigned int seed_x = seed_base + 9999999 + t*137;
            memset(error, 0, BB_NV);
            for (int i = 0; i < BB_NV; i++)
                if ((double)rand_r(&seed_x)/RAND_MAX < p) error[i] = 1;
            for (int chk = 0; chk < BB_NC; chk++) {
                uint8_t s = 0;
                for (int j = 0; j < BB_CHK_DEG; j++) s ^= error[chk_vars[chk][j]];
                syn[chk] = s;
            }
            fn(syn, prior, n_iter, post);
            uint8_t corr[BB_NV], res[BB_NV]; int rw = 0;
            for (int i = 0; i < BB_NV; i++) {
                corr[i] = (post[i] <= 0) ? 1 : 0;
                res[i] = error[i] ^ corr[i]; rw += res[i];
            }
            int fail = 0;
            if (rw > 0) {
                if (!check_conv(syn, post)) fail = 1;
                else if (!is_stabilizer(res)) fail = 1;
            }
            n_fail += fail;
        }
    }
    printf("  %-9s p=%.3f iter=%2d  LER(X only)=%.4e%%  (n=%ld, n_fail=%ld)\n",
           label, p, n_iter, (double)n_fail/n_trials*100.0, n_trials, n_fail);
}

int main(void) {
    build_pcm();
    build_hz();
    unsigned int seed_base = 424242;
    long n_trials = 2000000;
    int n_cores = omp_get_max_threads();
    double error_rates[] = {0.001, 0.003, 0.005, 0.007, 0.009};

    printf("BP schedule LER check: flooding (task_opt.c) vs layered (task_layered.c)\n");
    printf("n_trials=%ld per row, X-only LER, %d threads\n\n", n_trials, n_cores);

    for (int pi = 0; pi < 5; pi++) {
        double p = error_rates[pi];
        fp_t prior = (fp_t)(int)(log((1.0-p)/p)*256);
        printf("== p=%.3f  prior=%d ==\n", p, prior);
        for (int it = 1; it <= 4; it++) {
            run_ler("flooding", bp_decode_flooding, p, prior, it, n_trials, seed_base, n_cores);
        }
        for (int it = 1; it <= 3; it++) {
            run_ler("layered", bp_decode_layered, p, prior, it, n_trials, seed_base, n_cores);
        }
        printf("\n");
    }
    printf("Expect: layered@iter ~ flooding@(iter+1) or better, i.e. fewer iterations\n");
    printf("needed for the same LER -> fewer DPU cycles per single-tasklet decode.\n");
    return 0;
}
