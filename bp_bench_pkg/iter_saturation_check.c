// iter_saturation_check.c
// Diagnose why LER saturates from iter=3 to iter=10 at p=0.001.
// For each sampled error (same seed formula as run_bp_increase.sh's LER phase),
// decode with n_iter=3 and n_iter=10 using the *exact* fixed-iteration opt decoder,
// and classify:
//   - whether post[] is bit-identical between iter=3 and iter=10 (BP itself converged/stalled)
//   - whether the failure (if any) is type1 (syndrome not satisfied => not converged)
//     or type2 (syndrome satisfied but residual is a nontrivial stabilizer-equivalent
//     logical error => converged to a wrong codeword / degenerate trapping set)
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
static uint8_t  Hz_mat[BB_NC][BB_NV];

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
    memset(Hz_mat,0,sizeof(Hz_mat));
    for (int c=0;c<BB_NC;c++)
        for (int j=0;j<BB_CHK_DEG;j++) Hz_mat[c][chk_vars[c][j]]=1;
}

static inline fp_t ascale_opt(fp_t x){return(fp_t)((((int32_t)x<<2)+(int32_t)x)>>3);}
static inline fp_t fab(fp_t x){return x<0?-x:x;}
static inline fp_t fp_sat(int32_t x){return(fp_t)(x>FP_MAX?FP_MAX:(x<-FP_MAX?-FP_MAX:x));}

static void bp_decode_opt(const uint8_t *syn, fp_t prior, int n_iter,
                           fp_t *v2c, fp_t *c2v, fp_t *post) {
    for (int j=0;j<BB_NNZ;j++){v2c[j]=prior;c2v[j]=0;}
    for (int iter=0;iter<n_iter;iter++) {
        for (int chk=0;chk<BB_NC;chk++) {
            int b=chk*BB_CHK_DEG;
            fp_t m[BB_CHK_DEG];
            for (int j=0;j<BB_CHK_DEG;j++) m[j]=v2c[b+j];
            int sign=syn[chk]?1:0;
            for (int j=0;j<BB_CHK_DEG;j++) if (m[j]<0) sign^=1;
            fp_t a[BB_CHK_DEG];
            for (int j=0;j<BB_CHK_DEG;j++) a[j]=fab(m[j]);
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
            post[var]=fp_sat((int32_t)prior+c0+c1+c2);
            v2c[i0]=fp_sat((int32_t)prior+c1+c2);
            v2c[i1]=fp_sat((int32_t)prior+c0+c2);
            v2c[i2]=fp_sat((int32_t)prior+c0+c1);
        }
    }
}

static int check_conv(const uint8_t *syn, const fp_t *post) {
    for (int chk=0;chk<BB_NC;chk++){
        uint8_t s=0;
        for (int j=0;j<BB_CHK_DEG;j++) if(post[chk_vars[chk][j]]<=0) s^=1;
        if(s!=syn[chk]) return 0;
    }
    return 1;
}

static int is_stabilizer(const uint8_t *res) {
    uint8_t mat[BB_NC][BB_NV]; memcpy(mat,Hz_mat,sizeof(mat));
    uint8_t tgt[BB_NV]; memcpy(tgt,res,BB_NV); int pr=0;
    for (int col=0;col<BB_NV&&pr<BB_NC;col++){
        int f=-1;
        for (int r=pr;r<BB_NC;r++) if(mat[r][col]){f=r;break;}
        if(f<0) continue;
        if(f!=pr) for(int c=0;c<BB_NV;c++){uint8_t t=mat[pr][c];mat[pr][c]=mat[f][c];mat[f][c]=t;}
        if(tgt[col]) for(int c=0;c<BB_NV;c++) tgt[c]^=mat[pr][c];
        for(int r=0;r<BB_NC;r++) if(r!=pr&&mat[r][col]) for(int c=0;c<BB_NV;c++) mat[r][c]^=mat[pr][c];
        pr++;
    }
    for (int i=0;i<BB_NV;i++) if(tgt[i]) return 0;
    return 1;
}

int main(int argc, char **argv) {
    build_pcm();

    double p = argc > 1 ? atof(argv[1]) : 0.001;
    long long n_trials = argc > 2 ? atoll(argv[2]) : 100000000LL;
    unsigned int seed_base = argc > 3 ? (unsigned int)atol(argv[3]) : 42;
    int iter_lo = argc > 4 ? atoi(argv[4]) : 3;
    int iter_hi = argc > 5 ? atoi(argv[5]) : 10;

    fp_t prior = (fp_t)(int)(log((1.0-p)/p)*256);

    printf("p=%.4f prior=%d n_trials=%lld iter_lo=%d iter_hi=%d\n",
           p, prior, n_trials, iter_lo, iter_hi);

    long long n_fail_lo=0, type1_lo=0, type2_lo=0;
    long long n_fail_hi=0, type1_hi=0, type2_hi=0;
    long long n_post_identical=0;      // post[] bit-identical between iter_lo and iter_hi
    long long n_post_diff_same_outcome=0;
    long long n_post_diff_diff_outcome=0;
    // transition breakdown: for trials where hi still fails, what was lo's state?
    long long hi_fail_lo_type1=0, hi_fail_lo_type2=0, hi_fail_lo_success=0;

    #pragma omp parallel for schedule(static) reduction(+:n_fail_lo,type1_lo,type2_lo,n_fail_hi,type1_hi,type2_hi,n_post_identical,n_post_diff_same_outcome,n_post_diff_diff_outcome,hi_fail_lo_type1,hi_fail_lo_type2,hi_fail_lo_success)
    for (long long t=0; t<n_trials; t++) {
        unsigned int seed = (unsigned int)(seed_base + 9999999 + t*137);
        uint8_t error[BB_NV], syn[BB_NC];
        memset(error,0,BB_NV);
        for (int i=0;i<BB_NV;i++)
            if ((double)rand_r(&seed)/RAND_MAX < p) error[i]=1;
        for (int chk=0;chk<BB_NC;chk++){
            uint8_t s=0;
            for (int j=0;j<BB_CHK_DEG;j++) s^=error[chk_vars[chk][j]];
            syn[chk]=s;
        }

        fp_t v2c[BB_NNZ], c2v[BB_NNZ], post_lo[BB_NV], post_hi[BB_NV];

        bp_decode_opt(syn, prior, iter_lo, v2c, c2v, post_lo);
        bp_decode_opt(syn, prior, iter_hi, v2c, c2v, post_hi);

        int identical = (memcmp(post_lo, post_hi, sizeof(post_lo)) == 0);
        if (identical) n_post_identical++;

        // classify lo
        uint8_t corr[BB_NV], res[BB_NV]; int rw;
        int lo_fail=0, lo_type=0; // 0=success,1=type1(non-conv),2=type2(trap)
        rw=0;
        for (int i=0;i<BB_NV;i++){ corr[i]=(post_lo[i]<=0)?1:0; res[i]=error[i]^corr[i]; rw+=res[i]; }
        if (rw>0) {
            if (!check_conv(syn, post_lo)) { lo_fail=1; lo_type=1; }
            else if (!is_stabilizer(res)) { lo_fail=1; lo_type=2; }
        }
        if (lo_fail) { n_fail_lo++; if(lo_type==1) type1_lo++; else type2_lo++; }

        int hi_fail=0, hi_type=0;
        rw=0;
        for (int i=0;i<BB_NV;i++){ corr[i]=(post_hi[i]<=0)?1:0; res[i]=error[i]^corr[i]; rw+=res[i]; }
        if (rw>0) {
            if (!check_conv(syn, post_hi)) { hi_fail=1; hi_type=1; }
            else if (!is_stabilizer(res)) { hi_fail=1; hi_type=2; }
        }
        if (hi_fail) { n_fail_hi++; if(hi_type==1) type1_hi++; else type2_hi++; }

        if (!identical) {
            if ((lo_fail!=0) == (hi_fail!=0) && lo_type==hi_type) n_post_diff_same_outcome++;
            else n_post_diff_diff_outcome++;
        }

        if (hi_fail) {
            if (!lo_fail) hi_fail_lo_success++;
            else if (lo_type==1) hi_fail_lo_type1++;
            else hi_fail_lo_type2++;
        }
    }

    double ler_lo = (double)n_fail_lo/n_trials;
    double ler_hi = (double)n_fail_hi/n_trials;

    printf("\n=== iter=%d ===\n", iter_lo);
    printf("  n_fail=%lld  LER=%.6e  type1(non-conv)=%lld (%.1f%%)  type2(trap/degenerate)=%lld (%.1f%%)\n",
           n_fail_lo, ler_lo, type1_lo, n_fail_lo? 100.0*type1_lo/n_fail_lo:0.0,
           type2_lo, n_fail_lo? 100.0*type2_lo/n_fail_lo:0.0);

    printf("\n=== iter=%d ===\n", iter_hi);
    printf("  n_fail=%lld  LER=%.6e  type1(non-conv)=%lld (%.1f%%)  type2(trap/degenerate)=%lld (%.1f%%)\n",
           n_fail_hi, ler_hi, type1_hi, n_fail_hi? 100.0*type1_hi/n_fail_hi:0.0,
           type2_hi, n_fail_hi? 100.0*type2_hi/n_fail_hi:0.0);

    printf("\n=== post[] identity between iter=%d and iter=%d ===\n", iter_lo, iter_hi);
    printf("  identical      : %lld / %lld (%.6f%%)\n", n_post_identical, n_trials, 100.0*n_post_identical/n_trials);
    printf("  differ,same outcome: %lld\n", n_post_diff_same_outcome);
    printf("  differ,diff outcome: %lld\n", n_post_diff_diff_outcome);

    printf("\n=== of the iter=%d failures, what were they at iter=%d? ===\n", iter_hi, iter_lo);
    printf("  already success at iter=%d (newly broke by more iters): %lld\n", iter_lo, hi_fail_lo_success);
    printf("  already type1 (non-conv) at iter=%d                    : %lld\n", iter_lo, hi_fail_lo_type1);
    printf("  already type2 (trap/degenerate) at iter=%d             : %lld\n", iter_lo, hi_fail_lo_type2);

    return 0;
}
