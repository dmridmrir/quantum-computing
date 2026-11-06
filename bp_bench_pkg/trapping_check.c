#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

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
static fp_t v2c[BB_NNZ], c2v[BB_NNZ], post[BB_NV];

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

static void bp_decode_opt(const uint8_t *syn, fp_t prior, int n_iter) {
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

static int check_conv(const uint8_t *syn) {
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

int main(void) {
    build_pcm();

    double p = 0.005;
    fp_t prior = (fp_t)(int)(log((1.0-p)/p)*256);
    int n_trials = 200000;
    int seed_base = 42;

    printf("Trapping set analysis\n");
    printf("p=%.3f  prior=%d  n_trials=%d\n\n", p, prior, n_trials);
    printf("  %4s | %8s | %10s | %10s | %10s | %8s\n",
           "iter", "total_err", "type1(no_conv)", "type2(trap)", "trap_ratio", "LER%%");
    printf("  -----+----------+---------------+------------+------------+----------\n");

    for (int n_iter = 1; n_iter <= 12; n_iter++) {
        int n_err=0, type1=0, type2=0;
        for (int t=0;t<n_trials;t++){
            unsigned int seed=(unsigned int)(seed_base+t*137);
            uint8_t error[BB_NV], syn[BB_NC];
            memset(error,0,BB_NV);
            for (int i=0;i<BB_NV;i++)
                if((double)rand_r(&seed)/RAND_MAX < p) error[i]=1;
            for (int chk=0;chk<BB_NC;chk++){
                uint8_t s=0;
                for (int j=0;j<BB_CHK_DEG;j++) s^=error[chk_vars[chk][j]];
                syn[chk]=s;
            }
            bp_decode_opt(syn, prior, n_iter);

            uint8_t corr[BB_NV], res[BB_NV]; int rw=0;
            for (int i=0;i<BB_NV;i++){
                corr[i]=(post[i]<=0)?1:0;
                res[i]=error[i]^corr[i];
                rw+=res[i];
            }
            if (rw > 0) {
                if (!check_conv(syn)) { n_err++; type1++; }
                else if (!is_stabilizer(res)) { n_err++; type2++; }
            }
        }
        double ler = (double)n_err/n_trials*100.0;
        double trap_ratio = n_err>0 ? (double)type2/n_err*100.0 : 0.0;
        printf("  %4d | %8d | %13d | %10d | %9.1f%% | %7.3f%%\n",
               n_iter, n_err, type1, type2, trap_ratio, ler);
    }
    return 0;
}
