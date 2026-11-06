#!/bin/bash
# CPU tail-latency benchmark plus phase1 DPU aggregate summary for cooperative v1.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DPU_SUMMARY_FILE="${1:-$SCRIPT_DIR/tail_layered_dpu_summary.txt}"
META_FILE="${2:-$SCRIPT_DIR/tail_layered_meta.txt}"
SUPPORT="$SCRIPT_DIR/uPIMulator/benchmark/BPMS/support"
SEED_BASE=${SHARED_SEED:-$(date +%s)}
OUT_DIR="$SCRIPT_DIR/tail_latency_data_layered"
TAIL_TRIALS=${TAIL_TRIALS:-10000000}
SAVE_RAW_LATENCIES=${SAVE_RAW_LATENCIES:-0}
mkdir -p "$OUT_DIR"

case "$TAIL_TRIALS" in
    ''|*[!0-9]*|0) echo "TAIL_TRIALS must be a positive integer" >&2; exit 1 ;;
esac
case "$SAVE_RAW_LATENCIES" in
    0|1) ;;
    *) echo "SAVE_RAW_LATENCIES must be 0 or 1" >&2; exit 1 ;;
esac

if [ ! -f "$DPU_SUMMARY_FILE" ]; then
    echo "DPU summary file not found: $DPU_SUMMARY_FILE" >&2
    echo "run bp_tail_layered_phase1.sh first." >&2
    exit 1
fi

MIN_DPU_ITER=1
MAX_DPU_ITER=5
N_DPU_RUNS=10000
if [ -f "$META_FILE" ]; then
    # shellcheck disable=SC1090
    source "$META_FILE"
fi

echo ""
echo "================================================================================"
echo "  [Phase 2 only] BP Tail Latency CPU benchmark + DPU summary (v1 layered)"
echo "  DPU summary : $DPU_SUMMARY_FILE"
echo "  iter sweep  : $MIN_DPU_ITER..$MAX_DPU_ITER (from phase1 metadata)"
echo "  CPU trials  : $TAIL_TRIALS per (p,iter)"
echo "  DPU samples : $N_DPU_RUNS per (p,iter)"
echo "  raw CSV     : $([ "$SAVE_RAW_LATENCIES" = 1 ] && echo enabled || echo disabled)"
echo "================================================================================"
echo ""

cat > /tmp/bp_tail_layered_final.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "bb144_pcm.h"

#define FP_MAX 32000
#define N_WARMUP 2000
#define MAX_POINTS 64

typedef int16_t fp_t;
static volatile uint64_t benchmark_sink;

static const uint8_t check_colors[3][24] = {
    {1,4,6,9,14,17,19,22,24,27,32,35,37,40,42,45,50,53,55,58,60,63,68,71},
    {0,3,8,11,13,16,18,21,26,29,31,34,36,39,44,47,49,52,54,57,62,65,67,70},
    {2,5,7,10,12,15,20,23,25,28,30,33,38,41,43,46,48,51,56,59,61,64,66,69}
};
static const uint8_t color_order[3] = {1,2,0};

/* Cooperative v1 arithmetic and the three-color order used by task_layered.c. */
static int32_t c2v[BB_NNZ];
static int32_t lambda_[BB_NV];
static fp_t post[BB_NV];
static inline int32_t ascale32(int32_t x){ return ((x<<1)+x) >> 2; }
static inline int32_t iabs32(int32_t x){ return x<0?-x:x; }
static inline int32_t sat32(int32_t x){ return x>FP_MAX?FP_MAX:(x<-FP_MAX?-FP_MAX:x); }

static void bp_decode_layered(const uint8_t *syn, int32_t prior, int n_iter) {
    for (int v=0; v<BB_NV; v++) lambda_[v] = prior;
    for (int j=0; j<BB_NNZ; j++) c2v[j] = 0;

    for (int iter=0; iter<n_iter; iter++) {
        for (int cp=0; cp<3; cp++) {
            int color=color_order[cp];
            for (int pos=0; pos<24; pos++) {
                int chk=check_colors[color][pos];
                int b=chk*BB_CHK_DEG;
                int32_t q[BB_CHK_DEG], a[BB_CHK_DEG];
                int sign=syn[chk] ? 1 : 0;
                for (int j=0;j<BB_CHK_DEG;j++) {
                    uint16_t v=bb_chk_vars[chk][j];
                    int32_t qv=lambda_[v]-c2v[b+j];
                    q[j]=qv;
                    if (qv<0) sign^=1;
                    a[j]=iabs32(qv);
                }
                int32_t mn1=a[0], mn2=a[1]; int mp=0;
                if (mn2<=mn1) {
                    int32_t tmp=mn1;
                    mn1=mn2;
                    mn2=tmp;
                    mp=1;
                }
                for (int j=2;j<BB_CHK_DEG;j++) {
                    if (a[j]<=mn1) { mn2=mn1; mn1=a[j]; mp=j; }
                    else if (a[j]<mn2) mn2=a[j];
                }
                for (int j=0;j<BB_CHK_DEG;j++) {
                    uint16_t v=bb_chk_vars[chk][j];
                    int sign_out=sign ^ (q[j]<0 ? 1 : 0);
                    int32_t mn=(j==mp) ? mn2 : mn1;
                    int32_t cv=ascale32(mn);
                    if (sign_out) cv=-cv;
                    c2v[b+j]=cv;
                    lambda_[v]=q[j]+cv;
                }
            }
        }
    }
    for (int v=0;v<BB_NV;v++) post[v]=(fp_t)sat32(lambda_[v]);
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e6 + ts.tv_nsec/1e3;
}

static int cmp_double(const void *a, const void *b) {
    double da=*(const double*)a, db=*(const double*)b;
    return (da>db)-(da<db);
}

typedef struct {
    double mean, std, p50, p90, p99, p999, p9999, maximum;
} stats_t;

typedef struct {
    double p;
    int iter;
    int n_dpu;
    stats_t cpu;
    stats_t dpu;
} point_t;

static int percentile_index(int n, double q) {
    int index=(int)ceil(q*n)-1;
    if (index<0) index=0;
    if (index>=n) index=n-1;
    return index;
}

static stats_t calculate_stats(double *values, int n) {
    stats_t s={0};
    for (int i=0;i<n;i++) s.mean+=values[i];
    s.mean/=n;
    for (int i=0;i<n;i++) {
        double d=values[i]-s.mean;
        s.std+=d*d;
    }
    s.std=sqrt(s.std/n);
    qsort(values, (size_t)n, sizeof(double), cmp_double);
    s.p50=values[percentile_index(n,.50)];
    s.p90=values[percentile_index(n,.90)];
    s.p99=values[percentile_index(n,.99)];
    s.p999=values[percentile_index(n,.999)];
    s.p9999=values[percentile_index(n,.9999)];
    s.maximum=values[n-1];
    return s;
}

static int cmp_point(const void *a, const void *b) {
    const point_t *pa=(const point_t*)a, *pb=(const point_t*)b;
    if (pa->p<pb->p) return -1;
    if (pa->p>pb->p) return 1;
    return pa->iter-pb->iter;
}

static int find_point(point_t *points, int count, double p, int iter) {
    for (int i=0;i<count;i++)
        if (fabs(points[i].p-p)<1e-12 && points[i].iter==iter) return i;
    return -1;
}

static void print_stats_row(double p, int iter, int prior, const char *type,
                            const stats_t *s, const char *status) {
    printf("  %7.3f %4d %7d %-4s %10.6f %10.6f %10.6f %10.6f %10.6f "
           "%11.6f %12.6f %10.6f %7s\n",
           p,iter,prior,type,
           s->mean/1000.0,s->std/1000.0,s->p50/1000.0,s->p90/1000.0,
           s->p99/1000.0,s->p999/1000.0,s->p9999/1000.0,
           s->maximum/1000.0,status);
}

int main(int argc, char **argv) {
    if (argc<6) {
        fprintf(stderr,"Usage: %s <dpu_summary_file> <seed_base> <out_dir> <trials> <save_raw>\n",argv[0]);
        return 1;
    }
    unsigned int seed_base=(unsigned int)strtoul(argv[2],NULL,10);
    const char *out_dir=argv[3];
    int n_trials=atoi(argv[4]);
    int save_raw=atoi(argv[5]);
    if (n_trials<=0 || (save_raw!=0 && save_raw!=1)) {
        fprintf(stderr,"invalid trials/save_raw argument\n");
        return 1;
    }

    point_t points[MAX_POINTS];
    memset(points,0,sizeof(points));
    int point_count=0;
    FILE *cf=fopen(argv[1],"r");
    if (!cf) { perror("DPU summary"); return 1; }
    char line[256];
    int line_number=0;
    while (fgets(line,sizeof(line),cf)) {
        line_number++;
        double p,mean_ms,std_ms,p50_ms,p90_ms,p99_ms,p999_ms,p9999_ms,max_ms;
        int iter,n_dpu; char extra;
        int fields=sscanf(line," %lf %d %d %lf %lf %lf %lf %lf %lf %lf %lf %c",
                          &p,&iter,&n_dpu,&mean_ms,&std_ms,&p50_ms,&p90_ms,
                          &p99_ms,&p999_ms,&p9999_ms,&max_ms,&extra);
        if (fields!=11 || n_dpu<=0) {
            fprintf(stderr,"invalid DPU summary line %d; rerun phase1\n",line_number);
            fclose(cf); return 1;
        }
        int idx=find_point(points,point_count,p,iter);
        if (idx>=0) {
            fprintf(stderr,"duplicate DPU summary point at line %d\n",line_number);
            fclose(cf); return 1;
        }
        if (point_count>=MAX_POINTS) { fprintf(stderr,"too many sweep points\n"); fclose(cf); return 1; }
        idx=point_count++;
        points[idx].p=p;
        points[idx].iter=iter;
        points[idx].n_dpu=n_dpu;
        points[idx].dpu.mean=mean_ms*1000.0;
        points[idx].dpu.std=std_ms*1000.0;
        points[idx].dpu.p50=p50_ms*1000.0;
        points[idx].dpu.p90=p90_ms*1000.0;
        points[idx].dpu.p99=p99_ms*1000.0;
        points[idx].dpu.p999=p999_ms*1000.0;
        points[idx].dpu.p9999=p9999_ms*1000.0;
        points[idx].dpu.maximum=max_ms*1000.0;
    }
    fclose(cf);
    if (point_count==0) { fprintf(stderr,"no DPU samples\n"); return 1; }
    qsort(points,(size_t)point_count,sizeof(point_t),cmp_point);

    fprintf(stderr,"Measuring %d CPU tail samples for each of %d (p,iter) points...\n",
            n_trials,point_count);

    for (int pi=0;pi<point_count;pi++) {
        point_t *pt=&points[pi];
        int32_t prior=(int32_t)(int)(log((1.0-pt->p)/pt->p)*256);
        uint8_t syn[BB_NC],error[BB_NV];

        for (int w=0;w<N_WARMUP;w++) {
            unsigned int seed=seed_base+77777+w;
            memset(error,0,BB_NV);
            for (int i=0;i<BB_NV;i++) if ((double)rand_r(&seed)/RAND_MAX<pt->p) error[i]=1;
            for (int chk=0;chk<BB_NC;chk++) {
                uint8_t v=0;
                for (int j=0;j<BB_CHK_DEG;j++) v^=error[bb_chk_vars[chk][j]];
                syn[chk]=v;
            }
            bp_decode_layered(syn,prior,pt->iter);
        }

        double *cpu_lats=malloc((size_t)n_trials*sizeof(double));
        if (!cpu_lats) { perror("malloc cpu_lats"); return 1; }
        for (int t=0;t<n_trials;t++) {
            unsigned int seed=seed_base+t*137;
            memset(error,0,BB_NV);
            for (int i=0;i<BB_NV;i++) if ((double)rand_r(&seed)/RAND_MAX<pt->p) error[i]=1;
            for (int chk=0;chk<BB_NC;chk++) {
                uint8_t v=0;
                for (int j=0;j<BB_CHK_DEG;j++) v^=error[bb_chk_vars[chk][j]];
                syn[chk]=v;
            }
            double t0=now_us();
            bp_decode_layered(syn,prior,pt->iter);
            benchmark_sink^=(uint16_t)post[t%BB_NV];
            double t1=now_us();
            cpu_lats[t]=(t1-t0)*2.0;
            if ((t+1)%1000000==0 || t+1==n_trials) {
                fprintf(stderr,"\r  [%2d/%2d] CPU p=%.3f iter=%d: %3.0f%%",
                        pi+1,point_count,pt->p,pt->iter,100.0*(t+1)/n_trials);
                fflush(stderr);
            }
        }
        fprintf(stderr,"\n");

        if (save_raw) {
            char path[512];
            snprintf(path,sizeof(path),"%s/cpu_lats_p%.3f_iter%d.csv",out_dir,pt->p,pt->iter);
            FILE *fp=fopen(path,"w");
            if (fp) {
                fprintf(fp,"latency_us\n");
                for (int t=0;t<n_trials;t++) fprintf(fp,"%.6f\n",cpu_lats[t]);
                fclose(fp);
            }
        }
        pt->cpu=calculate_stats(cpu_lats,n_trials);
        free(cpu_lats);

    }

    printf("\n==================================================================================================================================================\n");
    printf("  BP Tail Latency Summary: cooperative v1, 11 tasklets -> 1 syndrome, iter sweep\n");
    printf("  Metric: serialized X+Z decode latency = 2 x one-component compute latency; all values below are milliseconds\n");
    printf("  CPU samples/(p,iter): %d; DPU samples/(p,iter): aggregate from phase1; DPU clock: 350 MHz; raw CSV: %s\n",
           n_trials,save_raw?"enabled":"disabled");
    printf("==================================================================================================================================================\n");
    printf("        p iter   prior type    mean ms     std ms     p50 ms     p90 ms     p99 ms   p99.9 ms   p99.99 ms     max ms    <1ms\n");
    printf("  ------- ---- ------- ---- ---------- ---------- ---------- ---------- ---------- ----------- ------------ ---------- -------\n");
    double previous_p=-1.0;
    for (int pi=0;pi<point_count;pi++) {
        point_t *pt=&points[pi];
        if (previous_p>=0.0 && fabs(previous_p-pt->p)>1e-12)
            printf("  ------- ---- ------- ---- ---------- ---------- ---------- ---------- ---------- ----------- ------------ ---------- -------\n");
        previous_p=pt->p;
        int prior=(int)(log((1.0-pt->p)/pt->p)*256);
        print_stats_row(pt->p,pt->iter,prior,"CPU",&pt->cpu,
                        pt->cpu.maximum<1000.0?"PASS":"FAIL");
        print_stats_row(pt->p,pt->iter,prior,"DPU",&pt->dpu,
                        pt->dpu.maximum<1000.0?"PASS":"FAIL");
    }
    printf("==================================================================================================================================================\n");
    printf("  <1ms uses the maximum sampled serialized X+Z latency. With few DPU samples, p99/p99.9/p99.99 may equal the sampled maximum.\n");
    printf("  DPU result excludes host transfer, runtime launch, program load, and controller integration.\n");
    printf("==================================================================================================================================================\n\n");
    return benchmark_sink==UINT64_C(0xdeadbeef)?1:0;
}
CEOF

gcc -O2 -I"$SUPPORT" -o /tmp/bp_tail_layered_final /tmp/bp_tail_layered_final.c -lm
/tmp/bp_tail_layered_final "$DPU_SUMMARY_FILE" "$SEED_BASE" "$OUT_DIR" "$TAIL_TRIALS" "$SAVE_RAW_LATENCIES"

echo ""
if [ "$SAVE_RAW_LATENCIES" = 1 ]; then
    echo "Raw latency data saved to: $OUT_DIR"
else
    echo "Raw latency CSV disabled; summary statistics only."
fi
