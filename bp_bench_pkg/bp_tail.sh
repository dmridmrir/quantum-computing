#!/bin/bash
# bp_tail.sh — Tail latency distribution: CPU vs DPU
# iter=10 fixed, p=0.001 and p=0.009, N_DPU_RUNS runs per p

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/uPIMulator"
BPMS_GO="$ROOT/src/assembler/prim/bpms.go"
TASK_C="$ROOT/benchmark/BPMS/dpu/task.c"
NR_TASKLETS=1
DPU_ITER=5
ERROR_RATES="0.001 0.009"
N_DPU_RUNS=15
SEED_BASE=${SHARED_SEED:-$(date +%s)}
OUT_DIR="$SCRIPT_DIR/tail_latency_data"
mkdir -p "$OUT_DIR"

echo ""
echo "================================================================================"
echo "  BP Tail Latency Analysis: CPU vs DPU"
echo "  iter=$DPU_ITER (fixed), p in {$ERROR_RATES}, DPU runs=$N_DPU_RUNS"
echo "  Seed base: $SEED_BASE"
echo "================================================================================"
echo ""

# ---- 백업 + 복원 ----
cp "$TASK_C" "$TASK_C.backup"
cp "$BPMS_GO" "$BPMS_GO.backup"
for f in "$ROOT/benchmark/BPMS/dpu/task_opt.c" \
         "$ROOT/benchmark/BPMS/dpu/task_noopt.c" \
         "$ROOT/benchmark/BPMS/dpu/task_adaptive.c"; do
    test -f "$f" && mv "$f" "$f.bak" 2>/dev/null || true
done

cleanup() {
    cp "$TASK_C.backup" "$TASK_C" 2>/dev/null || true
    cp "$BPMS_GO.backup" "$BPMS_GO" 2>/dev/null || true
    rm -f "$TASK_C.backup" "$BPMS_GO.backup"
    for f in "$ROOT/benchmark/BPMS/dpu/task_opt.c" \
             "$ROOT/benchmark/BPMS/dpu/task_noopt.c" \
             "$ROOT/benchmark/BPMS/dpu/task_adaptive.c"; do
        test -f "$f.bak" && mv "$f.bak" "$f" 2>/dev/null || true
    done
}
trap cleanup EXIT

cp "$ROOT/benchmark/BPMS/dpu/task_opt.c.bak" "$TASK_C"

# ---- Phase 1: DPU 시뮬레이션 (N_DPU_RUNS runs per p) ----
echo "[Phase 1] DPU cycle distribution ($N_DPU_RUNS runs per p)"
echo ""

DPU_CYCLES_FILE=$(mktemp)

for P in $ERROR_RATES; do
    PRIOR=$(python3 -c "import math; print(int(math.log((1-$P)/$P)*256))")
    echo "  p=$P  prior=$PRIOR"

    for RUN in $(seq 1 $N_DPU_RUNS); do
        SEED=$((SEED_BASE + RUN * 997))

        python3 << PYEOF
import re
with open('$TASK_C') as f: s = f.read()
s = re.sub(r'decode\(tid, [0-9]+, [0-9]+\)', 'decode(tid, $PRIOR, $DPU_ITER)', s)
with open('$TASK_C', 'w') as f: f.write(s)
PYEOF

        cat > "$BPMS_GO" << GOEOF
package prim

import (
	"errors"
	"math/rand"
	"uPIMulator/src/abi/encoding"
	"uPIMulator/src/abi/word"
	"uPIMulator/src/misc"
)

type Bpms struct {
	num_dpus       int
	num_tasklets   int
	num_executions int
	nc             int64
	nv             int64
	syndromes      [][]byte
}

func (this *Bpms) Init(command_line_parser *misc.CommandLineParser) {
	num_channels := int(command_line_parser.IntParameter("num_channels"))
	num_ranks_per_channel := int(command_line_parser.IntParameter("num_ranks_per_channel"))
	num_dpus_per_rank := int(command_line_parser.IntParameter("num_dpus_per_rank"))
	this.num_dpus = num_channels * num_ranks_per_channel * num_dpus_per_rank
	this.num_tasklets = int(command_line_parser.IntParameter("num_tasklets"))
	this.num_executions = 1
	this.nc = 72
	this.nv = 144
	this.syndromes = make([][]byte, this.num_tasklets)
	for t := 0; t < this.num_tasklets; t++ {
		rng := rand.New(rand.NewSource(int64($SEED + t*137)))
		syn := make([]byte, this.nc)
		for i := int64(0); i < this.nc; i++ {
			if rng.Float64() < $P {
				syn[i] = 1
			}
		}
		padded := ((int(this.nc) + 7) / 8) * 8
		for len(syn) < padded { syn = append(syn, 0) }
		this.syndromes[t] = syn
	}
}

func (this *Bpms) InputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	if execution >= this.num_executions { panic(errors.New("execution >= num executions")) }
	bs := new(encoding.ByteStream); bs.Init()
	for _, v := range []int64{0, this.nc, this.nv, 0, 0, 0, 0} {
		w := new(word.Word); w.Init(32); w.SetValue(v); bs.Merge(w.ToByteStream())
	}
	dpu_host := make(map[string]*encoding.ByteStream)
	dpu_host["DPU_INPUT_ARGUMENTS"] = bs
	return dpu_host
}

func (this *Bpms) OutputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	return make(map[string]*encoding.ByteStream)
}

func (this *Bpms) InputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	bs := new(encoding.ByteStream); bs.Init()
	for t := 0; t < this.num_tasklets; t++ {
		for _, b := range this.syndromes[t] {
			w := new(word.Word); w.Init(8); w.SetValue(int64(b)); bs.Merge(w.ToByteStream())
		}
	}
	return 0, bs
}

func (this *Bpms) OutputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	bs := new(encoding.ByteStream); bs.Init()
	prior := int16($PRIOR)
	nIter := $DPU_ITER
	for t := 0; t < this.num_tasklets; t++ {
		post, _ := bpDecode(this.syndromes[t], prior, nIter)
		for i := 0; i < int(this.nv); i++ {
			val := post[i]
			lo := new(word.Word); lo.Init(8); lo.SetValue(int64(uint8(val & 0xFF))); bs.Merge(lo.ToByteStream())
			hi := new(word.Word); hi.Init(8); hi.SetValue(int64(uint8((val >> 8) & 0xFF))); bs.Merge(hi.ToByteStream())
		}
	}
	slotSize := ((int(this.nc) + 7) / 8) * 8
	return int64(this.num_tasklets * slotSize), bs
}

func (this *Bpms) NumExecutions() int { return this.num_executions }
GOEOF

        cd "$ROOT/src" && go build -o "$ROOT/build/uPIMulator" . 2>/dev/null
        cd "$ROOT/benchmark" && rm -rf build && mkdir build
        cmake -D NR_DPUS=1 -D NR_TASKLETS=$NR_TASKLETS -S . -B build -G Ninja >/dev/null 2>&1
        ninja -C build BPMS_device >/dev/null 2>&1

        rm -rf "$ROOT/bin" && mkdir -p "$ROOT/bin"
        timeout 600 "$ROOT/build/uPIMulator" \
            --root_dirpath "$ROOT" --bin_dirpath "$ROOT/bin" \
            --benchmark BPMS \
            --num_channels 1 --num_ranks_per_channel 1 --num_dpus_per_rank 1 \
            --num_tasklets $NR_TASKLETS --data_prep_params 8 \
            --num_simulation_threads 4 >/dev/null 2>&1

        CYCLES=$(grep "logic_cycle" "$ROOT/bin/log.txt" 2>/dev/null | sed 's/.*: //')
        echo "    run=$RUN  cycles=$CYCLES"
        echo "$P $CYCLES" >> "$DPU_CYCLES_FILE"
    done
    echo ""
done

# ---- Phase 2: CPU + 결과 출력 ----
echo "[Phase 2] CPU tail latency + combined results"
echo ""

cat > /tmp/bp_tail_final.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define BB_NC 72
#define BB_NV 144
#define BB_NNZ 432
#define BB_CHK_DEG 6
#define BB_VAR_DEG 3
#define FP_MAX 32000
#define N_TRIALS 10000000
#define N_WARMUP 2000
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

static fp_t v2c[BB_NNZ], c2v[BB_NNZ], post[BB_NV];
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
            for (int j=0;j<BB_CHK_DEG;j++) if(m[j]<0) sign^=1;
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

static double now_us(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e6+ts.tv_nsec/1e3;}

static int cmp_double(const void *a, const void *b){
    double da=*(double*)a, db=*(double*)b;
    return (da>db)-(da<db);
}

static void print_percentiles(double *sorted, int n, const char *label) {
    double mean=0, var=0;
    for (int i=0;i<n;i++) mean+=sorted[i];
    mean/=n;
    for (int i=0;i<n;i++) var+=(sorted[i]-mean)*(sorted[i]-mean);
    double std=sqrt(var/n);
    printf("  %-8s | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f\n",
        label,
        mean,
        std,
        sorted[(int)(n*0.50)],
        sorted[(int)(n*0.90)],
        sorted[(int)(n*0.99)],
        sorted[(int)(n*0.999)],
        sorted[(int)(n*0.9999)],
        sorted[n-1]);
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "Usage: %s <cycles_file> <seed_base> <iter> <out_dir>\n", argv[0]); return 1; }
    build_pcm();
    unsigned int seed_base = (unsigned int)atol(argv[2]);
    int n_iter = atoi(argv[3]);
    const char *out_dir = argv[4];
    double DPU_MHZ = 350.0;

    /* DPU cycles 읽기 */
    double p_list[2]; int p_count = 0;
    double dpu_cyc[2][200]; int dpu_cnt[2] = {0,0};
    FILE *cf = fopen(argv[1], "r");
    if (!cf) { perror("cycles"); return 1; }
    double p_val; double cyc_val;
    while (fscanf(cf, "%lf %lf", &p_val, &cyc_val) == 2) {
        int idx = -1;
        for (int i=0;i<p_count;i++) if (fabs(p_list[i]-p_val)<1e-6) {idx=i;break;}
        if (idx < 0 && p_count < 2) { p_list[p_count]=p_val; idx=p_count++; }
        if (idx >= 0) dpu_cyc[idx][dpu_cnt[idx]++] = cyc_val;
    }
    fclose(cf);

    printf("\n================================================================================\n");
    printf("  BP Tail Latency: CPU vs DPU  (iter=%d fixed, NMS opt)\n", n_iter);
    printf("================================================================================\n");
    printf("  * CPU: %d individual decode measurements\n", N_TRIALS);
    printf("  * DPU: cycle-accurate simulation, E2E per decode (X+Z x2, compute only), single tasklet allocated on DPU (no batching, revolver stalls fully exposed)\n");
    printf("\n");
    printf("  %-8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s\n",
           "type", "mean", "std", "50%", "90%", "99%", "99.9%", "99.99%", "max");
    printf("  ---------+----------+----------+----------+----------+----------+----------+----------+----------\n");

    for (int pi=0; pi<p_count; pi++) {
        double p = p_list[pi];
        fp_t prior = (fp_t)(int)(log((1.0-p)/p)*256);

        printf("\n  [p=%.3f  prior=%d]\n", p, prior);

        /* CPU 측정 */
        uint8_t syn[BB_NC], error[BB_NV];
        for (int w=0;w<N_WARMUP;w++){
            unsigned int seed=seed_base+77777+w; memset(error,0,BB_NV);
            for(int i=0;i<BB_NV;i++) if((double)rand_r(&seed)/RAND_MAX<p) error[i]=1;
            for(int chk=0;chk<BB_NC;chk++){uint8_t s=0;for(int j=0;j<BB_CHK_DEG;j++)s^=error[chk_vars[chk][j]];syn[chk]=s;}
            bp_decode_opt(syn,prior,n_iter);
        }

        double *cpu_lats = malloc(N_TRIALS * sizeof(double));
        for (int t=0;t<N_TRIALS;t++){
            unsigned int seed=seed_base+t*137; memset(error,0,BB_NV);
            for(int i=0;i<BB_NV;i++) if((double)rand_r(&seed)/RAND_MAX<p) error[i]=1;
            for(int chk=0;chk<BB_NC;chk++){uint8_t s=0;for(int j=0;j<BB_CHK_DEG;j++)s^=error[chk_vars[chk][j]];syn[chk]=s;}
            double t0=now_us(); bp_decode_opt(syn,prior,n_iter); double t1=now_us();
            cpu_lats[t]=(t1-t0)*2.0;
        }

        /* CPU 원시 데이터 저장 */
        char cpu_path[512];
        snprintf(cpu_path, sizeof(cpu_path), "%s/cpu_lats_p%.3f.csv", out_dir, p);
        FILE *cpu_fp = fopen(cpu_path, "w");
        if (cpu_fp) {
            fprintf(cpu_fp, "latency_us\n");
            for (int t=0; t<N_TRIALS; t++) fprintf(cpu_fp, "%.6f\n", cpu_lats[t]);
            fclose(cpu_fp);
        }

        qsort(cpu_lats, N_TRIALS, sizeof(double), cmp_double);
        print_percentiles(cpu_lats, N_TRIALS, "CPU");
        free(cpu_lats);

        /* DPU E2E 계산 */
        int n = dpu_cnt[pi];
        if (n > 0) {
            double *dpu_lats = malloc(n * sizeof(double));
            for (int i=0;i<n;i++)
                dpu_lats[i] = 2.0 * (dpu_cyc[pi][i]/DPU_MHZ);

            /* DPU 원시 데이터 저장 */
            char dpu_path[512];
            snprintf(dpu_path, sizeof(dpu_path), "%s/dpu_lats_p%.3f.csv", out_dir, p);
            FILE *dpu_fp = fopen(dpu_path, "w");
            if (dpu_fp) {
                fprintf(dpu_fp, "latency_us\n");
                for (int i=0; i<n; i++) fprintf(dpu_fp, "%.6f\n", dpu_lats[i]);
                fclose(dpu_fp);
            }

            qsort(dpu_lats, n, sizeof(double), cmp_double);
            print_percentiles(dpu_lats, n, "DPU");
            free(dpu_lats);
        }
    }

    printf("\n================================================================================\n\n");
    return 0;
}
CEOF

gcc -O2 -o /tmp/bp_tail_final /tmp/bp_tail_final.c -lm
/tmp/bp_tail_final "$DPU_CYCLES_FILE" "$SEED_BASE" "$DPU_ITER" "$OUT_DIR"

rm -f "$DPU_CYCLES_FILE"

echo "Raw latency data saved to: $OUT_DIR"
echo "  CPU: cpu_lats_p{P}.csv  (${N_TRIALS} rows)"
echo "  DPU: dpu_lats_p{P}.csv  ($N_DPU_RUNS rows, E2E per-decode us)"
