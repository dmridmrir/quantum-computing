#!/bin/bash
# BP Throughput Benchmark: DPU (uPIMulator) + CPU 실측
#
# 실행 방법:
#   ./run_bp_bench.sh          → 비최적화 (곱셈 + 수렴기반 max 20회)
#   ./run_opt_bp_bench.sh      → 최적화 (shift+add + 고정 3회)
#   ./run_adaptive_bp_bench.sh → 적응 (shift+add + prior 기반 3/5회)
#
# 파일명으로 모드 자동 감지

set -e

# ---- 모드 감지 ----
SCRIPT_NAME=$(basename "$0")
case "$SCRIPT_NAME" in
    run_opt_bp_bench.sh)      MODE="opt" ;;
    run_adaptive_bp_bench.sh) MODE="adaptive" ;;
    *)                        MODE="noopt" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/uPIMulator"
BPMS_GO="$ROOT/src/assembler/prim/bpms.go"
TASK_C="$ROOT/benchmark/BPMS/dpu/task.c"
NR_TASKLETS=11
ERROR_RATES="0.001 0.003 0.005 0.007 0.01 0.02 0.03"
SEED_BASE=${SHARED_SEED:-$(date +%s)}

# ---- 모드별 설명 ----
case "$MODE" in
    noopt)
        LABEL="Non-optimized: multiplication alpha + convergence-based (max 20 iter)"
        ITER_DESC="max 20 (convergence-based, early termination)"
        ;;
    opt)
        LABEL="Optimized: shift+add alpha + fixed 3 iterations"
        ITER_DESC="3 (fixed, no early termination)"
        ;;
    adaptive)
        LABEL="Adaptive: shift+add alpha + prior-based 3/5 iterations"
        ITER_DESC="3 (p<=0.005) or 5 (p>0.005), fixed per error rate"
        ;;
esac

# ---- 백업 + 종료 시 복원 ----
cp "$TASK_C" "$TASK_C.backup"
cp "$BPMS_GO" "$BPMS_GO.backup"
# 빌드 충돌 방지: task_*.c 임시 이동
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

# ---- task.c 교체 ----
case "$MODE" in
    noopt)    cp "$ROOT/benchmark/BPMS/dpu/task_noopt.c.bak" "$TASK_C" ;;
    opt)      cp "$ROOT/benchmark/BPMS/dpu/task_opt.c.bak" "$TASK_C" ;;
    adaptive) cp "$ROOT/benchmark/BPMS/dpu/task_adaptive.c.bak" "$TASK_C" ;;
esac

echo ""
echo "================================================================================"
echo "  BP Throughput Benchmark ($MODE)"
echo "  $LABEL"
echo "  Seed base: $SEED_BASE (매 실행마다 다른 syndrome)"
echo "================================================================================"
echo ""

# ---- Phase 1: DPU 시뮬레이션 ----
echo "[Phase 1] DPU uPIMulator cycle-accurate simulation"
echo ""

CYCLES_FILE=$(mktemp)

for P in $ERROR_RATES; do
    PRIOR=$(python3 -c "import math; print(int(math.log((1-$P)/$P)*256))")

    # 적응 모드: prior에 따라 반복 횟수 결정
    if [ "$MODE" = "adaptive" ]; then
        if [ "$PRIOR" -ge 1355 ]; then DPU_ITER=3; else DPU_ITER=5; fi
    elif [ "$MODE" = "opt" ]; then
        DPU_ITER=3
    else
        DPU_ITER=20
    fi

    # bpms.go: 매번 템플릿에서 새로 생성 (이전 상태 무관)
    if [ "$MODE" = "noopt" ]; then
        DECODE_LINE='post, _ := bpDecodeNoopt(this.syndromes[t], prior)'
        NITER_LINE=''
    elif [ "$MODE" = "adaptive" ]; then
        DECODE_LINE='post, _ := bpDecodeAdaptive(this.syndromes[t], prior)'
        NITER_LINE=''
    else
        DECODE_LINE='post, _ := bpDecode(this.syndromes[t], prior, nIter)'
        NITER_LINE='	nIter := 3'
    fi

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
		rng := rand.New(rand.NewSource(int64($SEED_BASE + t*137)))
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
$NITER_LINE
	for t := 0; t < this.num_tasklets; t++ {
		$DECODE_LINE
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

    # task.c prior 수정
    python3 << PYEOF
import re
with open('$TASK_C') as f: s = f.read()
s = re.sub(r'(fp_t prior = )[0-9]+', r'\g<1>$PRIOR', s)
s = re.sub(r'decode\(tid, [0-9]+, [0-9]+\)', 'decode(tid, $PRIOR, $DPU_ITER)', s)
s = re.sub(r'decode\(tid, [0-9]+\)', 'decode(tid, $PRIOR)', s)
with open('$TASK_C', 'w') as f: f.write(s)
PYEOF

    # 빌드
    cd "$ROOT/src" && go build -o "$ROOT/build/uPIMulator" . 2>/dev/null
    cd "$ROOT/benchmark" && rm -rf build && mkdir build
    cmake -D NR_DPUS=1 -D NR_TASKLETS=$NR_TASKLETS -S . -B build -G Ninja >/dev/null 2>&1
    ninja -C build BPMS_device >/dev/null 2>&1

    # uPIMulator 실행
    rm -rf "$ROOT/bin" && mkdir -p "$ROOT/bin"
    RESULT=$(timeout 600 "$ROOT/build/uPIMulator" \
        --root_dirpath "$ROOT" --bin_dirpath "$ROOT/bin" \
        --benchmark BPMS \
        --num_channels 1 --num_ranks_per_channel 1 --num_dpus_per_rank 1 \
        --num_tasklets $NR_TASKLETS --data_prep_params 8 \
        --num_simulation_threads 4 2>&1)

    CYCLES=$(grep "logic_cycle" "$ROOT/bin/log.txt" 2>/dev/null | sed 's/.*: //')
    INSTR=$(grep "num_instructions" "$ROOT/bin/log.txt" 2>/dev/null | sed 's/.*: //')

    if echo "$RESULT" | grep -q "MISMATCH\|panic"; then
        echo "  p=$P  prior=$PRIOR  iter=$DPU_ITER  FAIL (DPU != CPU)"
        echo "$P $PRIOR $DPU_ITER 0 FAIL" >> "$CYCLES_FILE"
    else
        echo "  p=$P  prior=$PRIOR  iter=$DPU_ITER  cycles=$CYCLES  instr=$INSTR  PASS"
        echo "$P $PRIOR $DPU_ITER $CYCLES PASS" >> "$CYCLES_FILE"
    fi
done

echo ""

# ---- Phase 2: CPU 실측 + 결과표 ----
echo "[Phase 2] CPU benchmark + results"
echo ""

cat > /tmp/bp_bench_final.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
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

static void build_pcm(void){
    int l=12,m=6,lm=72;
    uint8_t Hz[BB_NC][BB_NV];memset(Hz,0,sizeof(Hz));
    for(int i=0;i<l;i++)for(int j=0;j<m;j++){int col=i*m+j;
        Hz[i*m+((j+3)%m)][col]^=1;Hz[((i+1)%l)*m+j][col]^=1;Hz[((i+2)%l)*m+j][col]^=1;
        Hz[((i+3)%l)*m+j][lm+col]^=1;Hz[i*m+((j+1)%m)][lm+col]^=1;Hz[i*m+((j+2)%m)][lm+col]^=1;}
    for(int c=0;c<BB_NC;c++){int pos=0;for(int v=0;v<BB_NV;v++)if(Hz[c][v])chk_vars[c][pos++]=v;}
    uint8_t sl[BB_NV];memset(sl,0,BB_NV);
    for(int c=0;c<BB_NC;c++)for(int p=0;p<BB_CHK_DEG;p++){int v=chk_vars[c][p];var_map[v][sl[v]++]=c*BB_CHK_DEG+p;}
}

static inline fp_t ascale_opt(fp_t x){return(fp_t)((((int32_t)x<<2)+(int32_t)x)>>3);}
static inline fp_t ascale_mul(fp_t x){return(fp_t)(((int32_t)x*160)>>8);}
static inline fp_t fab(fp_t x){return x<0?-x:x;}
static inline fp_t fp_sat(int32_t x){return(fp_t)(x>FP_MAX?FP_MAX:(x<-FP_MAX?-FP_MAX:x));}

static uint8_t Hz_mat[BB_NC][BB_NV];
static void build_hz(void){
    memset(Hz_mat,0,sizeof(Hz_mat));
    for(int c=0;c<BB_NC;c++)for(int j=0;j<BB_CHK_DEG;j++)Hz_mat[c][chk_vars[c][j]]=1;
}
static int is_stabilizer(const uint8_t *res){
    uint8_t mat[BB_NC][BB_NV];memcpy(mat,Hz_mat,sizeof(mat));
    uint8_t tgt[BB_NV];memcpy(tgt,res,BB_NV);int pr=0;
    for(int col=0;col<BB_NV&&pr<BB_NC;col++){
        int f=-1;for(int r=pr;r<BB_NC;r++)if(mat[r][col]){f=r;break;}
        if(f<0)continue;
        if(f!=pr)for(int c=0;c<BB_NV;c++){uint8_t t=mat[pr][c];mat[pr][c]=mat[f][c];mat[f][c]=t;}
        if(tgt[col])for(int c=0;c<BB_NV;c++)tgt[c]^=mat[pr][c];
        for(int r=0;r<BB_NC;r++)if(r!=pr&&mat[r][col])for(int c=0;c<BB_NV;c++)mat[r][c]^=mat[pr][c];
        pr++;}
    for(int i=0;i<BB_NV;i++)if(tgt[i])return 0;return 1;
}
static int check_conv(const uint8_t *syn,const fp_t *post){
    for(int chk=0;chk<BB_NC;chk++){uint8_t s=0;for(int j=0;j<BB_CHK_DEG;j++)if(post[chk_vars[chk][j]]<=0)s^=1;if(s!=syn[chk])return 0;}return 1;
}

static void bp_decode(const uint8_t *syn,fp_t prior,int max_iter,int early_term,int use_mul,fp_t *v2c,fp_t *c2v,fp_t *post){
    for(int j=0;j<BB_NNZ;j++){v2c[j]=prior;c2v[j]=0;}
    for(int iter=0;iter<max_iter;iter++){
        for(int chk=0;chk<BB_NC;chk++){int b=chk*BB_CHK_DEG;fp_t m[BB_CHK_DEG];
            for(int j=0;j<BB_CHK_DEG;j++)m[j]=v2c[b+j];
            int sign=syn[chk]?1:0;for(int j=0;j<BB_CHK_DEG;j++)if(m[j]<0)sign^=1;
            fp_t a[BB_CHK_DEG];for(int j=0;j<BB_CHK_DEG;j++)a[j]=fab(m[j]);
            fp_t mn1=FP_MAX,mn2=FP_MAX;int mp=0;
            for(int j=0;j<BB_CHK_DEG;j++){if(a[j]<=mn1){mn2=mn1;mn1=a[j];mp=j;}else if(a[j]<mn2)mn2=a[j];}
            for(int j=0;j<BB_CHK_DEG;j++){int s=sign^(m[j]<0?1:0);fp_t mn=(j==mp)?mn2:mn1;
                fp_t cv=use_mul?ascale_mul(mn):ascale_opt(mn);if(s)cv=-cv;c2v[b+j]=cv;}}
        for(int var=0;var<BB_NV;var++){uint16_t i0=var_map[var][0],i1=var_map[var][1],i2=var_map[var][2];
            fp_t c0=c2v[i0],c1=c2v[i1],c2=c2v[i2];
            post[var]=fp_sat((int32_t)prior+c0+c1+c2);
            v2c[i0]=fp_sat((int32_t)prior+c1+c2);v2c[i1]=fp_sat((int32_t)prior+c0+c2);v2c[i2]=fp_sat((int32_t)prior+c0+c1);}
        if(early_term){uint8_t conv=1;
            for(int chk=0;chk<BB_NC;chk++){uint8_t s=0;for(int j=0;j<BB_CHK_DEG;j++)if(post[chk_vars[chk][j]]<=0)s^=1;if(s!=syn[chk]){conv=0;break;}}
            if(conv)break;}
    }
}

static double now_us(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e6+ts.tv_nsec/1e3;}
static double now_sec(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static void fmt(double v,char *b,int sz){if(v>=1e6)snprintf(b,sz,"%.2f M/s",v/1e6);else if(v>=1e3)snprintf(b,sz,"%.1f K/s",v/1e3);else snprintf(b,sz,"%.0f /s",v);}

int main(int argc,char **argv){
    if(argc<4){fprintf(stderr,"Usage: %s <cycles_file> <seed_base> <mode>\n",argv[0]);return 1;}
    build_pcm();build_hz();
    unsigned int seed_base=(unsigned int)atol(argv[2]);
    char *mode=argv[3];
    int is_noopt=strcmp(mode,"noopt")==0;
    int is_opt=strcmp(mode,"opt")==0;
    int is_adaptive=strcmp(mode,"adaptive")==0;

    FILE *cf=fopen(argv[1],"r");if(!cf){perror("cycles");return 1;}
    double p_arr[7];int prior_arr[7];int iter_arr[7];double cyc_arr[7];char status[7][8];int n_entries=0;
    while(n_entries<7&&fscanf(cf,"%lf %d %d %lf %s",&p_arr[n_entries],&prior_arr[n_entries],&iter_arr[n_entries],&cyc_arr[n_entries],status[n_entries])==5)n_entries++;
    fclose(cf);

    FILE *fi=fopen("/proc/cpuinfo","r");char cpu_model[256]="Unknown";int n_phys=0,n_logical=0;
    if(fi){char line[512];while(fgets(line,sizeof(line),fi)){if(strncmp(line,"model name",10)==0&&cpu_model[0]=='U'){char *p=strchr(line,':');if(p){p+=2;p[strlen(p)-1]=0;strncpy(cpu_model,p,255);}}if(strncmp(line,"processor",9)==0)n_logical++;if(strncmp(line,"cpu cores",9)==0){char *p=strchr(line,':');if(p)n_phys=atoi(p+2);}}fclose(fi);}
    int n_cores=n_phys>0?n_phys:n_logical;int K=11;double DPU_MHZ=350.0;int n_trials=50000,n_warmup=2000;

    const char *mode_label=is_opt?"optimized":(is_adaptive?"adaptive":"non-optimized");
    const char *iter_label=is_opt?"3 (fixed, no early termination)":(is_adaptive?"3/5 (prior-based, fixed per p)":"max 20 (convergence-based, early termination)");
    const char *alpha_label=is_noopt?"multiplication: (x*160)>>8":"shift+add: ((x<<2)+x)>>3";

    printf("\n================================================================================\n");
    printf("  BP Throughput Benchmark: CPU vs DPU (%s)\n",mode_label);
    printf("================================================================================\n\n");
    printf("  Code         : [[144,12,12]] Bivariate Bicycle QLDPC\n");
    printf("  Decoder      : Normalized Min-Sum BP (int16 fixed-point)\n");
    printf("  Alpha scaling: %s\n",alpha_label);
    printf("  BP iterations: %s\n",iter_label);
    printf("  CPU trials   : %d per error rate\n",n_trials);
    printf("  Verified     : DPU output = CPU output (byte-exact, %d tasklets)\n",K);
    printf("\n");
    printf("  CPU          : %s (%d cores)\n",cpu_model,n_cores);
    printf("  DPU          : UPMEM PIM @ %.0f MHz, K=%d tasklets/DPU\n",DPU_MHZ,K);
    printf("  DPU cycles   : uPIMulator cycle-accurate simulation (this run)\n");
    printf("\n================================================================================\n");
    printf("  Throughput Comparison (decodes/sec)\n");
    printf("================================================================================\n\n");
    printf("  %6s | %5s | %4s | %10s | %10s | %10s | %12s | %12s | %12s | %12s | %7s | %8s\n",
           "p","prior","iter","DPU cycles","CPU us/dec","E2E us/dec","CPU 1-core","CPU 16-core","1 DPU","2560 DPU","ratio","LER");
    printf("  -------+-------+------+------------+------------+------------+--------------+--------------+--------------+--------------+--------+----------\n");

    for(int pi=0;pi<n_entries;pi++){
        if(strcmp(status[pi],"PASS")!=0){
            printf("  %6.3f | %5d | %4d | %10s | %10s | %10s | %12s | %12s | %12s | %12s | %7s | %8s\n",
                   p_arr[pi],prior_arr[pi],iter_arr[pi],"FAIL","-","-","-","-","-","-","-","-");continue;}
        double p=p_arr[pi];fp_t prior=(fp_t)prior_arr[pi];
        int max_iter=iter_arr[pi];
        int early_term=is_noopt?1:0;
        int use_mul=is_noopt?1:0;
        double cpu_us,cpu1;
        {fp_t lv2c[BB_NNZ],lc2v[BB_NNZ],lpost[BB_NV];uint8_t lsyn[BB_NC],lerr[BB_NV];
        for(int w=0;w<n_warmup;w++){unsigned int seed=seed_base+77777+w;memset(lerr,0,BB_NV);
            for(int i=0;i<BB_NV;i++)if((double)rand_r(&seed)/RAND_MAX<p)lerr[i]=1;
            for(int chk=0;chk<BB_NC;chk++){uint8_t s=0;for(int j=0;j<BB_CHK_DEG;j++)s^=lerr[chk_vars[chk][j]];lsyn[chk]=s;}
            bp_decode(lsyn,prior,max_iter,early_term,use_mul,lv2c,lc2v,lpost);}
        double t1c=now_us();
        for(int t=0;t<5000;t++){unsigned int seed=seed_base+t*7;memset(lerr,0,BB_NV);
            for(int i=0;i<BB_NV;i++)if((double)rand_r(&seed)/RAND_MAX<p)lerr[i]=1;
            for(int chk=0;chk<BB_NC;chk++){uint8_t s=0;for(int j=0;j<BB_CHK_DEG;j++)s^=lerr[chk_vars[chk][j]];lsyn[chk]=s;}
            bp_decode(lsyn,prior,max_iter,early_term,use_mul,lv2c,lc2v,lpost);}
        cpu_us=(now_us()-t1c)/5000.0;cpu1=1e6/cpu_us;}
        int n_logical_err=0;
        double ts_mp=now_sec();
        #pragma omp parallel num_threads(n_cores) reduction(+:n_logical_err)
        {fp_t v2c[BB_NNZ],c2v[BB_NNZ],post[BB_NV];uint8_t syn[BB_NC],error[BB_NV];
        #pragma omp for schedule(static)
        for(int t=0;t<n_trials;t++){unsigned int seed=seed_base+t*137;memset(error,0,BB_NV);
            for(int i=0;i<BB_NV;i++)if((double)rand_r(&seed)/RAND_MAX<p)error[i]=1;
            for(int chk=0;chk<BB_NC;chk++){uint8_t s=0;for(int j=0;j<BB_CHK_DEG;j++)s^=error[chk_vars[chk][j]];syn[chk]=s;}
            bp_decode(syn,prior,max_iter,early_term,use_mul,v2c,c2v,post);
            uint8_t corr[BB_NV],res[BB_NV];int rw=0;
            for(int i=0;i<BB_NV;i++){corr[i]=(post[i]<=0)?1:0;res[i]=error[i]^corr[i];rw+=res[i];}
            if(rw>0){if(!check_conv(syn,post))n_logical_err++;else if(!is_stabilizer(res))n_logical_err++;}}}
        double cpu_all=(double)n_trials/(now_sec()-ts_mp);
        double ler=(double)n_logical_err/n_trials*100.0;
        double cyc=cyc_arr[pi];double dpu_us=cyc/DPU_MHZ;double dpu1=K*1e6/dpu_us;double dpu2560=dpu1*2560;double ratio=dpu2560/cpu_all;
        double e2e_per_us=dpu_us/K;
        char s1[16],s16[16],sd1[16],sd2560[16];
        fmt(cpu1,s1,16);fmt(cpu_all,s16,16);fmt(dpu1,sd1,16);fmt(dpu2560,sd2560,16);
        printf("  %6.3f | %5d | %4d | %10.0f | %10.3f | %10.3f | %12s | %12s | %12s | %12s | %6.1fx | %7.3f%%\n",
               p,prior_arr[pi],iter_arr[pi],cyc,cpu_us,e2e_per_us,s1,s16,sd1,sd2560,ratio,ler);
    }
    printf("  -------+-------+------+------------+------------+------------+--------------+--------------+--------------+--------------+--------+----------\n");

    double avg_cyc=0;int cnt=0;
    for(int i=0;i<n_entries;i++)if(strcmp(status[i],"PASS")==0){avg_cyc+=cyc_arr[i];cnt++;}
    if(cnt>0)avg_cyc/=cnt;
    printf("\n================================================================================\n");
    printf("  DPU Latency Breakdown (%d tasklets)\n",K);
    printf("================================================================================\n\n");
    double dpu_bp_us=avg_cyc/DPU_MHZ;
    printf("  BP compute : %8.1f us  (%.0f cycles @ %.0f MHz)\n",dpu_bp_us,avg_cyc,DPU_MHZ);
    printf("  Per-decode : %8.1f us\n",dpu_bp_us/K);
    printf("\n================================================================================\n\n");
    return 0;
}
CEOF

gcc -O2 -fopenmp -o /tmp/bp_bench_final /tmp/bp_bench_final.c -lm
/tmp/bp_bench_final "$CYCLES_FILE" "$SEED_BASE" "$MODE"

rm -f "$CYCLES_FILE"
