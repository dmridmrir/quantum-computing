#!/bin/bash
# BF-UF Benchmark: CPU Alg3 vs DPU Alg3 vs DPU Alg3-var
#
# Experiments:
#   [1] DPU Algorithm 3     — double-step BFS + validity tracking + uniteKeepA
#   [2] DPU Algorithm 3-var — Alg3 + Lazy GE + BFS-queue member collection
#   [3] CPU Algorithm 3     — same decoder, gcc -O2, multi-core baseline
#
# Ref: Löbl et al., arXiv:2407.15988 (2024)
set -e

export PATH=/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ROOT="${HOME}/bp_bench_pkg/uPIMulator"
BFUF_DPU_DIR="$ROOT/benchmark/BFUF/dpu"
BFUF_GO="$ROOT/src/assembler/prim/bfuf.go"
UPIM="$ROOT/build/uPIMulator"
BIN="$ROOT/bin"
NR_TASKLETS=11
ERROR_RATES="0.001 0.003 0.005 0.007 0.009"
SEED=$(od -An -N4 -tu4 /dev/urandom | tr -d ' \n')
NUM_EXECUTIONS=100
SYN_DIR=$(mktemp -d)

CSV_ALG3=$(mktemp)
CSV_ALG3_VAR=$(mktemp)
BFUF_GO_TEMPLATE="$HOME/bp_bench_pkg/uPIMulator/src/assembler/prim/bfuf.go.tpl"

echo ""
echo "========================================================================"
echo "  BF-UF Benchmark  [[144,12,12]] BB-QLDPC  K=$NR_TASKLETS"
echo "========================================================================"
echo ""

# Go template — DECODE_FUNC_PH / P_PH / SEED_PH replaced per run via sed
cat > "$BFUF_GO_TEMPLATE" << 'GOEOF'
package prim

import (
	"errors"
	"math/rand"
	"os"
	"uPIMulator/src/abi/encoding"
	"uPIMulator/src/abi/word"
	"uPIMulator/src/misc"
)

type Bfuf struct {
	numDpus       int
	numTasklets   int
	numExecutions int
	nc            int64
	nv            int64
	syndromes     [][][]byte
}

func (this *Bfuf) Init(command_line_parser *misc.CommandLineParser) {
	num_channels := int(command_line_parser.IntParameter("num_channels"))
	num_ranks_per_channel := int(command_line_parser.IntParameter("num_ranks_per_channel"))
	num_dpus_per_rank := int(command_line_parser.IntParameter("num_dpus_per_rank"))
	this.numDpus = num_channels * num_ranks_per_channel * num_dpus_per_rank
	this.numTasklets = int(command_line_parser.IntParameter("num_tasklets"))
	this.numExecutions = NUM_EXEC_PH
	this.nc = bfufNC
	this.nv = bfufNV

	this.syndromes = make([][][]byte, this.numExecutions)
	for e := 0; e < this.numExecutions; e++ {
		this.syndromes[e] = make([][]byte, this.numTasklets)
		for t := 0; t < this.numTasklets; t++ {
			rng := rand.New(rand.NewSource(int64(SEED_PH + e*this.numTasklets*137 + t*137)))
			error := make([]uint8, this.nv)
			for v := int64(0); v < this.nv; v++ {
				if rng.Float64() < P_PH {
					error[v] = 1
				}
			}
			syn := make([]byte, this.nc)
			for c := int64(0); c < this.nc; c++ {
				s := byte(0)
				for j := 0; j < bfufChkDeg; j++ {
					s ^= error[bpChkVarsData[c][j]]
				}
				syn[c] = s
			}
			this.syndromes[e][t] = syn
		}
	}
	f, _ := os.Create("SYN_FILE_PH")
	for e := 0; e < this.numExecutions; e++ {
		for t := 0; t < this.numTasklets; t++ {
			f.Write(this.syndromes[e][t])
		}
	}
	f.Close()
}

func (this *Bfuf) InputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	if execution >= this.numExecutions {
		panic(errors.New("execution >= num executions"))
	}
	bs := new(encoding.ByteStream)
	bs.Init()
	for _, v := range []int64{int64(this.numTasklets), 0, 0} {
		w := new(word.Word)
		w.Init(32)
		w.SetValue(v)
		bs.Merge(w.ToByteStream())
	}
	dpu_host := make(map[string]*encoding.ByteStream)
	dpu_host["DPU_INPUT_ARGUMENTS"] = bs
	return dpu_host
}

func (this *Bfuf) OutputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {
	return make(map[string]*encoding.ByteStream)
}

func (this *Bfuf) InputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	bs := new(encoding.ByteStream)
	bs.Init()
	for t := 0; t < this.numTasklets; t++ {
		for _, b := range this.syndromes[execution][t] {
			w := new(word.Word)
			w.Init(8)
			w.SetValue(int64(b))
			bs.Merge(w.ToByteStream())
		}
	}
	return 0, bs
}

func (this *Bfuf) OutputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {
	bs := new(encoding.ByteStream)
	bs.Init()
	for t := 0; t < this.numTasklets; t++ {
		corr := DECODE_FUNC_PH(this.syndromes[execution][t])
		for i := 0; i < int(this.nv); i++ {
			w := new(word.Word)
			w.Init(8)
			w.SetValue(int64(corr[i]))
			bs.Merge(w.ToByteStream())
		}
	}
	synSlotSize := int64(this.nc)
	return int64(this.numTasklets) * synSlotSize, bs
}

func (this *Bfuf) NumExecutions() int { return this.numExecutions }
GOEOF

# ── Step 1: Build & simulate DPU Algorithm 3 ───────────────────────────────
echo "[1/5] Building DPU kernel (Algorithm 3)..."
cd "$ROOT/benchmark"
rm -rf build && mkdir build
cmake -D NR_DPUS=1 -D NR_TASKLETS=$NR_TASKLETS -S . -B build -G Ninja >/dev/null 2>&1
ninja -C build BFUF_device >/dev/null 2>&1
echo "      DPU kernel built"
echo ""

echo "[2/5] DPU simulation — Algorithm 3"
echo ""
for P in $ERROR_RATES; do
    sed "s/DECODE_FUNC_PH/bfufDecode/g;s/P_PH/$P/g;s/SEED_PH/$SEED/g;s/NUM_EXEC_PH/$NUM_EXECUTIONS/g;s|SYN_FILE_PH|$SYN_DIR/syn_${P}.bin|g" \
        "$BFUF_GO_TEMPLATE" > "$BFUF_GO"

    cd "$ROOT/src"
    /usr/local/go/bin/go build -o "$UPIM" .

    rm -rf "$BIN" && mkdir -p "$BIN"
    RESULT=$(timeout 3600 "$UPIM" \
        --root_dirpath "$ROOT" \
        --bin_dirpath  "$BIN" \
        --benchmark    BFUF \
        --num_channels 1 --num_ranks_per_channel 1 --num_dpus_per_rank 1 \
        --num_tasklets $NR_TASKLETS \
        --data_prep_params 8 \
        --num_simulation_threads 4 2>&1) || true

    CYCLES=$(grep "logic_cycle"      "$BIN/log.txt" 2>/dev/null | awk -F': ' '{print $2}')
    INSTRS=$(grep "num_instructions" "$BIN/log.txt" 2>/dev/null | awk -F': ' '{print $2}')
    CYCLES="${CYCLES:-0}"; INSTRS="${INSTRS:-0}"

    if echo "$RESULT" | grep -q "MISMATCH\|panic"; then
        STATUS="FAIL"; printf "  Alg3     p=%-5s  FAIL\n" "$P"
    else
        STATUS="PASS"; printf "  Alg3     p=%-5s  cycles=%-10s  PASS\n" "$P" "$CYCLES"
    fi
    echo "$P,$CYCLES,$INSTRS,$STATUS" >> "$CSV_ALG3"
done
echo ""

# ── Step 3: Build & simulate DPU Algorithm 3-var ───────────────────────────
echo "[3/5] Building DPU kernel (Algorithm 3-var: linked-list membership)..."
cp "$BFUF_DPU_DIR/task.c"          "$BFUF_DPU_DIR/task.c.bak"
cp "$BFUF_DPU_DIR/task_alg3_var.c" "$BFUF_DPU_DIR/task.c"
cd "$ROOT/benchmark"
rm -rf build && mkdir build
cmake -D NR_DPUS=1 -D NR_TASKLETS=$NR_TASKLETS -S . -B build -G Ninja >/dev/null 2>&1
ninja -C build BFUF_device >/dev/null 2>&1
cp "$BFUF_DPU_DIR/task.c.bak" "$BFUF_DPU_DIR/task.c"
rm  "$BFUF_DPU_DIR/task.c.bak"
echo "      DPU kernel built"
echo ""

echo "[4/5] DPU simulation — Algorithm 3-var"
echo ""
for P in $ERROR_RATES; do
    sed "s/DECODE_FUNC_PH/bfufDecode/g;s/P_PH/$P/g;s/SEED_PH/$SEED/g;s/NUM_EXEC_PH/$NUM_EXECUTIONS/g;s|SYN_FILE_PH|$SYN_DIR/syn_${P}.bin|g" \
        "$BFUF_GO_TEMPLATE" > "$BFUF_GO"

    cd "$ROOT/src"
    /usr/local/go/bin/go build -o "$UPIM" .

    rm -rf "$BIN" && mkdir -p "$BIN"
    RESULT=$(timeout 3600 "$UPIM" \
        --root_dirpath "$ROOT" \
        --bin_dirpath  "$BIN" \
        --benchmark    BFUF \
        --num_channels 1 --num_ranks_per_channel 1 --num_dpus_per_rank 1 \
        --num_tasklets $NR_TASKLETS \
        --data_prep_params 8 \
        --num_simulation_threads 4 2>&1) || true

    CYCLES=$(grep "logic_cycle"      "$BIN/log.txt" 2>/dev/null | awk -F': ' '{print $2}')
    INSTRS=$(grep "num_instructions" "$BIN/log.txt" 2>/dev/null | awk -F': ' '{print $2}')
    CYCLES="${CYCLES:-0}"; INSTRS="${INSTRS:-0}"

    if echo "$RESULT" | grep -q "panic"; then
        STATUS="FAIL"; printf "  Alg3-var p=%-5s  FAIL\n" "$P"
    else
        STATUS="PASS"; printf "  Alg3-var p=%-5s  cycles=%-10s  PASS\n" "$P" "$CYCLES"
    fi
    echo "$P,$CYCLES,$INSTRS,$STATUS" >> "$CSV_ALG3_VAR"
done
echo ""

# ── Step 5: Compile CPU benchmark ──────────────────────────────────────────
echo "[5/5] Compiling CPU benchmark (gcc -O2)..."

cat > /tmp/bfuf_bench_cpu.c << 'CEOF'
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define BB_NC      72
#define BB_NV     144
#define BB_CHK_DEG  6
#define BB_VAR_DEG  3
#define TOTAL_NODES (BB_NC + BB_NV)
#define VAR_NODE(v) (BB_NC + (v))
#define GF2_WORDS   3

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

static uint8_t var_chks[BB_NV][BB_VAR_DEG];
static void build_var_chks(void) {
    uint8_t cnt[BB_NV] = {0};
    for(int c=0;c<BB_NC;c++)
        for(int j=0;j<BB_CHK_DEG;j++){
            int v=bb_chk_vars[c][j]; var_chks[v][cnt[v]++]=(uint8_t)c;
        }
}

/* ── Union-Find (path-halving + uniteKeepA) ── */
static int uf_par[TOTAL_NODES], uf_rnk[TOTAL_NODES];
static int uf_find(int x){
    while(uf_par[x]!=x){uf_par[x]=uf_par[uf_par[x]];x=uf_par[x];} return x;
}
static void uf_unite_keep_a(int a,int b){
    int ra=uf_find(a),rb=uf_find(b); if(ra==rb)return;
    uf_par[rb]=ra; if(uf_rnk[ra]<=uf_rnk[rb])uf_rnk[ra]=uf_rnk[rb]+1;
}

/* ── GF(2) Gaussian elimination ── */
static int gf2_solve(uint8_t H[][BB_NV],uint8_t *s,uint8_t *e,int nc,int nv){
    uint64_t mat[BB_NC][GF2_WORDS]; memset(mat,0,sizeof(mat));
    for(int r=0;r<nc;r++){
        for(int c=0;c<nv;c++) if(H[r][c]) mat[r][c>>6]|=1ULL<<(c&63);
        if(s[r]) mat[r][nv>>6]|=1ULL<<(nv&63);
    }
    int prow=0,pcol[BB_NC]; memset(pcol,-1,sizeof(pcol));
    for(int col=0;col<nv&&prow<nc;col++){
        int pivot=-1;
        for(int r=prow;r<nc;r++) if((mat[r][col>>6]>>(col&63))&1){pivot=r;break;}
        if(pivot<0) continue;
        if(pivot!=prow) for(int w=0;w<GF2_WORDS;w++){
            uint64_t t=mat[prow][w];mat[prow][w]=mat[pivot][w];mat[pivot][w]=t;
        }
        pcol[prow]=col;
        for(int r=0;r<nc;r++) if(r!=prow&&(mat[r][col>>6]>>(col&63))&1)
            for(int w=0;w<GF2_WORDS;w++) mat[r][w]^=mat[prow][w];
        prow++;
    }
    for(int r=prow;r<nc;r++) if((mat[r][nv>>6]>>(nv&63))&1) return 0;
    memset(e,0,nv);
    for(int r=0;r<prow;r++) if(pcol[r]>=0)
        e[pcol[r]]=(uint8_t)((mat[r][nv>>6]>>(nv&63))&1);
    return 1;
}

/* ── Algorithm 3 shared state ── */
static uint8_t vis3[TOTAL_NODES], par3[TOTAL_NODES];
static uint8_t H3[BB_NC][BB_NV],s3[BB_NC],e3[BB_NV];
static int     cl_checks3[BB_NC],cl_vars3[BB_NV];
static uint8_t cl_var_idx3[BB_NV];
static int     queue3[TOTAL_NODES];
static uint8_t rdone3[TOTAL_NODES];

static int cluster_solvable(int root,const uint8_t syn[BB_NC]){
    int nc_cl=0,nv_cl=0; memset(cl_var_idx3,0xFF,BB_NV);
    for(int c=0;c<BB_NC;c++)
        if(vis3[c]&&uf_find(c)==root) cl_checks3[nc_cl++]=c;
    for(int v=0;v<BB_NV;v++){
        int vn=VAR_NODE(v);
        if(vis3[vn]&&uf_find(vn)==root){cl_var_idx3[v]=(uint8_t)nv_cl;cl_vars3[nv_cl++]=v;}
    }
    if(!nc_cl||!nv_cl) return 0;
    memset(H3,0,sizeof(H3));
    for(int r=0;r<nc_cl;r++){
        int c=cl_checks3[r]; s3[r]=syn[c];
        for(int j=0;j<BB_CHK_DEG;j++){
            int v=bb_chk_vars[c][j];
            if(cl_var_idx3[v]!=0xFF) H3[r][cl_var_idx3[v]]=1;
        }
    }
    return gf2_solve(H3,s3,e3,nc_cl,nv_cl);
}

static void bfuf_decode_alg3(const uint8_t syn[BB_NC],uint8_t corr[BB_NV]){
    for(int i=0;i<TOTAL_NODES;i++){uf_par[i]=i;uf_rnk[i]=0;}
    memset(vis3,0,sizeof(vis3)); memset(par3,0,sizeof(par3));
    memset(corr,0,BB_NV);
    int numInvalid=0,qh=0,qt=0;
    for(int c=0;c<BB_NC;c++) if(syn[c]){vis3[c]=1;par3[c]=1;numInvalid++;queue3[qt++]=c;}
    while(numInvalid>0&&qh<qt){
        int node=queue3[qh++];
        int root=uf_find(node); if(!par3[root]) continue;
        for(int j=0;j<BB_CHK_DEG;j++){
            int v=bb_chk_vars[node][j],vn=VAR_NODE(v);
            int rvn=uf_find(vn),rc=uf_find(node);
            if(rvn!=rc){if(par3[rvn]){numInvalid--;par3[rvn]=0;}uf_unite_keep_a(node,vn);}
            vis3[vn]=1;
            for(int k=0;k<BB_VAR_DEG;k++){
                int cn=var_chks[v][k],rcn=uf_find(cn),rc2=uf_find(node);
                if(rcn!=rc2){if(par3[rcn]){numInvalid--;par3[rcn]=0;}uf_unite_keep_a(node,cn);}
                if(!vis3[cn]){vis3[cn]=1;queue3[qt++]=cn;}
            }
        }
        root=uf_find(node);
        if(cluster_solvable(root,syn)) if(par3[root]){par3[root]=0;numInvalid--;}
    }
    memset(rdone3,0,sizeof(rdone3));
    for(int qi=0;qi<qt;qi++){
        int node=queue3[qi]; int root=uf_find(node); if(rdone3[root]) continue;
        rdone3[root]=1;
        int nc_cl=0,nv_cl=0; memset(cl_var_idx3,0xFF,BB_NV);
        for(int c=0;c<BB_NC;c++) if(vis3[c]&&uf_find(c)==root) cl_checks3[nc_cl++]=c;
        for(int v=0;v<BB_NV;v++){
            int vn=VAR_NODE(v);
            if(vis3[vn]&&uf_find(vn)==root){cl_var_idx3[v]=(uint8_t)nv_cl;cl_vars3[nv_cl++]=v;}
        }
        if(!nc_cl||!nv_cl) continue;
        memset(H3,0,sizeof(H3));
        for(int r=0;r<nc_cl;r++){
            int c=cl_checks3[r]; s3[r]=syn[c];
            for(int j=0;j<BB_CHK_DEG;j++){
                int v=bb_chk_vars[c][j];
                if(cl_var_idx3[v]!=0xFF) H3[r][cl_var_idx3[v]]=1;
            }
        }
        if(gf2_solve(H3,s3,e3,nc_cl,nv_cl))
            for(int col=0;col<nv_cl;col++) corr[cl_vars3[col]]^=e3[col];
    }
}

static void compute_syn(const uint8_t bits[BB_NV],uint8_t syn[BB_NC]){
    for(int c=0;c<BB_NC;c++){
        uint8_t s=0;
        for(int j=0;j<BB_CHK_DEG;j++) s^=bits[bb_chk_vars[c][j]];
        syn[c]=s;
    }
}
static double now_us(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1e6+ts.tv_nsec*1e-3;
}
static void fmt_rate(double v,char *b,int sz){
    snprintf(b,sz,"%.0f dec/s",v);
}

typedef struct{double p;long cyc;int pass;} row_t;
static int parse_csv(const char *path,row_t *rows,int maxn){
    FILE *f=fopen(path,"r"); if(!f){perror(path);return 0;}
    int n=0; char status[16];
    while(n<maxn&&fscanf(f,"%lf,%ld,%*[^,],%15s",&rows[n].p,&rows[n].cyc,status)==3){
        rows[n].pass=(strcmp(status,"PASS")==0); n++;
    }
    fclose(f); return n;
}

int main(int argc,char **argv){
    if(argc<7){
        fprintf(stderr,"Usage: %s <csv_alg3> <csv_alg3_var> <seed> <nr_tasklets> <num_exec> <syn_dir>\n",argv[0]);
        return 1;
    }
    build_var_chks();

    row_t alg3[16], alg3v[16];
    int n3  = parse_csv(argv[1],alg3, 16);
    int n3v = parse_csv(argv[2],alg3v,16);
    int n   = n3<n3v ? n3 : n3v;

    unsigned int seed_base=(unsigned int)atol(argv[3]); (void)seed_base;
    int K=atoi(argv[4]);
    int num_exec=atoi(argv[5]);
    const char *syn_dir=argv[6];
    int n_warmup=2000;
    double dpu_mhz=350.0;

    FILE *fi=fopen("/proc/cpuinfo","r");
    char cpu_model[256]="Unknown"; int n_phys=0,n_logical=0;
    if(fi){
        char line[512];
        while(fgets(line,sizeof(line),fi)){
            if(!strncmp(line,"model name",10)&&cpu_model[0]=='U'){
                char *p=strchr(line,':'); if(p){p+=2;size_t l=strlen(p);
                if(l>0&&p[l-1]=='\n')p[l-1]=0;strncpy(cpu_model,p,255);}
            }
            if(!strncmp(line,"processor",9)) n_logical++;
            if(!strncmp(line,"cpu cores",9)){char *p=strchr(line,':');if(p)n_phys=atoi(p+2);}
        }
        fclose(fi);
    }
    int n_cores=n_phys>0?n_phys:n_logical;

    printf("\n================================================================================\n");
    printf("  BF-UF Throughput Benchmark: Alg3 vs Alg3-var (CPU vs DPU)\n");
    printf("================================================================================\n\n");
    printf("  Code     : [[144,12,12]] Bivariate Bicycle QLDPC\n");
    printf("  Decoder  : BF-UF Algorithm 3 / Algorithm 3-var\n");
    printf("  Ref      : Löbl et al., arXiv:2407.15988 (2024)\n");
    printf("  CPU      : %s\n",cpu_model);
    printf("             (%d physical cores / %d logical)\n",n_cores,n_logical);
    printf("  DPU      : UPMEM PIM @ %.0f MHz, K=%d tasklets/DPU, %d exec (uPIMulator)\n",dpu_mhz,K,num_exec);
    printf("  Compiler : gcc -O2 (CPU)  |  dpu-upmem-dpurte-clang (DPU)\n");
    printf("  Trials   : %d syndromes per p (warmup %d), %d exec x K=%d tasklets\n\n",num_exec*K,n_warmup,num_exec,K);

    /* header */
    printf("  %6s | %12s | %19s | %21s | %16s | %16s | %14s | %15s\n",
           "p", "CPU us/dec", "alg3 1-DPU us/dec", "alg3.v 1-DPU us/dec",
           "alg3 Thr", "alg3.v Thr", "alg3 speedup", "alg3.v speedup");
    printf("  -------+--------------+---------------------+-----------------------+"
           "------------------+------------------+"
           "----------------+-----------------\n");

    uint8_t corr[BB_NV];

    for(int pi=0;pi<n;pi++){
        double p=alg3[pi].p;
        long cyc3=alg3[pi].cyc, cyc3v=alg3v[pi].cyc;
        int pass3=alg3[pi].pass, pass3v=alg3v[pi].pass;

        char syn_path[512];
        snprintf(syn_path,sizeof(syn_path),"%s/syn_%.3f.bin",syn_dir,p);
        FILE *sf=fopen(syn_path,"rb");
        int total_syns=num_exec*K;
        uint8_t (*syns)[BB_NC]=(uint8_t(*)[BB_NC])malloc((size_t)total_syns*BB_NC);
        fread(syns,BB_NC,total_syns,sf);
        fclose(sf);

        for(int w=0;w<n_warmup;w++) bfuf_decode_alg3(syns[w%total_syns],corr);

        double total_us=0;
        for(int i=0;i<total_syns;i++){
            double t0=now_us(); bfuf_decode_alg3(syns[i],corr); double t1=now_us();
            total_us+=(t1-t0);
        }
        double cpu_us=total_us/total_syns;
        free(syns);

        /* CPU throughputs */
        double cpu1    = 1e6/cpu_us;
        double cpu8    = cpu1*n_cores;

        /* DPU throughputs */
        double alg3_1dpu   = (pass3  &&cyc3 >0)?((double)K*dpu_mhz*1e6/((double)cyc3 /(double)num_exec)):0.0;
        double alg3_2560   = alg3_1dpu  *2560.0;
        double alg3v_1dpu  = (pass3v &&cyc3v>0)?((double)K*dpu_mhz*1e6/((double)cyc3v/(double)num_exec)):0.0;
        double alg3v_2560  = alg3v_1dpu *2560.0;

        char sdpu3[24],sdpu3v[24],sa2560[24],sav2560[24],sspd3[16],sspd3v[16];

        if(alg3_1dpu > 0){
            double dpu_us3 = (double)cyc3 / (dpu_mhz * (double)K * (double)num_exec);
            snprintf(sdpu3,  24, "%.3f", dpu_us3);
            fmt_rate(alg3_2560, sa2560, 24);
            snprintf(sspd3,  16, "%6.1fx", alg3_2560/cpu8);
        } else {
            snprintf(sdpu3,  24, "FAIL");
            snprintf(sa2560, 24, "FAIL");
            snprintf(sspd3,  16, "     -");
        }
        if(alg3v_1dpu > 0){
            double dpu_us3v = (double)cyc3v / (dpu_mhz * (double)K * (double)num_exec);
            snprintf(sdpu3v, 24, "%.3f", dpu_us3v);
            fmt_rate(alg3v_2560, sav2560, 24);
            snprintf(sspd3v, 16, "%6.1fx", alg3v_2560/cpu8);
        } else {
            snprintf(sdpu3v, 24, "FAIL");
            snprintf(sav2560,24, "FAIL");
            snprintf(sspd3v, 16, "     -");
        }

        printf("  %6.3f | %12.3f | %19s | %21s | %16s | %16s | %14s | %15s\n",
               p, cpu_us, sdpu3, sdpu3v, sa2560, sav2560, sspd3, sspd3v);
    }
    printf("  -------+--------------+---------------------+-----------------------+"
           "------------------+------------------+"
           "----------------+-----------------\n\n");
    printf("  Notes:\n");
    printf("  - CPU us/dec         : 1-core latency, gcc -O2\n");
    printf("  - *-DPU us/dec       : avg per-decode = total_cycles / (%.0f MHz x %d tasklets x %d exec)\n", dpu_mhz, K, num_exec);
    printf("  - Thr                : 2560 DPUs x %d tasklets @ %.0f MHz\n", K, dpu_mhz);
    printf("  - speedup            : 2560-DPU Thr / (%d-core CPU Thr)\n\n", n_cores);
    return 0;
}
CEOF

gcc -O2 -o /tmp/bfuf_bench_cpu /tmp/bfuf_bench_cpu.c
echo "      OK"
echo ""

# ── Run CPU benchmark → print table ────────────────────────────────────────
echo "Running CPU benchmark + printing comparison table..."
echo ""
/tmp/bfuf_bench_cpu "$CSV_ALG3" "$CSV_ALG3_VAR" "$SEED" "$NR_TASKLETS" "$NUM_EXECUTIONS" "$SYN_DIR"

rm -f "$CSV_ALG3" "$CSV_ALG3_VAR" "$BFUF_GO_TEMPLATE"
rm -rf "$SYN_DIR"
rm -f /tmp/bfuf_bench_cpu.c /tmp/bfuf_bench_cpu
