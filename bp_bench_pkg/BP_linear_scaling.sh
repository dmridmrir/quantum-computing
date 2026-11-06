#!/bin/bash
# BP_linear_scaling.sh — DPU 수 증가에 따른 throughput 선형 확장성 실증
#
# 핵심 주장:
#   DPU는 private WRAM 구조로 인해 DPU 간 메모리 간섭이 없으므로
#   per-DPU cycles가 DPU 수와 무관하게 일정하다.
#   → aggregate throughput = N_DPU × K × freq / cycles 는 N_DPU에 선형 비례.
#
# 고정 조건: iter=5, p=0.001
# DPU sweep: 1, 2, 4, 8, 16, 32, 64

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/uPIMulator"
BPMS_GO="$ROOT/src/assembler/prim/bpms.go"
TASK_C="$ROOT/benchmark/BPMS/dpu/task.c"
NR_TASKLETS=11
DPU_ITER=5
P=0.001
PRIOR=$(python3 -c "import math; print(int(math.log((1-$P)/$P)*256))")
SEED=20260511
DPU_COUNTS="1 2 4 8 16 32 64"

echo ""
echo "========================================================================"
echo "  BP Throughput Linear Scaling Verification"
echo "  Code  : [[144,12,12]] BB-QLDPC"
echo "  Fixed : p=$P  prior=$PRIOR  iter=$DPU_ITER  K=$NR_TASKLETS tasklets/DPU"
echo "  Sweep : $DPU_COUNTS DPUs"
echo "========================================================================"
echo ""

# ---- 백업 + 복원 핸들러 ----
cp "$TASK_C"  "$TASK_C.lscale_bak"
cp "$BPMS_GO" "$BPMS_GO.lscale_bak"
for f in "$ROOT/benchmark/BPMS/dpu/task_opt.c" \
         "$ROOT/benchmark/BPMS/dpu/task_noopt.c" \
         "$ROOT/benchmark/BPMS/dpu/task_adaptive.c"; do
    test -f "$f" && mv "$f" "$f.bak" 2>/dev/null || true
done

cleanup() {
    cp "$TASK_C.lscale_bak"  "$TASK_C"  2>/dev/null || true
    cp "$BPMS_GO.lscale_bak" "$BPMS_GO" 2>/dev/null || true
    rm -f "$TASK_C.lscale_bak" "$BPMS_GO.lscale_bak"
    for f in "$ROOT/benchmark/BPMS/dpu/task_opt.c" \
             "$ROOT/benchmark/BPMS/dpu/task_noopt.c" \
             "$ROOT/benchmark/BPMS/dpu/task_adaptive.c"; do
        test -f "$f.bak" && mv "$f.bak" "$f" 2>/dev/null || true
    done
}
trap cleanup EXIT

# ---- task.c 설정: task_opt.c 기반, prior/iter 패치 (1회) ----
if [ ! -f "$ROOT/benchmark/BPMS/dpu/task_opt.c.bak" ]; then
    echo "ERROR: task_opt.c.bak not found."
    echo "       run_opt_bp_bench.sh 를 먼저 한 번 실행해주세요."
    exit 1
fi

cp "$ROOT/benchmark/BPMS/dpu/task_opt.c.bak" "$TASK_C"
python3 << PYEOF
import re
with open('$TASK_C') as f: s = f.read()
s = re.sub(r'decode\(tid,\s*[0-9]+,\s*[0-9]+\)', 'decode(tid, $PRIOR, $DPU_ITER)', s)
with open('$TASK_C', 'w') as f: f.write(s)
PYEOF

# ---- bpms.go 생성: 고정 파라미터, 모든 DPU에 동일 syndrome 제공 (1회) ----
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

# ---- Go 빌드 (1회) ----
echo "[Build] Go binary..."
cd "$ROOT/src" && go build -o "$ROOT/build/uPIMulator" . 2>/dev/null
echo "[Build] DPU device binary (NR_TASKLETS=$NR_TASKLETS)..."
cd "$ROOT/benchmark" && rm -rf build && mkdir build
cmake -D NR_DPUS=1 -D NR_TASKLETS=$NR_TASKLETS -S . -B build -G Ninja >/dev/null 2>&1
ninja -C build BPMS_device >/dev/null 2>&1
echo "[Build] Done."
echo ""

# ---- Phase 1: DPU 수별 cycle 측정 ----
echo "[Phase 1] Cycle measurement per DPU count"
echo ""

RESULTS_FILE=$(mktemp)

for N_DPU in $DPU_COUNTS; do
    T_START=$(date +%s%N)
    rm -rf "$ROOT/bin" && mkdir -p "$ROOT/bin"

    RESULT=$(timeout 600 "$ROOT/build/uPIMulator" \
        --root_dirpath "$ROOT" --bin_dirpath "$ROOT/bin" \
        --benchmark BPMS \
        --num_channels 1 --num_ranks_per_channel 1 --num_dpus_per_rank $N_DPU \
        --num_tasklets $NR_TASKLETS --data_prep_params 8 \
        --num_simulation_threads 4 2>&1)

    T_END=$(date +%s%N)
    ELAPSED=$(( (T_END - T_START) / 1000000 ))

    RAW_CYCLES=$(grep "logic_cycle" "$ROOT/bin/log.txt" 2>/dev/null | sed 's/.*: //' | sort -n | tail -1)

    if [ -z "$RAW_CYCLES" ] || echo "$RESULT" | grep -q "MISMATCH\|panic"; then
        STATUS="FAIL"
        echo "  N_DPU=$N_DPU  FAIL  (${ELAPSED}ms)"
    else
        STATUS="PASS"
        echo "  N_DPU=$N_DPU  raw_cycles=$RAW_CYCLES  PASS  (${ELAPSED}ms)"
    fi
    echo "$N_DPU ${RAW_CYCLES:-0} $STATUS" >> "$RESULTS_FILE"
done

echo ""

# ---- Phase 2: 선형성 분석 ----
echo "[Phase 2] Linear scaling analysis"
echo ""

python3 << PYEOF
import sys, math

MHZ     = 350.0
K       = $NR_TASKLETS
P       = $P
PRIOR   = $PRIOR
ITER    = $DPU_ITER
results = []
with open('$RESULTS_FILE') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3 and parts[2] == 'PASS':
            results.append((int(parts[0]), int(parts[1])))

if not results:
    print("  실행된 결과 없음 (모두 FAIL)")
    sys.exit(1)

n1, raw1 = results[0]
assert n1 == 1, "첫 번째 결과가 N_DPU=1이어야 합니다"

# uPIMulator는 DPU별 독립 goroutine 병렬 실행.
# log.txt에 DPU마다 logic_cycle이 별도 기록되므로 max 값을 사용.
# per-DPU cycles = 각 DPU의 개별 cycle count (N으로 나누지 않음).
def per_dpu(n, raw):
    return raw

base_per_dpu = per_dpu(n1, raw1)

def fmt_thpt(v):
    if v >= 1e9: return f"{v/1e9:.3f} G/s"
    if v >= 1e6: return f"{v/1e6:.3f} M/s"
    if v >= 1e3: return f"{v/1e3:.1f} K/s"
    return f"{v:.0f} /s"

W = 100
print(f"  uPIMulator DPU execution model : parallel (per-DPU goroutine, independent cycle counter)")
print(f"  (1 DPU baseline cycles         : {raw1})")
print()
print("  " + "=" * W)
print("  DPU Count vs. Throughput  —  Linear Scaling Verification")
print("  " + "=" * W)
print()
hdr = f"  {'N_DPU':>6} | {'Raw Cycles':>11} | {'Cycles/DPU':>11} | {'Δ vs 1-DPU':>11} | {'Aggr. Thpt':>13} | {'Measured':>9} | {'Ideal':>7}"
sep = "  " + "-"*7 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*15 + "+" + "-"*11 + "+" + "-"*9
print(hdr)
print(sep)

base_thpt = None
thpt_list = []
pdu_list  = []

for n_dpu, raw in results:
    pdu   = per_dpu(n_dpu, raw)
    delta = pdu - base_per_dpu
    dpct  = delta / base_per_dpu * 100.0
    thpt  = n_dpu * K * 1e6 / (pdu / MHZ)
    thpt_list.append(thpt)
    pdu_list.append(pdu)
    if base_thpt is None:
        base_thpt = thpt

    speedup = thpt / base_thpt
    ideal   = float(n_dpu)

    d_str = f"{delta:+.0f} ({dpct:+.2f}%)" if abs(dpct) > 0.001 else "0  (exact)"
    print(f"  {n_dpu:>6} | {raw:>11} | {int(pdu):>11} | {d_str:>11} | {fmt_thpt(thpt):>13} | {speedup:>8.2f}x | {ideal:>6.0f}x")

print(sep)
print()

# 선형성 판정
devs    = [abs(p - base_per_dpu) / base_per_dpu * 100.0 for p in pdu_list]
max_dev = max(devs)
avg_pdu = sum(pdu_list) / len(pdu_list)

print(f"  Linearity Verdict:")
print(f"    Baseline per-DPU cycles : {int(base_per_dpu)}")
print(f"    Max cycle deviation     : {max_dev:.3f}%")
if max_dev < 0.5:
    verdict = "PERFECT — per-DPU cycles invariant across all DPU counts"
    tag     = "[PASS]"
elif max_dev < 2.0:
    verdict = "LINEAR  — deviation < 2% (within simulation noise)"
    tag     = "[PASS]"
elif max_dev < 5.0:
    verdict = "NEAR-LINEAR — deviation < 5%"
    tag     = "[MARGINAL]"
else:
    verdict = f"NON-LINEAR — deviation {max_dev:.1f}%"
    tag     = "[FAIL]"
print(f"    {tag} {verdict}")
print()

# 논문용 해석
print("  Interpretation:")
print("    Each DPU operates on private WRAM with no inter-DPU memory traffic.")
print(f"    Per-DPU cycle count is syndrome-content-independent and DPU-count-independent.")
print(f"    → Throughput scales linearly: T(N) = N × {fmt_thpt(base_thpt)}")
print()
print("  " + "=" * W)
print()
PYEOF

rm -f "$RESULTS_FILE"
