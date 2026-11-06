#!/bin/bash
# run_bp_increase_layered_phase1.sh — DPU simulation ONLY (layered schedule).
#
# Split out of run_bp_increase_layered.sh: this half has no CPU benchmarking
# at all, just uPIMulator cycle-accurate simulation. The reported cycle count
# (and everything derived from it) is a deterministic function of the
# simulated program+data, independent of the host machine's speed -- run this
# on whatever machine finishes fastest, then copy the two output files to
# wherever you'll run phase2 (which measures real CPU wall-clock time, so
# THAT one needs to run on the machine you're actually comparing against,
# e.g. the EC2 instance).
#
# Went back to simulating 1 DPU + linear projection (dpu1 x PROJECTION_DPUS),
# NOT actually simulating PROJECTION_DPUS real DPU objects. Tried the real
# simulation approach first: even on a 32-core/504GB RAM machine, simulating
# 2560 real DPUs pushed a single uPIMulator process to ~474GB RSS and 100%
# swap, freezing the whole host. This is a memory-per-DPU-object limitation
# of uPIMulator itself, not a CPU/thread issue, and not something more RAM
# headroom easily fixes at that scale. Smaller real-DPU-count runs (16, 64)
# already confirmed all DPUs report the identical cycle count (no cross-DPU
# coupling in uPIMulator's code), which is the actual justification for the
# linear projection -- so there's no need to brute-force the full count.
#
# Outputs (in this script's directory):
#   layered_cycles_raw.txt  - "$P $PRIOR $DPU_ITER $CYCLES PASS/FAIL" per line
#   layered_meta.txt        - TOTAL_DPUS=<n> (projection target, phase2 needs this)
#
# 실행 방법:
#   ./run_bp_increase_layered_phase1.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/uPIMulator"
BPMS_GO="$ROOT/src/assembler/prim/bpms.go"
TASK_C="$ROOT/benchmark/BPMS/dpu/task.c"
NR_TASKLETS=11
ERROR_RATES="${ERROR_RATES:-0.001 0.003 0.005 0.007 0.009}"
MAX_DPU_ITER="${MAX_DPU_ITER:-10}"
SEED_BASE=${SHARED_SEED:-$(date +%s)}

NUM_CHANNELS=${NUM_CHANNELS:-1}
NUM_RANKS_PER_CHANNEL=${NUM_RANKS_PER_CHANNEL:-1}
NUM_DPUS_PER_RANK=${NUM_DPUS_PER_RANK:-1}
NUM_SIMULATION_THREADS=${NUM_SIMULATION_THREADS:-4}
PROJECTION_DPUS=${PROJECTION_DPUS:-2560}
TOTAL_DPUS=$((NUM_CHANNELS * NUM_RANKS_PER_CHANNEL * NUM_DPUS_PER_RANK))

echo ""
echo "================================================================================"
echo "  [Phase 1 only] BP DPU simulation: iter 1~${MAX_DPU_ITER} sweep (LAYERED / Gauss-Seidel)"
echo "  Mapping: 11 tasklets cooperate on 1 syndrome"
echo "  DPUs actually simulated: $TOTAL_DPUS (channels=$NUM_CHANNELS x ranks=$NUM_RANKS_PER_CHANNEL x dpus/rank=$NUM_DPUS_PER_RANK), sim_threads=$NUM_SIMULATION_THREADS"
echo "  Projection target for phase2's '2560 DPU' column: $PROJECTION_DPUS (linear, dpu1 x $PROJECTION_DPUS)"
echo "  Seed base: $SEED_BASE"
echo "  총 조합: $(echo $ERROR_RATES | wc -w | tr -d ' ') 에러율 × ${MAX_DPU_ITER} iter = $(($(echo $ERROR_RATES | wc -w | tr -d ' ') * MAX_DPU_ITER)) 시뮬레이션"
echo "================================================================================"
echo ""

# ---- 백업 + 종료 시 복원 ----
cp "$TASK_C" "$TASK_C.backup"
cp "$BPMS_GO" "$BPMS_GO.backup"
for f in "$ROOT/benchmark/BPMS/dpu/task_opt.c" \
         "$ROOT/benchmark/BPMS/dpu/task_noopt.c" \
         "$ROOT/benchmark/BPMS/dpu/task_adaptive.c" \
         "$ROOT/benchmark/BPMS/dpu/task_layered.c" \
         "$ROOT/benchmark/BPMS/dpu/task_layered_v2.c"; do
    test -f "$f" && mv "$f" "$f.bak" 2>/dev/null || true
done

cleanup() {
    cp "$TASK_C.backup" "$TASK_C" 2>/dev/null || true
    cp "$BPMS_GO.backup" "$BPMS_GO" 2>/dev/null || true
    rm -f "$TASK_C.backup" "$BPMS_GO.backup"
    for f in "$ROOT/benchmark/BPMS/dpu/task_opt.c" \
             "$ROOT/benchmark/BPMS/dpu/task_noopt.c" \
             "$ROOT/benchmark/BPMS/dpu/task_adaptive.c" \
             "$ROOT/benchmark/BPMS/dpu/task_layered.c" \
             "$ROOT/benchmark/BPMS/dpu/task_layered_v2.c"; do
        test -f "$f.bak" && mv "$f.bak" "$f" 2>/dev/null || true
    done
}
trap cleanup EXIT

CYCLES_FILE=$(mktemp)

for DPU_ITER in $(seq 1 $MAX_DPU_ITER); do
    echo "  ---- iter = $DPU_ITER ----"
    for P in $ERROR_RATES; do
        PRIOR=$(python3 -c "import math; print(int(math.log((1-$P)/$P)*256))")

        # 매번 task_layered.c 원본에서 새로 복사하여 패치
        cp "$ROOT/benchmark/BPMS/dpu/task_layered.c.bak" "$TASK_C"

        # task.c 패치: prior + iter 수정 (decode(tid, prior, n_iter))
        python3 << PYEOF
import re
with open('$TASK_C') as f: s = f.read()
s = re.sub(r'decode\(tid, [0-9]+, [0-9]+\)', 'decode(tid, $PRIOR, $DPU_ITER)', s)
with open('$TASK_C', 'w') as f: f.write(s)
PYEOF

        # bpms.go 생성: layered 방식 — bpDecodeLayered(syn, prior, nIter) 사용
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
	if this.num_tasklets != 11 { panic(errors.New("cooperative v1 requires 11 tasklets")) }
	this.syndromes = make([][]byte, 1)
	for t := 0; t < len(this.syndromes); t++ {
		rng := rand.New(rand.NewSource(int64($SEED_BASE + t*137)))
		// p is the physical data-qubit error probability. Generate the
		// Bernoulli error vector first, then derive the syndrome s = H*e.
		// Sampling syndrome bits directly with probability p would not be
		// the same channel because every check has degree bpChkDeg.
		errVec := make([]byte, this.nv)
		for i := range errVec {
			if rng.Float64() < $P {
				errVec[i] = 1
			}
		}
		syn := make([]byte, this.nc)
		for chk := 0; chk < int(this.nc); chk++ {
			var s byte
			for j := 0; j < bpChkDeg; j++ {
				s ^= errVec[bpChkVarsData[chk][j]]
			}
			syn[chk] = s
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
	for t := 0; t < len(this.syndromes); t++ {
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
	for t := 0; t < len(this.syndromes); t++ {
		post, _ := bpDecodeLayered(this.syndromes[t], prior, nIter)
		for i := 0; i < int(this.nv); i++ {
			val := post[i]
			lo := new(word.Word); lo.Init(8); lo.SetValue(int64(uint8(val & 0xFF))); bs.Merge(lo.ToByteStream())
			hi := new(word.Word); hi.Init(8); hi.SetValue(int64(uint8((val >> 8) & 0xFF))); bs.Merge(hi.ToByteStream())
		}
	}
	slotSize := ((int(this.nc) + 7) / 8) * 8
	return int64(len(this.syndromes) * slotSize), bs
}

func (this *Bpms) NumExecutions() int { return this.num_executions }
GOEOF

        # 빌드
        cd "$ROOT/src" && go build -o "$ROOT/build/uPIMulator" . 2>/dev/null
        cd "$ROOT/benchmark" && rm -rf build && mkdir build
        cmake -D NR_DPUS=1 -D NR_TASKLETS=$NR_TASKLETS -S . -B build -G Ninja >/dev/null 2>&1
        ninja -C build BPMS_device >/dev/null 2>&1

        # uPIMulator 실행 -- 1개 DPU만 실제로 시뮬레이션
        rm -rf "$ROOT/bin" && mkdir -p "$ROOT/bin"
        RESULT=$(timeout 600 "$ROOT/build/uPIMulator" \
            --root_dirpath "$ROOT" --bin_dirpath "$ROOT/bin" \
            --benchmark BPMS \
            --num_channels $NUM_CHANNELS --num_ranks_per_channel $NUM_RANKS_PER_CHANNEL --num_dpus_per_rank $NUM_DPUS_PER_RANK \
            --num_tasklets $NR_TASKLETS --data_prep_params 8 \
            --num_simulation_threads $NUM_SIMULATION_THREADS 2>&1)

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
done

cp "$CYCLES_FILE" "$SCRIPT_DIR/layered_cycles_raw.txt"
rm -f "$CYCLES_FILE"
echo "TOTAL_DPUS=$PROJECTION_DPUS" > "$SCRIPT_DIR/layered_meta.txt"

echo ""
echo "================================================================================"
echo "  Phase 1 done. Copy BOTH files to wherever phase2 will run:"
echo "    $SCRIPT_DIR/layered_cycles_raw.txt"
echo "    $SCRIPT_DIR/layered_meta.txt"
echo "  Then run run_bp_increase_layered_phase2.sh there (needs real CPU wall-clock"
echo "  timing, so run phase2 on the machine you're actually comparing against)."
echo "================================================================================"
