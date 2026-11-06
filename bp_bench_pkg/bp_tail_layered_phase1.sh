#!/bin/bash
# bp_tail_layered_phase1.sh — DPU simulation ONLY (layered schedule, tail latency).
# Split out of bp_tail_layered.sh: 11 tasklets cooperate on one syndrome.
# Cycle counts are host-independent (deterministic simulation),
# so run this on whatever machine finishes fastest, then copy the two output
# files to wherever phase2 will run (phase2 needs real CPU wall-clock timing,
# so run that on the machine you're actually comparing against).
#
# Outputs (in this script's directory):
#   tail_layered_dpu_summary.txt — aggregate statistics per (p,iter)
#   tail_layered_meta.txt        — sweep metadata required by phase2
#
# 실행 방법:
#   ./bp_tail_layered_phase1.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/uPIMulator"
BPMS_GO="$ROOT/src/assembler/prim/bpms.go"
TASK_C="$ROOT/benchmark/BPMS/dpu/task.c"
NR_TASKLETS=11
MIN_DPU_ITER=1
MAX_DPU_ITER="${MAX_DPU_ITER:-5}"
ERROR_RATES="${ERROR_RATES:-0.001 0.003 0.005 0.007 0.009}"
N_DPU_RUNS="${N_DPU_RUNS:-10000}"
SEED_BASE=${SHARED_SEED:-$(date +%s)}

case "$MAX_DPU_ITER" in
    ''|*[!0-9]*|0) echo "MAX_DPU_ITER must be a positive integer" >&2; exit 1 ;;
esac
case "$N_DPU_RUNS" in
    ''|*[!0-9]*|0) echo "N_DPU_RUNS must be a positive integer" >&2; exit 1 ;;
esac

echo ""
echo "================================================================================"
echo "  [Phase 1 only] BP Tail Latency DPU simulation (LAYERED / Gauss-Seidel)"
echo "  Mapping: 11 tasklets cooperate on 1 syndrome"
echo "  iter=$MIN_DPU_ITER..$MAX_DPU_ITER, p in {$ERROR_RATES}, DPU runs/point=$N_DPU_RUNS"
echo "  Seed base: $SEED_BASE"
echo "================================================================================"
echo ""

# ---- 백업 + 복원 ----
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
    rm -f "${DPU_SUMMARY_TMP:-}" "${POINT_CYCLES_FILE:-}"
}
trap cleanup EXIT

cp "$ROOT/benchmark/BPMS/dpu/task_layered.c.bak" "$TASK_C"

DPU_SUMMARY_TMP=$(mktemp)

for P in $ERROR_RATES; do
    PRIOR=$(python3 -c "import math; print(int(math.log((1-$P)/$P)*256))")
  for DPU_ITER in $(seq "$MIN_DPU_ITER" "$MAX_DPU_ITER"); do
    echo "  p=$P  prior=$PRIOR  iter=$DPU_ITER"

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
	"os"
	"strconv"
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
	// p is the physical data-qubit error rate, not a syndrome-bit rate.
	// Generate a length-144 error vector and derive its 72-bit syndrome,
	// matching the CPU tail benchmark and the v2 host data generator.
	seed := int64($SEED_BASE)
	if seedText := os.Getenv("BP_TAIL_SEED"); seedText != "" {
		parsedSeed, err := strconv.ParseInt(seedText, 10, 64)
		if err != nil { panic(err) }
		seed = parsedSeed
	}
	rng := rand.New(rand.NewSource(seed))
	errVec := make([]byte, this.nv)
	for i := range errVec {
		if rng.Float64() < $P { errVec[i] = 1 }
	}
	this.syndromes = make([][]byte, 1)
	for t := 0; t < len(this.syndromes); t++ {
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

        cd "$ROOT/src" && go build -o "$ROOT/build/uPIMulator" . 2>/dev/null
        cd "$ROOT/benchmark" && rm -rf build && mkdir build
        cmake -D NR_DPUS=1 -D NR_TASKLETS=$NR_TASKLETS -S . -B build -G Ninja >/dev/null 2>&1
        ninja -C build BPMS_device >/dev/null 2>&1

        POINT_CYCLES_FILE=$(mktemp)
        for RUN in $(seq 1 "$N_DPU_RUNS"); do
        SEED=$((SEED_BASE + RUN * 997))
        rm -rf "$ROOT/bin" && mkdir -p "$ROOT/bin"
        BP_TAIL_SEED="$SEED" timeout 600 "$ROOT/build/uPIMulator" \
            --root_dirpath "$ROOT" --bin_dirpath "$ROOT/bin" \
            --benchmark BPMS \
            --num_channels 1 --num_ranks_per_channel 1 --num_dpus_per_rank 1 \
            --num_tasklets $NR_TASKLETS --data_prep_params 8 \
            --num_simulation_threads 4 >/dev/null 2>&1

        CYCLES=$(grep "logic_cycle" "$ROOT/bin/log.txt" 2>/dev/null | sed 's/.*: //')
        case "$CYCLES" in
            ''|*[!0-9]*) echo "invalid or missing DPU cycle count at p=$P iter=$DPU_ITER run=$RUN" >&2; exit 1 ;;
        esac
        echo "$CYCLES" >> "$POINT_CYCLES_FILE"
        if [ "$RUN" -eq 1 ] || [ "$RUN" -eq "$N_DPU_RUNS" ] || [ $((RUN % 100)) -eq 0 ]; then
            printf '\r    DPU progress: %d/%d (%d%%)' "$RUN" "$N_DPU_RUNS" $((RUN * 100 / N_DPU_RUNS))
        fi
    done
    printf '\n'

    python3 - "$POINT_CYCLES_FILE" "$P" "$DPU_ITER" >> "$DPU_SUMMARY_TMP" << 'PYEOF'
import math
import sys

path, p_text, iter_text = sys.argv[1:]
with open(path) as f:
    values = sorted(int(line) for line in f if line.strip())
if not values:
    raise SystemExit("no DPU cycle samples")

n = len(values)
mean = sum(values) / n
std = math.sqrt(sum((value - mean) ** 2 for value in values) / n)

def percentile(q):
    index = max(0, min(n - 1, math.ceil(q * n) - 1))
    return values[index]

# Full CSS latency: serialized X+Z, converted from 350-MHz cycles to ms.
scale = 2.0 / 350000.0
stats = [
    mean, std, percentile(0.50), percentile(0.90), percentile(0.99),
    percentile(0.999), percentile(0.9999), values[-1],
]
print(p_text, iter_text, n, *(f"{value * scale:.9f}" for value in stats))
PYEOF
    rm -f "$POINT_CYCLES_FILE"
    POINT_CYCLES_FILE=
    echo ""
  done
done

cp "$DPU_SUMMARY_TMP" "$SCRIPT_DIR/tail_layered_dpu_summary.txt"
rm -f "$DPU_SUMMARY_TMP"
DPU_SUMMARY_TMP=
{
    echo "MIN_DPU_ITER=$MIN_DPU_ITER"
    echo "MAX_DPU_ITER=$MAX_DPU_ITER"
    echo "N_DPU_RUNS=$N_DPU_RUNS"
    printf 'ERROR_RATES=%q\n' "$ERROR_RATES"
} > "$SCRIPT_DIR/tail_layered_meta.txt"

echo ""
echo "================================================================================"
echo "  Phase 1 done. Copy BOTH files to wherever phase2 will run:"
echo "    $SCRIPT_DIR/tail_layered_dpu_summary.txt"
echo "    $SCRIPT_DIR/tail_layered_meta.txt"
echo "  Individual DPU cycle samples were not retained."
echo "  Then run bp_tail_layered_phase2.sh there (needs real CPU wall-clock timing,"
echo "  so run phase2 on the machine you're actually comparing against)."
echo "================================================================================"
