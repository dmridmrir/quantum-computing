#!/bin/bash
# Minimal repro: run DPU cycle measurement for p=0.003 and p=0.005, iter=1,
# using the exact same patching approach as run_bp_increase.sh, to check
# whether the "identical DPU cycles across p" phenomenon in Table 1 reproduces
# from a live uPIMulator run (not a stale/copied cycles file).
set -e

ROOT="/home/bp_bench_pkg/uPIMulator"
BPMS_GO="$ROOT/src/assembler/prim/bpms.go"
TASK_C="$ROOT/benchmark/BPMS/dpu/task.c"
NR_TASKLETS=11
DPU_ITER=${DPU_ITER:-1}

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

P=${FIXED_P:-0.007}
for SEED_BASE in 1 2 3 4 5 6 7 8 9 10; do
    PRIOR=$(python3 -c "import math; print(int(math.log((1-$P)/$P)*256))")

    cp "$ROOT/benchmark/BPMS/dpu/task_opt.c.bak" "$TASK_C"
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

    cd "$ROOT/src" && go build -o "$ROOT/build/uPIMulator" . 2>&1
    cd "$ROOT/benchmark" && rm -rf build && mkdir build
    cmake -D NR_DPUS=1 -D NR_TASKLETS=$NR_TASKLETS -S . -B build -G Ninja >/dev/null 2>&1
    ninja -C build BPMS_device >/dev/null 2>&1

    rm -rf "$ROOT/bin" && mkdir -p "$ROOT/bin"
    RESULT=$(timeout 600 "$ROOT/build/uPIMulator" \
        --root_dirpath "$ROOT" --bin_dirpath "$ROOT/bin" \
        --benchmark BPMS \
        --num_channels 1 --num_ranks_per_channel 1 --num_dpus_per_rank 1 \
        --num_tasklets $NR_TASKLETS --data_prep_params 8 \
        --num_simulation_threads 4 2>&1)

    CYCLES=$(grep "logic_cycle" "$ROOT/bin/log.txt" 2>/dev/null | sed 's/.*: //')
    echo "p=$P prior=$PRIOR iter=$DPU_ITER cycles=$CYCLES"
    if echo "$RESULT" | grep -q "MISMATCH\|panic"; then
        echo "  (DPU/CPU MISMATCH detected in this run)"
    fi
done
