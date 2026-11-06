#!/usr/bin/env bash
# Self-contained BP-PIM cooperative v2 run.
# Fixed mapping: 11 tasklets cooperate on one syndrome; iter=1..10 sweep.
# Optional internal arguments:
#   bp_tail_layered_v2_phase1.sh [result_file meta_file [projection_dpus [report_hint]]]

set -euo pipefail

if [[ $# -eq 1 || $# -gt 4 ]]; then
    echo "Usage: $0 [result_file meta_file [projection_dpus [report_hint]]]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/uPIMulator"
SDK_ROOT="$SCRIPT_DIR/qec_pim_package/sdk"
CLANG="$SDK_ROOT/bin/clang"
SIM="$ROOT/build/uPIMulator"
TASK_C="$ROOT/benchmark/BPMS/dpu/task_layered_v2.c"
DEFAULT_TASK_C="$ROOT/benchmark/BPMS/dpu/task.c"
COLORS="$ROOT/benchmark/BPMS/support/bb144_colors.h"
SUPPORT="$ROOT/benchmark/BPMS/support"
ASM_OUT="$ROOT/benchmark/build/BPMS/dpu/CMakeFiles/BPMS_device.dir/task.c.o"
BPMS_GO="$ROOT/src/assembler/prim/bpms.go"
HOST_TEMPLATE="$ROOT/src/assembler/prim/bpms_colored_coop_host.go.src"
GOLDEN_MODEL="$ROOT/src/assembler/prim/bpms_layered_v2_verify.go"
RESULT_FILE="${1:-$SCRIPT_DIR/tail_layered_v2_cycles_raw.txt}"
META_FILE="${2:-$SCRIPT_DIR/tail_layered_v2_meta.txt}"
PROJECTION_DPUS="${3:-2560}"
REPORT_HINT="${4:-bp_tail_layered_v2_phase2.sh}"

readonly MAX_DPU_ITER=10
readonly NR_TASKLETS=11
readonly BP_GROUP_SIZE=11
readonly ERROR_RATES="0.001 0.003 0.005 0.007 0.009"
readonly N_DPU_RUNS="${N_DPU_RUNS:-15}"
readonly SEED_BASE="${SHARED_SEED:-42}"
readonly EXPECTED_KERNEL_SHA256="fa6170defe31d3bb9b40190f7cce80d1e98c1aba53a7b68b637b3ad90a5adaad"
readonly EXPECTED_COLORS_SHA256="f8549d04585c4842265c256e6ae5dfc796e8355d9c77eedff222d69242cc808a"

echo "================================================================================"
echo "  BP-PIM cooperative v2: exact local latency run"
echo "  11 tasklets -> 1 syndrome, iter=1..$MAX_DPU_ITER"
echo "  p in {$ERROR_RATES}, runs/p=$N_DPU_RUNS, seed_base=$SEED_BASE"
echo "================================================================================"

for required in "$CLANG" "$TASK_C" "$DEFAULT_TASK_C" "$COLORS" \
                "$BPMS_GO" "$HOST_TEMPLATE" "$GOLDEN_MODEL"; do
    if [[ ! -e "$required" ]]; then
        echo "required local reproduction file not found: $required" >&2
        exit 1
    fi
done

KERNEL_SHA256="$(sha256sum "$TASK_C" | awk '{print $1}')"
COLORS_SHA256="$(sha256sum "$COLORS" | awk '{print $1}')"
if [[ "$KERNEL_SHA256" != "$EXPECTED_KERNEL_SHA256" ]]; then
    echo "task_layered_v2.c is not the pinned BP-PIM task_colored_coop.c: $KERNEL_SHA256" >&2
    exit 1
fi
if [[ "$COLORS_SHA256" != "$EXPECTED_COLORS_SHA256" ]]; then
    echo "bb144_colors.h is not the pinned BP-PIM color table: $COLORS_SHA256" >&2
    exit 1
fi

LD_LOADER="${DPU_LD_LOADER:-/lib64/ld-linux-x86-64.so.2}"
if [[ ! -x "$LD_LOADER" ]]; then
    echo "DPU compiler loader not executable: $LD_LOADER" >&2
    exit 1
fi

BPMS_BACKUP="$(mktemp)"
RUN_ROOT="$(mktemp -d /tmp/bp_tail_layered_v2.XXXXXX)"
mkdir -p "$(dirname "$SIM")" "$(dirname "$ASM_OUT")"
cp "$BPMS_GO" "$BPMS_BACKUP"

compile_default() {
    "$LD_LOADER" "$CLANG" \
        --target=dpu-upmem-dpurte -mcpu=v1A -g -w \
        -I"$SUPPORT" -O2 -S -g0 -fno-stack-size-section \
        -DNR_TASKLETS=11 -DBL=8 "$DEFAULT_TASK_C" -o "$ASM_OUT"
}

restore() {
    cp "$BPMS_BACKUP" "$BPMS_GO" 2>/dev/null || true
    rm -f "$BPMS_BACKUP"
    compile_default >/dev/null 2>&1 || true
}
trap restore EXIT

# Use the fixed BP-PIM input generator only for this run. The trap restores
# the ordinary host implementation even when compilation or simulation fails.
cp "$HOST_TEMPLATE" "$BPMS_GO"

(
    cd "$ROOT/src"
    go test -count=1 ./assembler/prim ./assembler
    go build -o "$SIM" .
)

# This is the requested mapping: all 11 tasklets belong to one decoder group.
"$LD_LOADER" "$CLANG" \
    --target=dpu-upmem-dpurte -mcpu=v1A -g -w \
    -I"$SUPPORT" -O2 -S -g0 -fno-stack-size-section \
    -DNR_TASKLETS="$NR_TASKLETS" -DBL=8 -DBP_GROUP_SIZE="$BP_GROUP_SIZE" \
    "$TASK_C" -o "$ASM_OUT"

mkdir -p "$(dirname "$RESULT_FILE")" "$(dirname "$META_FILE")"
echo "p,error_ppm,prior_q8,run,seed,group_size,concurrent_syndromes,iterations,logic_cycles,latency_ms_at_350MHz,instructions,throughput_dec_per_s,posterior_exact" >"$RESULT_FILE"

for P in $ERROR_RATES; do
    PRIOR_Q8="$(python3 -c "import math; p=float('$P'); print(int(math.log((1-p)/p)*256))")"
    ERROR_PPM="$(python3 -c "print(int(round(float('$P')*1000000)))")"
    echo "p=$P prior=$PRIOR_Q8 error_ppm=$ERROR_PPM"

    for DPU_ITER in $(seq 1 "$MAX_DPU_ITER"); do
        echo "  iter=$DPU_ITER"
        for RUN in $(seq 1 "$N_DPU_RUNS"); do
            INPUT_SEED=$((SEED_BASE + (RUN - 1) * 997))
            OUT="$RUN_ROOT/p${P}_iter${DPU_ITER}_run${RUN}"
            mkdir -p "$OUT"
            PARAMS="$PRIOR_Q8,$DPU_ITER,$ERROR_PPM,$INPUT_SEED"

            if ! "$LD_LOADER" "$SIM" \
                --root_dirpath "$ROOT" --bin_dirpath "$OUT" \
                --benchmark BPMS \
                --num_channels 1 --num_ranks_per_channel 1 --num_dpus_per_rank 1 \
                --num_tasklets "$NR_TASKLETS" --data_prep_params "$PARAMS" \
                --num_simulation_threads 4 >"$OUT/stdout.txt" 2>&1; then
                cat "$OUT/stdout.txt" >&2
                exit 1
            fi

            if grep -Eq 'MISMATCH|Total mismatches|panic' "$OUT/stdout.txt"; then
                cat "$OUT/stdout.txt" >&2
                echo "CPU-DPU posterior comparison failed for p=$P iter=$DPU_ITER run=$RUN" >&2
                exit 1
            fi

            CYCLES="$(awk -F': ' '/_logic_cycle:/{print $2; exit}' "$OUT/log.txt")"
            INSTRUCTIONS="$(awk -F': ' '/_num_instructions:/{print $2; exit}' "$OUT/log.txt")"
            if [[ -z "$CYCLES" || -z "$INSTRUCTIONS" ]]; then
                echo "missing counters for p=$P iter=$DPU_ITER run=$RUN in $OUT/log.txt" >&2
                exit 1
            fi

            LATENCY_MS="$(awk -v c="$CYCLES" 'BEGIN {printf "%.6f", c/350000.0}')"
            THROUGHPUT="$(awk -v c="$CYCLES" 'BEGIN {printf "%.1f", 350000000.0/c}')"
            echo "$P,$ERROR_PPM,$PRIOR_Q8,$RUN,$INPUT_SEED,$BP_GROUP_SIZE,1,$DPU_ITER,$CYCLES,$LATENCY_MS,$INSTRUCTIONS,$THROUGHPUT,yes" >>"$RESULT_FILE"
            echo "    run=$RUN cycles=$CYCLES latency_ms=$LATENCY_MS"
        done
    done
done

GOLDEN_SHA256="$(sha256sum "$GOLDEN_MODEL" | awk '{print $1}')"
HOST_SHA256="$(sha256sum "$HOST_TEMPLATE" | awk '{print $1}')"
{
    echo "MODE=bp_pim_colored_cooperative_exact_local"
    echo "ITERATIONS=\"1 2 3 4 5 6 7 8 9 10\""
    echo "MAX_DPU_ITER=$MAX_DPU_ITER"
    echo "NR_TASKLETS=$NR_TASKLETS"
    echo "BP_GROUP_SIZE=$BP_GROUP_SIZE"
    echo "CONCURRENT_SYNDROMES=1"
    echo "ERROR_RATES=\"$ERROR_RATES\""
    echo "N_DPU_RUNS=$N_DPU_RUNS"
    echo "SEED_BASE=$SEED_BASE"
    echo "PROJECTION_DPUS=$PROJECTION_DPUS"
    echo "KERNEL_SHA256=$KERNEL_SHA256"
    echo "COLORS_SHA256=$COLORS_SHA256"
    echo "GOLDEN_SHA256=$GOLDEN_SHA256"
    echo "HOST_SHA256=$HOST_SHA256"
    echo "RAW_SIMULATOR_DIR=$RUN_ROOT"
} >"$META_FILE"

cat "$RESULT_FILE"
echo "raw simulator outputs: $RUN_ROOT" >&2
echo "Run $REPORT_HINT to print the result interpretation."
