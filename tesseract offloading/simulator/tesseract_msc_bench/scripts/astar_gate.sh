#!/usr/bin/env bash
# Keep-only-if-faster gate for the resident A* kernel.
#  1. rebuild kernel (SDK + uPIMulator), bridge, Tesseract
#  2. SDK resident run on the 64-shot fixture; predictions must equal CPU-only
#  3. cycle-level run of a 4-trace subset (4 parallel); prints summed cycles
# usage: astar_gate.sh LABEL
set -euo pipefail
LABEL="${1:?label}"
BASELINE="${2:-141385457}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export GOFLAGS=-mod=mod
OUT="results/resident/gate_${LABEL}"
mkdir -p "${OUT}"

source upmem_sdk/upmem_env.sh simulator >/dev/null
dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS=16 -DTESSERACT_ASTAR_GROUPS=1 -DBL=8 -I uPIMulator/benchmark/TESSERACT_ASTAR/support \
    uPIMulator/benchmark/TESSERACT_ASTAR/dpu/task.c -o uPIMulator/benchmark/build/TESSERACT_ASTAR/host/tesseract_astar_dpu_t16
bash scripts/build_async_host.sh 16 >/dev/null
bash scripts/build_tesseract.sh >/dev/null 2>&1

W=workloads/msc_grafted_d3_d7
bash scripts/run_resident_search_stats.sh $W/model.dem $W/syndromes.01 $W/observables.01 "${OUT}/fixture64" 0 64 16 16 >/dev/null 2>&1
if cmp -s "${OUT}/fixture64/predictions.01" results/live_msc_grafted_d3_d7_all64/cpu/predictions.01; then
    echo "fixture64: predictions MATCH cpu-only"
else
    echo "fixture64: predictions MISMATCH"; exit 1
fi
grep -h "Tesseract DPU resident search" "${OUT}/fixture64/dpu_live.log"

# uPIMulator build once (kernel object + simulator), then run subset with SKIP_BUILD.
TESSERACT_TRACE=traces/msc_grafted_d3_d7_100m_sample32/shot_62500_q16.json bash scripts/build_and_run_astar.sh 16 > "${OUT}/build.log" 2>&1
export TESSERACT_ASTAR_SKIP_BUILD=1
for s in 71875 68750 15625; do
    (TESSERACT_TRACE=traces/msc_grafted_d3_d7_100m_sample32/shot_${s}_q16.json bash scripts/build_and_run_astar.sh 16 > "${OUT}/run_${s}.log" 2>&1; echo "shot_${s} exit=$?") &
done
wait
total=0
for s in 62500 71875 68750 15625; do
    c="$(awk -F, '/^logic_cycles,/ { print $2 }' "uPIMulator/results/astar_dpu1_t16_trace_shot_${s}_q16/summary.csv")"
    echo "shot_${s}: ${c}"
    total=$((total + c))
    cp "uPIMulator/results/astar_dpu1_t16_trace_shot_${s}_q16/summary.csv" "${OUT}/shot_${s}.csv"
done
echo "SUBSET_TOTAL_CYCLES ${total}  (baseline ${BASELINE}; ratio $(awk -v t="${total}" -v b="${BASELINE}" 'BEGIN { printf "%.4f", t / b }'))"
