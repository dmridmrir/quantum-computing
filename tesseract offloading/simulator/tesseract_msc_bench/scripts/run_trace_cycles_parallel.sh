#!/usr/bin/env bash
# Cycle-level uPIMulator runs for every trace in a directory, several at a time.
# Reuses the DPU program and simulator binary already built by build_and_run.sh
# for the requested tasklet count; each run gets its own bin directory.
#
# usage: run_trace_cycles_parallel.sh TRACE_DIR OUT_DIR [TASKLETS] [JOBS] [TIMEOUT_SECONDS]
set -euo pipefail

TRACE_DIR="$(realpath "$1")"
OUT_DIR="$(realpath -m "$2")"
TASKLETS="${3:-16}"
JOBS="${4:-8}"
TIMEOUT_SECONDS="${5:-21600}"
LOGIC_FREQUENCY_MHZ="${TESSERACT_LOGIC_FREQUENCY_MHZ:-350}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="${PROJECT_ROOT}/uPIMulator"
SIM_BIN="${SIM_ROOT}/build/uPIMulator"

if [[ ! -x "${SIM_BIN}" ]]; then
    echo "simulator binary missing; run build_and_run.sh ${TASKLETS} once first" >&2
    exit 1
fi
if ! grep -Eq "^NR_TASKLETS:[A-Z]+=${TASKLETS}$" "${SIM_ROOT}/benchmark/build/CMakeCache.txt"; then
    echo "benchmark build is not configured for NR_TASKLETS=${TASKLETS}; rerun build_and_run.sh ${TASKLETS}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

run_one() {
    local trace="$1"
    local name
    name="$(basename "${trace}" .json)"
    local bin_dir="${OUT_DIR}/${name}/bin"
    local start end status
    rm -rf -- "${bin_dir}"
    mkdir -p "${bin_dir}"
    start="$(date +%s)"
    set +e
    timeout "${TIMEOUT_SECONDS}" "${SIM_BIN}" \
        --root_dirpath "${SIM_ROOT}" \
        --bin_dirpath "${bin_dir}" \
        --benchmark TESSERACT \
        --num_channels 1 --num_ranks_per_channel 1 --num_dpus_per_rank 1 \
        --num_simulation_threads 1 \
        --num_tasklets "${TASKLETS}" \
        --logic_frequency "${LOGIC_FREQUENCY_MHZ}" \
        --data_prep_params 64,256,512,42,16 \
        --tesseract_trace "${trace}" > "${OUT_DIR}/${name}/stdout.log" 2>&1
    status=$?
    set -e
    end="$(date +%s)"
    local cycles instructions candidates
    cycles="$(awk -F': ' '/Logic\[[0-9]+_[0-9]+_[0-9]+\]_logic_cycle/ { print $2 + 0; exit }' "${bin_dir}/log.txt" 2>/dev/null || true)"
    instructions="$(awk -F': ' '/Logic\[[0-9]+_[0-9]+_[0-9]+\]_num_instructions/ { print $2 + 0; exit }' "${bin_dir}/log.txt" 2>/dev/null || true)"
    candidates="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["candidate_detector_offsets"]) - 1)' "${trace}")"
    echo "${name},${candidates},${TASKLETS},${status},${cycles:-},${instructions:-},$((end - start))" > "${OUT_DIR}/${name}/row.csv"
    echo "done ${name}: status=${status} cycles=${cycles:-NA} wall=$((end - start))s"
}
export -f run_one
export OUT_DIR TASKLETS TIMEOUT_SECONDS LOGIC_FREQUENCY_MHZ SIM_ROOT SIM_BIN

# Largest traces first so the long tail starts early.
python3 - "${TRACE_DIR}" <<'PY' | xargs -P "${JOBS}" -I{} bash -c 'run_one "$@"' _ {}
import json, os, sys
d = sys.argv[1]
rows = []
for f in os.listdir(d):
    if f.endswith(".json"):
        p = os.path.join(d, f)
        with open(p) as fh:
            rows.append((len(json.load(fh)["candidate_detector_offsets"]) - 1, p))
for _, p in sorted(rows, reverse=True):
    print(p)
PY

{
    echo "trace,candidates,tasklets,exit_status,logic_cycles,instructions,sim_wall_seconds"
    cat "${OUT_DIR}"/*/row.csv | sort -t, -k2,2n
} > "${OUT_DIR}/summary.csv"
echo "wrote ${OUT_DIR}/summary.csv"
