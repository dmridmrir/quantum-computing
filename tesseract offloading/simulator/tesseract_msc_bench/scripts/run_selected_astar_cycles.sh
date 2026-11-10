#!/usr/bin/env bash
# Sequential, resumable cycle-level replay of compact resident-A* traces.
# One simulator process is used at a time to preserve the repository's
# single-core validation policy.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 TRACE_DIR [OUTPUT_DIR]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_ROOT="${PROJECT_ROOT}/uPIMulator"
TRACE_DIR="$(realpath "$1")"
OUTPUT_DIR="$(realpath -m "${2:-${PROJECT_ROOT}/results/full_validation_100m/cycle_results}")"
TASKLETS="${TESSERACT_SELECTED_TASKLETS:-16}"
CAPACITY="${TESSERACT_SELECTED_CAPACITY:-1048576}"
DET_BEAM="${TESSERACT_SELECTED_DET_BEAM:--1}"
MAX_EXPANSIONS="${TESSERACT_SELECTED_MAX_EXPANSIONS:--1}"
MAX_DPU_CANDIDATES="${TESSERACT_SELECTED_MAX_DPU_CANDIDATES:-4096}"
mkdir -p "${OUTPUT_DIR}"

mapfile -d '' traces < <(find "${TRACE_DIR}" -maxdepth 1 -type f -name '*.json' -print0 | sort -z)
if (( ${#traces[@]} == 0 )); then
    echo "no JSON traces found in ${TRACE_DIR}" >&2
    exit 1
fi

need_build=1
for trace in "${traces[@]}"; do
    name="$(basename "${trace}" .json)"
    destination="${OUTPUT_DIR}/${name}"
    if [[ -s "${destination}/summary.csv" ]]; then
        echo "skip completed cycle trace: ${name}"
        continue
    fi
    mkdir -p "${destination}"
    if [[ "${name}" =~ _dpu_([0-9]+)$ ]] && (( BASH_REMATCH[1] > MAX_DPU_CANDIDATES )); then
        printf 'skipped_full_cycle_dpu_candidates=%s limit=%s\n' \
            "${BASH_REMATCH[1]}" "${MAX_DPU_CANDIDATES}" >"${destination}/status"
        echo "skip infeasible full cycle trace: ${name}"
        continue
    fi
    echo "cycle trace: ${name}"
    set +e
    TESSERACT_TRACE="${trace}" \
    TESSERACT_ASTAR_GROUPS=1 \
    TESSERACT_ASTAR_SKIP_BUILD="$((1 - need_build))" \
        "${SCRIPT_DIR}/build_and_run_astar.sh" \
        "${TASKLETS}" "${CAPACITY}" "${DET_BEAM}" "${MAX_EXPANSIONS}" \
        >"${destination}/driver.log" 2>&1
    status=$?
    set -e

    source_result="${SIM_ROOT}/results/astar_dpu1_t${TASKLETS}_g1_trace_${name}"
    if [[ -s "${source_result}/summary.csv" ]]; then
        cp -a "${source_result}/." "${destination}/"
        echo "PASS" >"${destination}/status"
    else
        printf 'exit_status=%d\n' "${status}" >"${destination}/status"
    fi
    if [[ -x "${SIM_ROOT}/build/uPIMulator" ]]; then
        need_build=0
    fi
done

python3 - "${OUTPUT_DIR}" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/summary.csv")):
    with path.open(newline="") as f:
        values = {row["metric"]: row["value"] for row in csv.DictReader(f)}
    values["trace"] = path.parent.name
    rows.append(values)
fields = [
    "trace", "trace_file", "tasklets", "groups", "capacity",
    "logic_frequency_mhz", "logic_cycles", "instructions", "expansions",
    "candidates", "dpu_seconds", "sim_wall_seconds",
]
with (root / "cycle_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"completed cycle traces: {len(rows)}/{len(list(root.glob('*/status')))}")
PY
