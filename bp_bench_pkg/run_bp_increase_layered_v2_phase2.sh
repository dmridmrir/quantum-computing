#!/usr/bin/env bash
# CPU throughput + LER + DPU combine, matching the v1 increase table.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_FILE="${1:-$SCRIPT_DIR/layered_v2_cycles_raw.txt}"
META_FILE="${2:-$SCRIPT_DIR/layered_v2_meta.txt}"
CPU_SOURCE="$SCRIPT_DIR/bp_increase_layered_v2_final.c"
SUPPORT="$SCRIPT_DIR/uPIMulator/benchmark/BPMS/support"

[[ -f "$RESULT_FILE" ]] || { echo "result file not found: $RESULT_FILE" >&2; exit 1; }
[[ -f "$META_FILE" ]] || { echo "meta file not found: $META_FILE" >&2; exit 1; }
[[ -f "$CPU_SOURCE" ]] || { echo "CPU benchmark source not found: $CPU_SOURCE" >&2; exit 1; }

# shellcheck disable=SC1090
source "$META_FILE"
TOTAL_DPUS="${PROJECTION_DPUS:-2560}"
TIMING_TRIALS="${TIMING_TRIALS:-50000}"
LER_TRIALS="${LER_TRIALS:-10000000}"
AGGREGATE="$(mktemp)"
BINARY="$(mktemp)"
trap 'rm -f "$AGGREGATE" "$BINARY"' EXIT

# Phase1 has repeated samples per p. v1 has one cycle point per configuration,
# so use the verified mean cycle count as the v2 representative point.
python3 - "$RESULT_FILE" >"$AGGREGATE" <<'PY'
import csv
import sys
from collections import defaultdict

groups = defaultdict(list)
with open(sys.argv[1], newline="") as f:
    rows = list(csv.DictReader(f))
for row in rows:
    if (row["group_size"], row["concurrent_syndromes"],
            row["posterior_exact"]) != ("11", "1", "yes"):
        raise SystemExit(f"invalid cooperative v2 row: {row}")
    iterations = int(row["iterations"])
    if not 1 <= iterations <= 10:
        raise SystemExit(f"iteration outside [1,10]: {iterations}")
    key = (float(row["p"]), int(row["prior_q8"]), iterations)
    groups[key].append(float(row["logic_cycles"]))
if not groups:
    raise SystemExit("no result rows")
expected = {(p, iterations) for p in (0.001, 0.003, 0.005, 0.007, 0.009)
            for iterations in range(1, 11)}
actual = {(p, iterations) for p, _prior, iterations in groups}
if actual != expected:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise SystemExit(f"incomplete sweep: missing={missing}, extra={extra}")
for (p, prior, iterations), values in sorted(groups.items()):
    print(f"{p:.3f} {prior} {iterations} {sum(values)/len(values):.6f}")
PY

gcc -O2 -fopenmp -I"$SUPPORT" -o "$BINARY" "$CPU_SOURCE" -lm
"$BINARY" "$AGGREGATE" "${SEED_BASE:-42}" "$TOTAL_DPUS" \
    "$TIMING_TRIALS" "$LER_TRIALS"
