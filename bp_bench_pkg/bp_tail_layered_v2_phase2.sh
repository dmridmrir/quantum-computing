#!/usr/bin/env bash
# Summarize the 11-tasklet cooperative v2 latency sweep from phase1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_FILE="${1:-$SCRIPT_DIR/tail_layered_v2_cycles_raw.txt}"
META_FILE="${2:-$SCRIPT_DIR/tail_layered_v2_meta.txt}"

[[ -f "$RESULT_FILE" ]] || { echo "result file not found: $RESULT_FILE" >&2; exit 1; }
[[ -f "$META_FILE" ]] || { echo "meta file not found: $META_FILE" >&2; exit 1; }

python3 - "$RESULT_FILE" <<'PY'
import csv
import math
import sys
from collections import defaultdict

path = sys.argv[1]
required = {
    "p", "error_ppm", "prior_q8", "run", "seed", "group_size",
    "concurrent_syndromes", "iterations", "logic_cycles",
    "latency_ms_at_350MHz", "instructions", "throughput_dec_per_s",
    "posterior_exact",
}
with open(path, newline="") as f:
    reader = csv.DictReader(f)
    if set(reader.fieldnames or ()) != required:
        raise SystemExit(f"unexpected result header: {reader.fieldnames}")
    rows = list(reader)

if not rows:
    raise SystemExit("no result rows")

by_point = defaultdict(list)
for row in rows:
    if (row["group_size"], row["concurrent_syndromes"],
            row["posterior_exact"]) != ("11", "1", "yes"):
        raise SystemExit(f"invalid v2 result row: {row}")
    p = float(row["p"])
    if p < 0.001 or p > 0.009:
        raise SystemExit(f"p outside project range: {p}")
    iterations = int(row["iterations"])
    if not 1 <= iterations <= 10:
        raise SystemExit(f"iteration outside [1,10]: {iterations}")
    # Phase1 stores one CSS-component latency. The project tail-latency
    # convention is a serialized full CSS decode, so report X+Z = 2x here.
    by_point[(p, iterations)].append(2.0 * float(row["latency_ms_at_350MHz"]))

def percentile(values, q):
    values = sorted(values)
    index = max(0, math.ceil(q * len(values)) - 1)
    return values[index]

sample_counts = sorted({len(values) for values in by_point.values()})
sample_text = str(sample_counts[0]) if len(sample_counts) == 1 else str(sample_counts)

print("=" * 143)
print("  BP-PIM cooperative v2 tail latency: 11 tasklets -> 1 syndrome, iter=1..10")
print("  Metric: serialized full CSS decode latency, X+Z = 2 x component logic latency")
print(f"  Samples per (p,iter): {sample_text}; clock: 350 MHz")
print("=" * 143)
print(f"  {'p':>7} {'iter':>4} {'prior':>7} {'mean ms':>10} {'std ms':>10} {'p50 ms':>10} "
      f"{'p90 ms':>10} {'p99 ms':>10} {'p99.9 ms':>11} {'p99.99 ms':>12} "
      f"{'max ms':>10} {'<1ms':>7}")
print("  " + "-" * 139)
previous_p = None
for p, iterations in sorted(by_point):
    if previous_p is not None and p != previous_p:
        print("  " + "-" * 139)
    previous_p = p
    values = by_point[(p, iterations)]
    prior = int(math.log((1-p)/p) * 256)
    mean = sum(values) / len(values)
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
    maximum = max(values)
    passed = "PASS" if maximum < 1.0 else "FAIL"
    print(f"  {p:7.3f} {iterations:4d} {prior:7d} {mean:10.6f} {std:10.6f} "
          f"{percentile(values, .50):10.6f} {percentile(values, .90):10.6f} "
          f"{percentile(values, .99):10.6f} {percentile(values, .999):11.6f} "
          f"{percentile(values, .9999):12.6f} {maximum:10.6f} {passed:>7}")
print("=" * 143)
print("  <1ms PASS requires the maximum sampled serialized X+Z latency to be below 1 ms.")
print("  With only 15 samples/(p,iter), p99, p99.9, and p99.99 collapse to the sampled max;")
print("  reliable high-percentile estimation requires substantially more samples.")
print("  Excluded: host transfer, runtime launch, program load, controller integration.")
PY
