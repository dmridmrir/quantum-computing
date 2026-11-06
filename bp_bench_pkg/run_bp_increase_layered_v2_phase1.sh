#!/usr/bin/env bash
# Self-contained exact BP-PIM cooperative v2 simulation, iter=1..10 sweep.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "================================================================================"
echo "  BP-PIM cooperative v2 (local exact copy): throughput reference"
echo "  11 tasklets -> 1 syndrome, iter=1..10, p=0.001..0.009"
echo "================================================================================"

/bin/bash "$SCRIPT_DIR/bp_tail_layered_v2_phase1.sh" \
    "$SCRIPT_DIR/layered_v2_cycles_raw.txt" \
    "$SCRIPT_DIR/layered_v2_meta.txt" \
    "${PROJECTION_DPUS:-2560}" \
    "run_bp_increase_layered_v2_phase2.sh"
