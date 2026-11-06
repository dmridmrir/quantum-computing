#!/usr/bin/env bash
# Compatibility entry point for the cooperative v1 throughput/LER workflow.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
/bin/bash "$SCRIPT_DIR/run_bp_increase_layered_phase1.sh"
/bin/bash "$SCRIPT_DIR/run_bp_increase_layered_phase2.sh"
