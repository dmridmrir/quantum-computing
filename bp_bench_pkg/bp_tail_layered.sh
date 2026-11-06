#!/usr/bin/env bash
# Compatibility entry point for the cooperative v1 tail workflow.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
/bin/bash "$SCRIPT_DIR/bp_tail_layered_phase1.sh"
/bin/bash "$SCRIPT_DIR/bp_tail_layered_phase2.sh"
