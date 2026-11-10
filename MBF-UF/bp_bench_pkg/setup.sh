#!/bin/bash
# Build the MBF-UF simulator, its runtime, and the BFUF DPU kernel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/uPIMulator"

for command in go gcc cmake ninja; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Missing dependency: $command" >&2
        echo "Install Go, GCC, CMake, and Ninja, then rerun this script." >&2
        exit 1
    fi
done

chmod +x "$SCRIPT_DIR/qec_pim_package/sdk/bin/dpu-upmem-dpurte-clang"
chmod +x "$SCRIPT_DIR/qec_pim_package/sdk/bin/clang"
chmod +x "$SCRIPT_DIR/run_bfuf_bench.sh"

echo "[1/3] Building the uPIMulator runtime..."
cmake -D NR_TASKLETS=11 -S "$ROOT/sdk" -B "$ROOT/sdk/build" -G Ninja
ninja -C "$ROOT/sdk/build"

echo "[2/3] Building uPIMulator..."
mkdir -p "$ROOT/build"
(cd "$ROOT/src" && go build -o "$ROOT/build/uPIMulator" .)

echo "[3/3] Building the BFUF DPU kernel..."
cmake -D NR_TASKLETS=11 -S "$ROOT/benchmark" -B "$ROOT/benchmark/build" -G Ninja
ninja -C "$ROOT/benchmark/build" BFUF_device

echo "Setup complete. Run: ./run_bfuf_bench.sh"
