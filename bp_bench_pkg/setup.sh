#!/bin/bash
# setup.sh — 의존성 설치 + 첫 빌드
# 사용법: ./setup.sh

set -e

echo "=== BP Bench Setup ==="
echo ""

# 의존성 설치
echo "[1/4] Installing dependencies..."
if command -v sudo &> /dev/null; then
    sudo apt update -qq
    sudo apt install -y -qq golang gcc cmake ninja-build python3 > /dev/null 2>&1
else
    apt update -qq 2>/dev/null
    apt install -y -qq golang gcc cmake ninja-build python3 > /dev/null 2>&1
fi
echo "  Done."

# Go 시뮬레이터 빌드
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/uPIMulator"

echo "[2/6] Setting permissions..."
chmod +x "$SCRIPT_DIR/qec_pim_package/sdk/bin/dpu-upmem-dpurte-clang"
chmod +x "$SCRIPT_DIR"/*.sh
echo "  Done."

echo "[3/6] Fixing hardcoded paths..."
cd "$SCRIPT_DIR" && python3 fix_paths.py
echo "  Done."

echo "[4/6] Fixing go.mod version..."
GO_VER=$(go version | grep -oP 'go\K[0-9]+\.[0-9]+')
sed -i "s/^go .*/go $GO_VER/" "$ROOT/go.mod"
echo "  go.mod set to go $GO_VER"

echo "[5/6] Building SDK..."
cd "$ROOT/sdk" && rm -rf build && mkdir build
cmake -D NR_TASKLETS=11 -S . -B build -G Ninja > /dev/null 2>&1
ninja -C build > /dev/null 2>&1
echo "  Done."

echo "[6/6] Building uPIMulator..."
mkdir -p "$ROOT/build"
cd "$ROOT/src" && go build -o "$ROOT/build/uPIMulator" . 2>&1
echo "  Done."

echo ""
echo "=== Setup complete ==="
echo ""
echo "Commands:"
echo "  ./run_bp_bench.sh            # 비최적화"
echo "  ./run_opt_bp_bench.sh        # 최적화"
echo "  ./run_adaptive_bp_bench.sh   # 적응"
echo "  ./run_compare_all.sh         # 세 모드 비교"
