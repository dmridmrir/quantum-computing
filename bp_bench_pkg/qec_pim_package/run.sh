#!/bin/bash
set -e

DIR=$(cd $(dirname $0) && pwd)
SDK=$DIR/sdk
SRC=$DIR/upmem_bp_bench

echo "=== QEC PIM Benchmark Setup ==="
echo "SDK: $SDK"
echo "SRC: $SRC"

# Setup SDK environment
source $SDK/upmem_env.sh simulator

# Compile DPU binary
echo "Compiling DPU UF decoder..."
dpu-upmem-dpurte-clang -O2 -w -DNR_TASKLETS=1 $SRC/uf_triage_dpu.c -o /tmp/uf_triage_binary
echo "DPU compile: OK"

# Compile host benchmark  
echo "Compiling host benchmark..."
gcc -O2 -o /tmp/multi_qubit_bench $SRC/multi_qubit_bench.c \
  -I$SDK/include/dpu -L$SDK/lib \
  -ldpu -ldpuverbose -lpthread -lelf -lnuma -lm
echo "Host compile: OK"

# Run
echo ""
echo "=== Running Benchmark ==="
LD_LIBRARY_PATH=$SDK/lib /tmp/multi_qubit_bench
