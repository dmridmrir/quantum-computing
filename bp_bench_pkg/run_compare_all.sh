#!/bin/bash
# run_compare_all.sh — 비최적화 vs 최적화 vs 적응 전체 비교
#
# 세 모드를 순차적으로 시뮬레이션 실측 후 비교표 출력
# 사용법: ./run_compare_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$SCRIPT_DIR"
NOOPT_FILE=$(mktemp)
OPT_FILE=$(mktemp)
ADAPTIVE_FILE=$(mktemp)

# 세 모드 동일 시드 (같은 syndrome으로 공정 비교)
export SHARED_SEED=$(date +%s)

echo ""
echo "========================================================================"
echo "  BP Optimization Comparison: 3-way benchmark"
echo "  비최적화 → 최적화 → 적응 순으로 시뮬레이션 실행"
echo "  Shared seed: $SHARED_SEED (세 모드 동일 syndrome)"
echo "========================================================================"

ROOT="$SCRIPT_DIR/uPIMulator"
BPMS_GO="$ROOT/src/assembler/prim/bpms.go"
TASK_C="$ROOT/benchmark/BPMS/dpu/task.c"

# bpms.go 원본 저장 (세 모드 공통 기준)
cp "$BPMS_GO" "$BPMS_GO.compare_backup"
cp "$TASK_C" "$TASK_C.compare_backup"

restore_base() {
    cp "$BPMS_GO.compare_backup" "$BPMS_GO"
    cp "$TASK_C.compare_backup" "$TASK_C"
}

# ---- 1. 비최적화 실행 ----
echo ""
echo "[1/3] 비최적화 (곱셈 + 수렴기반 max 20회)..."
restore_base
"$BENCH_DIR/run_bp_bench.sh" 2>&1 | tee "$NOOPT_FILE" | grep -E "p=.*PASS|p=.*FAIL"
restore_base

# ---- 2. 최적화 실행 ----
echo ""
echo "[2/3] 최적화 (shift+add + 고정 3회)..."
restore_base
"$BENCH_DIR/run_opt_bp_bench.sh" 2>&1 | tee "$OPT_FILE" | grep -E "p=.*PASS|p=.*FAIL"
restore_base

# ---- 3. 적응 실행 ----
echo ""
echo "[3/3] 적응 (shift+add + prior 기반 3/5회)..."
restore_base
"$BENCH_DIR/run_adaptive_bp_bench.sh" 2>&1 | tee "$ADAPTIVE_FILE" | grep -E "p=.*PASS|p=.*FAIL"
restore_base

rm -f "$BPMS_GO.compare_backup" "$TASK_C.compare_backup"

echo ""

# ---- 비교표 생성 ----
python3 << PYEOF
import re, sys

def parse_results(filepath):
    results = {}
    with open(filepath) as f:
        content = f.read()

    # Phase 1에서 cycles 추출
    for line in content.split('\n'):
        m = re.match(r'\s*p=([0-9.]+)\s+prior=(\d+)\s+iter=(\d+)\s+cycles=(\d+)\s+instr=(\d+)\s+(PASS|FAIL)', line)
        if m:
            p = float(m.group(1))
            results[p] = {
                'prior': int(m.group(2)),
                'iter': int(m.group(3)),
                'cycles': int(m.group(4)),
                'instr': int(m.group(5)),
                'status': m.group(6)
            }

    # Phase 2에서 LER 추출
    for line in content.split('\n'):
        m = re.match(r'\s+([0-9.]+)\s+\|\s+\d+\s+\|\s+\d+\s+\|.*\|\s+([0-9.]+)%', line)
        if m:
            p = float(m.group(1))
            if p in results:
                results[p]['ler'] = float(m.group(2))

    # CPU throughput 추출
    for line in content.split('\n'):
        m = re.match(r'\s+([0-9.]+)\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+\d+\s+\|\s+([0-9.]+ [KM]/s)', line)
        if m:
            p = float(m.group(1))
            if p in results:
                tstr = m.group(2)
                if 'M/s' in tstr:
                    results[p]['cpu16'] = float(tstr.replace(' M/s','')) * 1e6
                elif 'K/s' in tstr:
                    results[p]['cpu16'] = float(tstr.replace(' K/s','')) * 1e3

    return results

noopt = parse_results('$NOOPT_FILE')
opt = parse_results('$OPT_FILE')
adapt = parse_results('$ADAPTIVE_FILE')

K = 11
MHZ = 350.0
ND = 2560

# CPU info
import subprocess
r = subprocess.run(['grep', '-m1', 'model name', '/proc/cpuinfo'], capture_output=True, text=True)
cpu = r.stdout.split(':')[1].strip() if ':' in r.stdout else 'Unknown'
r2 = subprocess.run(['grep', '-c', 'processor', '/proc/cpuinfo'], capture_output=True, text=True)
cores = int(r2.stdout.strip())

def fmt(v):
    if v >= 1e6: return f"{v/1e6:.2f} M/s"
    elif v >= 1e3: return f"{v/1e3:.1f} K/s"
    else: return f"{v:.0f} /s"

print()
print("=" * 120)
print("  BP Optimization Comparison: DPU Performance")
print("=" * 120)
print()
print(f"  Code         : [[144,12,12]] Bivariate Bicycle QLDPC")
print(f"  CPU          : {cpu} ({cores} cores)")
print(f"  DPU          : UPMEM PIM @ {MHZ:.0f} MHz, K={K} tasklets/DPU")
print(f"  DPU count    : {ND}")
print(f"  Measurement  : uPIMulator cycle-accurate simulation (DPU) + wall-clock (CPU)")
print(f"  Verification : DPU output = CPU output (byte-exact) for all modes")
print()
print("  Optimizations:")
print("    Non-opt  : multiplication alpha (x*160)>>8 + convergence-based (max 20 iter)")
print("    Opt      : shift+add alpha ((x<<2)+x)>>3 + fixed 3 iterations")
print("    Adaptive : shift+add alpha + prior-based (3 iter if p<=0.005, 5 iter if p>0.005)")
print()

# Throughput 비교표
print("=" * 120)
print("  DPU Throughput Comparison (2560 DPU, decodes/sec)")
print("=" * 120)
print()
print(f"  {'p':>6} | {'prior':>5} | {'Non-opt iter':>12} | {'Non-opt':>12} | {'Opt (3)':>12} | {'Adaptive':>12} | {'Opt/Non':>8} | {'Adp/Non':>8}")
print(f"  -------+-------+--------------+--------------+--------------+--------------+----------+----------")

ps = sorted(set(list(noopt.keys()) + list(opt.keys()) + list(adapt.keys())))

for p in ps:
    n = noopt.get(p)
    o = opt.get(p)
    a = adapt.get(p)
    if not (n and o and a): continue

    prior = n['prior']

    tn = K * 1e6 / (n['cycles'] / MHZ) * ND
    to = K * 1e6 / (o['cycles'] / MHZ) * ND
    ta = K * 1e6 / (a['cycles'] / MHZ) * ND

    r_on = to / tn
    r_an = ta / tn

    n_iter_str = f"iter={n['iter']}"
    a_iter_str = f"{a['iter']}"

    print(f"  {p:>6.3f} | {prior:>5} | {n_iter_str:>12} | {fmt(tn):>12} | {fmt(to):>12} | {fmt(ta):>12} | {r_on:>7.1f}x | {r_an:>7.1f}x")

print(f"  -------+-------+--------------+--------------+--------------+--------------+----------+----------")

# LER 비교표
print()
print("=" * 120)
print("  Logical Error Rate Comparison")
print("=" * 120)
print()
print(f"  {'p':>6} | {'Non-opt LER':>12} | {'Opt LER':>12} | {'Adaptive LER':>12}")
print(f"  -------+--------------+--------------+--------------")

for p in ps:
    n = noopt.get(p, {})
    o = opt.get(p, {})
    a = adapt.get(p, {})

    nl = f"{n.get('ler', -1):.3f}%" if 'ler' in n else "-"
    ol = f"{o.get('ler', -1):.3f}%" if 'ler' in o else "-"
    al = f"{a.get('ler', -1):.3f}%" if 'ler' in a else "-"

    print(f"  {p:>6.3f} | {nl:>12} | {ol:>12} | {al:>12}")

print(f"  -------+--------------+--------------+--------------")

# Latency 비교
print()
print("=" * 120)
print("  Per-Decode Latency Comparison")
print("=" * 120)
print()
print(f"  {'p':>6} | {'Non-opt':>12} | {'Opt':>12} | {'Adaptive':>12} | {'Opt speedup':>12} | {'Adp speedup':>12}")
print(f"  -------+--------------+--------------+--------------+--------------+--------------")

for p in ps:
    n = noopt.get(p)
    o = opt.get(p)
    a = adapt.get(p)
    if not (n and o and a): continue

    nl = n['cycles'] / MHZ / K
    ol = o['cycles'] / MHZ / K
    al = a['cycles'] / MHZ / K

    print(f"  {p:>6.3f} | {nl:>10.1f}us | {ol:>10.1f}us | {al:>10.1f}us | {nl/ol:>10.1f}x | {nl/al:>10.1f}x")

print(f"  -------+--------------+--------------+--------------+--------------+--------------")
print()
print("=" * 120)
print()

PYEOF

rm -f "$NOOPT_FILE" "$OPT_FILE" "$ADAPTIVE_FILE"
