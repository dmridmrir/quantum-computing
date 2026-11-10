#!/usr/bin/env python3
"""Cycle-based full-system throughput model for CPU + 2,560-DPU Tesseract.

Follows the BP-PIM methodology: DPU service time comes from uPIMulator logic
cycles at a fixed clock, CPU time comes from real host measurements (thread CPU
time, so it is immune to contention and clock drift), and the functional
simulator's wall time is never used.

Inputs
  --cycles         summary.csv from run_trace_cycles_parallel.sh
  --cpu-stats      search_stats.csv from a CPU-only --threads 1 run over a long
                   shot stream (TESSERACT_SEARCH_STATS_PATH); row i is shot i
  --live-stats     one or more search_stats.csv from run_live_search_stats.sh,
                   each with --shot-begin giving the first shot index of that file
  --cpu-rate       measured CPU-only full-system throughput (shot/s) for the
                   modeled CPU worker count on the same shot stream

Model
  c_cpu  = mean CPU-only thread-CPU seconds per shot (population)
  rho    = sum(hybrid CPU-active) / sum(CPU-only CPU) over matched shots
  cpu_bound = cpu_rate / rho                        (same parallel efficiency)
  d      = mean candidates per shot (population) * cycles per candidate / f
  dpu_bound = dpus / d
  system = min(cpu_bound, dpu_bound)
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def quantiles(values: list[float]) -> dict[str, float]:
    s = sorted(values)
    n = len(s)
    return {
        "mean": statistics.fmean(s),
        "p50": s[n // 2],
        "p90": s[int(0.9 * (n - 1))],
        "p99": s[int(0.99 * (n - 1))],
        "max": s[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cycles", type=Path, required=True)
    parser.add_argument("--cpu-stats", type=Path, required=True)
    parser.add_argument("--live-stats", type=Path, nargs="+", required=True)
    parser.add_argument("--shot-begin", type=int, nargs="+", required=True)
    parser.add_argument("--cpu-rate", type=float, required=True)
    parser.add_argument("--cpu-workers", type=int, default=16)
    parser.add_argument("--dpus", type=int, default=2560)
    parser.add_argument("--mhz", type=float, default=350.0)
    parser.add_argument("--launch-overhead-us", type=float, default=0.0,
                        help="host/DPU launch + transfer cost per DPU batch added to DPU time")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if len(args.live_stats) != len(args.shot_begin):
        parser.error("--live-stats and --shot-begin must have the same length")

    # DPU kernel cost per candidate from cycle-level traces.
    traces = [r for r in read_csv(args.cycles) if r["exit_status"] == "0" and r["logic_cycles"]]
    t_cands = [float(r["candidates"]) for r in traces]
    t_cycles = [float(r["logic_cycles"]) for r in traces]
    per_cand_ratio = sum(t_cycles) / sum(t_cands)
    xm, ym = statistics.fmean(t_cands), statistics.fmean(t_cycles)
    slope = sum((x - xm) * (y - ym) for x, y in zip(t_cands, t_cycles)) / sum((x - xm) ** 2 for x in t_cands)
    per_cand_each = [y / x for x, y in zip(t_cands, t_cycles)]

    # CPU-only population statistics.
    cpu_rows = read_csv(args.cpu_stats)
    cpu_seconds = [float(r["cpu_seconds"]) for r in cpu_rows]
    cpu_cands = [float(r["candidates"]) for r in cpu_rows]
    cpu_batches_est = [float(r["expansions"]) for r in cpu_rows]
    mu_cand = statistics.fmean(cpu_cands)
    c_cpu = statistics.fmean(cpu_seconds)

    # Hybrid CPU-active time on matched shots.
    matched_hybrid = 0.0
    matched_cpu = 0.0
    matched_cands_live = 0.0
    matched_cands_cpu = 0.0
    matched_shots = 0
    hybrid_rows_all: list[tuple[float, float, float]] = []  # (candidates, cpu_active, dpu_batches)
    for path, begin in zip(args.live_stats, args.shot_begin):
        for i, r in enumerate(read_csv(path)):
            shot = begin + i
            active = float(r["cpu_seconds"]) - float(r["dpu_wait_cpu_seconds"])
            hybrid_rows_all.append((float(r["candidates"]), active, float(r["dpu_batches"])))
            if shot < len(cpu_rows):
                matched_hybrid += active
                matched_cpu += cpu_seconds[shot]
                matched_cands_live += float(r["candidates"])
                matched_cands_cpu += cpu_cands[shot]
                matched_shots += 1
    if matched_shots == 0:
        raise SystemExit("no live shots overlap the CPU-only stats range")
    rho = matched_hybrid / matched_cpu
    cand_ratio_live_vs_cpu = matched_cands_live / matched_cands_cpu
    hybrid_active_per_cand = sum(a for _, a, _ in hybrid_rows_all) / sum(c for c, _, _ in hybrid_rows_all)
    batches_per_cand = sum(b for _, _, b in hybrid_rows_all) / sum(c for c, _, _ in hybrid_rows_all)

    # Population-level per-shot times.
    c_hybrid = c_cpu * rho
    cands_hybrid = mu_cand * cand_ratio_live_vs_cpu
    launches = cands_hybrid * batches_per_cand

    def dpu_seconds(cycles_per_candidate: float) -> float:
        return cands_hybrid * cycles_per_candidate / (args.mhz * 1e6) + launches * args.launch_overhead_us * 1e-6

    d_primary = dpu_seconds(per_cand_ratio)
    d_pessimistic = dpu_seconds(slope)

    cpu_bound = args.cpu_rate / rho
    dpu_bound = args.dpus / d_primary
    dpu_bound_pess = args.dpus / d_pessimistic
    system = min(cpu_bound, dpu_bound)
    system_pess = min(cpu_bound, dpu_bound_pess)

    report = {
        "inputs": {
            "cycle_traces": len(traces),
            "cpu_only_shots": len(cpu_rows),
            "matched_live_shots": matched_shots,
            "cpu_workers": args.cpu_workers,
            "dpus": args.dpus,
            "dpu_mhz": args.mhz,
            "cpu_only_full_system_rate_per_s": args.cpu_rate,
            "launch_overhead_us_per_batch": args.launch_overhead_us,
        },
        "dpu_kernel": {
            "cycles_per_candidate_ratio_of_sums": per_cand_ratio,
            "cycles_per_candidate_linear_slope": slope,
            "cycles_per_candidate_per_trace": quantiles(per_cand_each),
            "us_per_candidate_primary": per_cand_ratio / args.mhz,
        },
        "cpu_only_population": {
            "cpu_seconds_per_shot": quantiles(cpu_seconds),
            "candidates_per_shot": quantiles(cpu_cands),
            "expansions_per_shot_mean": statistics.fmean(cpu_batches_est),
            "single_thread_rate_per_s": 1.0 / c_cpu,
            "implied_parallel_efficiency": args.cpu_rate * c_cpu / args.cpu_workers,
        },
        "hybrid_matched_shots": {
            "cpu_active_over_cpu_only_ratio_rho": rho,
            "live_candidates_over_cpu_candidates": cand_ratio_live_vs_cpu,
            "cpu_active_seconds_per_candidate": hybrid_active_per_cand,
            "dpu_batches_per_candidate": batches_per_cand,
        },
        "per_shot_population": {
            "cpu_only_cpu_seconds": c_cpu,
            "hybrid_cpu_active_seconds": c_hybrid,
            "hybrid_candidates": cands_hybrid,
            "hybrid_dpu_batches": launches,
            "dpu_seconds_primary": d_primary,
            "dpu_seconds_pessimistic": d_pessimistic,
        },
        "throughput_per_s": {
            "cpu_only": args.cpu_rate,
            "hybrid_cpu_bound": cpu_bound,
            "hybrid_dpu_bound_primary": dpu_bound,
            "hybrid_dpu_bound_pessimistic": dpu_bound_pess,
            "hybrid_system_primary": system,
            "hybrid_system_pessimistic": system_pess,
            "speedup_primary": system / args.cpu_rate,
            "speedup_pessimistic": system_pess / args.cpu_rate,
            "binding_resource_primary": "cpu" if cpu_bound <= dpu_bound else "dpu",
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
