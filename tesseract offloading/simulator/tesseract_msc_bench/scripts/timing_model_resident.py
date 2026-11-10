#!/usr/bin/env python3
"""Cycle-based full-system throughput model for the DPU-resident A* search.

Same methodology as timing_model.py (uPIMulator cycles for the DPU, real
thread-CPU time for the host, no functional-simulator wall time).  Prefer
--resident-population-stats: it uses the actual resident-oracle DPU attempts
and CPU-resume rows, including non-capacity fallbacks such as the sibling
limit.  Without that input, --capacity is only a legacy approximation.

Inputs
  --cycles-dir     uPIMulator/results directory holding astar_dpu1_t*_trace_*/summary.csv
  --cpu-stats      per-shot CPU-only search_stats.csv over a long shot stream
  --resident-stats per-shot search_stats.csv from run_resident_search_stats.sh
                   (thread-CPU minus DPU wait = host cost per DPU-decoded shot)
  --resident-population-stats
                   long resident-oracle search_stats.csv. Rows with
                   dpu_batches>0 are DPU attempts; dpu_batches==0 are CPU
                   continuation work after a fallback.
  --cpu-rate       measured CPU-only full-system throughput (shot/s)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_summary(path: Path) -> dict[str, str]:
    return {r["metric"]: r["value"] for r in read_csv(path)}


def quantiles(values: list[float]) -> dict[str, float]:
    s = sorted(values)
    n = len(s)
    return {"mean": statistics.fmean(s), "p50": s[n // 2], "p90": s[int(0.9 * (n - 1))],
            "p99": s[int(0.99 * (n - 1))], "max": s[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cycles-dir", type=Path, required=True)
    parser.add_argument("--cycles-glob", default="astar_dpu1_t*_trace_*/summary.csv",
                        help="summary.csv glob under --cycles-dir (e.g. astar_dpu1_t16_g1_trace_*/summary.csv)")
    parser.add_argument("--trace-selection", type=Path,
                        help="selected_traces.csv; use stratified population weights and exclude "
                             "coverage-only top-tail/fallback additions from timing calibration")
    parser.add_argument("--cpu-stats", type=Path,
                        help="per-shot CPU stats (required unless --population-summary is used)")
    parser.add_argument("--resident-stats", type=Path, nargs="+", required=True)
    parser.add_argument("--resident-population-stats", type=Path,
                        help="long resident-oracle stats; enables actual-fallback population mode")
    parser.add_argument("--population-summary", type=Path,
                        help="streamed full-run summary from select_full_validation_traces.py; "
                             "avoids loading 100M CSV rows into memory")
    parser.add_argument("--cpu-rate", type=float, required=True)
    parser.add_argument("--cpu-workers", type=int, default=16)
    parser.add_argument("--dpus", type=int, default=2560)
    parser.add_argument("--mhz", type=float, default=350.0)
    parser.add_argument("--capacity", type=int, default=2000000,
                        help="DPU heap/arena capacity in nodes; shots pushing more fall back to CPU")
    parser.add_argument("--host-us-per-shot", type=float, default=None,
                        help="override measured host CPU per DPU-decoded shot (e.g. hardware estimate)")
    parser.add_argument("--resume", action="store_true",
                        help="fallback shots resume on the CPU from the exported DPU queue: the DPU "
                             "work is kept and the CPU only finishes the remaining fraction")
    parser.add_argument("--export-bandwidth-gbps", type=float, default=2.0,
                        help="deprecated; retained for command-line compatibility")
    parser.add_argument("--resume-export-us-per-fallback", type=float, default=0.0,
                        help="measured queue/arena export cost per actual fallback; 0 means uncharged")
    parser.add_argument("--groups-per-dpu", type=int, default=1,
                        help="independent syndromes decoded concurrently per DPU")
    parser.add_argument("--group-slowdown", type=float, default=1.0,
                        help="per-syndrome DPU time multiplier when sharing a DPU (measured pair cycles / sum of singles)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.groups_per_dpu == 1 and args.group_slowdown != 1.0:
        parser.error("--group-slowdown must be 1.0 for the one-syndrome-per-DPU model")
    if args.population_summary is not None and args.resident_population_stats is not None:
        parser.error("use only one of --population-summary and --resident-population-stats")
    if args.population_summary is None and args.cpu_stats is None:
        parser.error("--cpu-stats is required unless --population-summary is used")
    if (args.resident_population_stats is not None or args.population_summary is not None) \
            and args.groups_per_dpu != 1:
        parser.error("actual resident population stats currently support one syndrome per DPU only")

    # DPU cycles per candidate from the resident cycle sweep.
    all_traces = []
    for summary in sorted(args.cycles_dir.glob(args.cycles_glob)):
        s = read_summary(summary)
        if s.get("logic_cycles") and s.get("candidates"):
            all_traces.append((float(s["candidates"]), float(s["logic_cycles"]),
                               float(s["expansions"]), s.get("trace_file", "")))
    traces = [(c, y, e, 1.0) for c, y, e, _ in all_traces]
    if args.trace_selection is not None:
        weights = {}
        for row in read_csv(args.trace_selection):
            tags = set(row["selection_tags"].split(";"))
            if "stratified" in tags and row["sampling_weight"]:
                weights[int(row["global_shot_index"])] = float(row["sampling_weight"])
        traces = []
        for candidates, cycles, expansions, trace_file in all_traces:
            match = re.search(r"shot_(\d+)_", Path(trace_file).name)
            if match is None:
                match = re.search(r"shot_(\d+)_", trace_file)
            if match is None:
                continue
            shot = int(match.group(1))
            if shot in weights:
                traces.append((candidates, cycles, expansions, weights[shot]))
    if len(traces) < 3:
        raise SystemExit("need at least three finished resident cycle traces")
    t_cands = [t[0] for t in traces]
    t_cycles = [t[1] for t in traces]
    t_weights = [t[3] for t in traces]
    weight_sum = sum(t_weights)
    per_cand = sum(w * y for w, y in zip(t_weights, t_cycles)) / sum(
        w * x for w, x in zip(t_weights, t_cands)
    )
    xm = sum(w * x for w, x in zip(t_weights, t_cands)) / weight_sum
    ym = sum(w * y for w, y in zip(t_weights, t_cycles)) / weight_sum
    slope_denominator = sum(w * (x - xm) ** 2 for w, x in zip(t_weights, t_cands))
    slope = (sum(w * (x - xm) * (y - ym)
                 for w, x, y in zip(t_weights, t_cands, t_cycles)) / slope_denominator
             if slope_denominator else per_cand)
    intercept = max(ym - slope * xm, 0.0)

    # Host cost per DPU-decoded shot.
    host_costs = []
    for path in args.resident_stats:
        for r in read_csv(path):
            # A fallback with CPU continuation emits a second row with no DPU
            # batch.  It is population CPU work, not per-launch bridge cost.
            if float(r.get("dpu_batches", "1")) > 0:
                host_costs.append(float(r["cpu_seconds"]) - float(r["dpu_wait_cpu_seconds"]))
    if not host_costs:
        raise SystemExit("resident host stats contain no DPU-attempt rows")
    host_per_shot = statistics.fmean(host_costs)
    if args.host_us_per_shot is not None:
        host_per_shot = args.host_us_per_shot * 1e-6

    # Population: assign each shot to DPU or CPU.
    cpu_rows = read_csv(args.cpu_stats) if args.cpu_stats is not None else []
    cpu_seconds = [float(r["cpu_seconds"]) for r in cpu_rows]
    cands = [float(r["candidates"]) for r in cpu_rows]
    population_quantiles = quantiles(cands) if cands else None
    cpu_shots = len(cpu_rows)
    c_cpu = statistics.fmean(cpu_seconds) if cpu_seconds else None

    summary_population = None
    if args.population_summary is not None:
        summary_population = json.loads(args.population_summary.read_text())
        cpu_summary = summary_population["cpu"]
        resident_summary = summary_population["resident"]
        predictions = summary_population["predictions"]
        if not predictions["all_solution_cost_deltas_within_q16_bound"]:
            raise SystemExit("population summary exceeds the Q16 solution-cost error bound")
        cpu_shots = int(cpu_summary["shots"])
        c_cpu = float(cpu_summary["mean_cpu_seconds_per_shot"])
        population_quantiles = cpu_summary["candidate_quantiles"]
        if int(resident_summary["dpu_attempts"]) != cpu_shots:
            raise SystemExit("population summary CPU shots and DPU attempts differ")

    assert c_cpu is not None
    eff = args.cpu_rate * c_cpu / args.cpu_workers

    actual_dpu_candidates = None
    actual_dpu_candidate_sum = None
    actual_dpu_count = None
    actual_resume_cpu_seconds = None
    actual_fallbacks = None
    if summary_population is not None:
        resident_summary = summary_population["resident"]
        actual_dpu_count = int(resident_summary["dpu_attempts"])
        actual_dpu_candidate_sum = float(resident_summary["dpu_candidate_sum"])
        actual_resume_cpu_seconds = float(resident_summary["resume_cpu_seconds_sum"])
        actual_fallbacks = int(resident_summary["resume_rows"])
    elif args.resident_population_stats is not None:
        population_rows = read_csv(args.resident_population_stats)
        if not population_rows or "dpu_batches" not in population_rows[0]:
            raise SystemExit("resident population stats must contain the dpu_batches column")
        dpu_rows = [r for r in population_rows if float(r["dpu_batches"]) > 0]
        resume_rows = [r for r in population_rows if float(r["dpu_batches"]) == 0]
        if len(dpu_rows) != len(cpu_rows):
            raise SystemExit(
                f"resident population has {len(dpu_rows)} DPU attempts but CPU population has "
                f"{len(cpu_rows)} shots"
            )
        actual_dpu_candidates = [float(r["candidates"]) for r in dpu_rows]
        actual_dpu_candidate_sum = sum(actual_dpu_candidates)
        actual_dpu_count = len(actual_dpu_candidates)
        actual_resume_cpu_seconds = sum(float(r["cpu_seconds"]) for r in resume_rows)
        actual_fallbacks = len(resume_rows)

    def split(cycles_per_candidate: float) -> dict:
        if actual_dpu_count is not None:
            n = actual_dpu_count
            dpu_total = (
                n * intercept + cycles_per_candidate * actual_dpu_candidate_sum
            ) / (args.mhz * 1e6)
            export_total = actual_fallbacks * args.resume_export_us_per_fallback * 1e-6
            fallback_cpu = actual_resume_cpu_seconds + export_total
            host_total = n * host_per_shot + fallback_cpu
            mean_host = host_total / n
            mean_dpu = dpu_total / n
            cpu_bound = args.cpu_workers * eff / mean_host
            dpu_bound = args.dpus / mean_dpu
            system = min(cpu_bound, dpu_bound)
            return {
                "population_mode": "actual_resident_oracle",
                "fallback_shots": actual_fallbacks,
                "fallback_share_of_cpu_time": fallback_cpu / host_total,
                "resume_export_us_per_fallback": args.resume_export_us_per_fallback,
                "mean_host_seconds_per_shot": mean_host,
                "mean_dpu_seconds_per_shot": mean_dpu,
                "cpu_bound_per_s": cpu_bound,
                "dpu_bound_per_s": dpu_bound,
                "system_per_s": system,
                "speedup": system / args.cpu_rate,
                "speedup_scope": "2560-DPU modeled full throughput / measured 16-thread CPU throughput",
                "binding": "cpu" if cpu_bound <= dpu_bound else "dpu",
            }

        host_total = 0.0
        dpu_total = 0.0
        fallback = 0
        fallback_cpu = 0.0
        for c, s in zip(cands, cpu_seconds):
            if c > args.capacity:
                fallback += 1
                dpu_total += (intercept + cycles_per_candidate * args.capacity) / (args.mhz * 1e6)
                if args.resume:
                    # The CPU continues from the exported queue: only the
                    # remaining fraction of the search plus the export cost.
                    remaining = s * max(0.0, 1.0 - args.capacity / c)
                    export = args.resume_export_us_per_fallback * 1e-6
                    host_total += host_per_shot + export + remaining
                    fallback_cpu += remaining
                else:
                    # The DPU work is wasted and the CPU decodes from scratch.
                    host_total += s
                    fallback_cpu += s
            else:
                host_total += host_per_shot
                dpu_total += (intercept + cycles_per_candidate * c) / (args.mhz * 1e6)
        n = len(cands)
        mean_host = host_total / n
        mean_dpu = dpu_total / n
        cpu_bound = args.cpu_workers * eff / mean_host
        dpu_bound = args.dpus * args.groups_per_dpu / (mean_dpu * args.group_slowdown)
        system = min(cpu_bound, dpu_bound)
        return {
            "population_mode": "legacy_candidate_capacity_approximation",
            "fallback_shots": fallback,
            "fallback_share_of_cpu_time": fallback_cpu / host_total,
            "mean_host_seconds_per_shot": mean_host,
            "mean_dpu_seconds_per_shot": mean_dpu,
            "cpu_bound_per_s": cpu_bound,
            "dpu_bound_per_s": dpu_bound,
            "system_per_s": system,
            "speedup": system / args.cpu_rate,
            "speedup_scope": "modeled full-system throughput / measured full CPU throughput",
            "binding": "cpu" if cpu_bound <= dpu_bound else "dpu",
        }

    report = {
        "inputs": {
            "cycle_traces_finished": len(all_traces),
            "cycle_traces_in_weighted_calibration": len(traces),
            "trace_selection": str(args.trace_selection) if args.trace_selection else None,
            "cpu_only_shots": cpu_shots,
            "resident_host_shots": len(host_costs),
            "cpu_workers": args.cpu_workers,
            "dpus": args.dpus,
            "dpu_mhz": args.mhz,
            "capacity_nodes": args.capacity,
            "cpu_only_full_system_rate_per_s": args.cpu_rate,
            "population_source": (str(args.population_summary)
                                  if args.population_summary else
                                  str(args.resident_population_stats)
                                  if args.resident_population_stats else None),
            "result_kind": "model_extrapolation_not_measured_2560-DPU_throughput",
        },
        "dpu_kernel": {
            "cycles_per_candidate_ratio_of_sums": per_cand,
            "linear_fit": {"slope_cycles_per_candidate": slope, "intercept_cycles": intercept},
            "us_per_candidate": per_cand / args.mhz,
            "trace_dpu_ms": quantiles([c / (args.mhz * 1e3) for c in t_cycles]),
        },
        "host": {
            "measured_host_seconds_per_dpu_shot": statistics.fmean(host_costs),
            "used_host_seconds_per_dpu_shot": host_per_shot,
            "cpu_only_seconds_per_shot": c_cpu,
            "implied_parallel_efficiency": eff,
        },
        "population_cpu_candidates_per_shot": population_quantiles,
        "throughput_primary": split(per_cand),
        "throughput_pessimistic_slope": split(slope),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
