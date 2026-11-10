#!/usr/bin/env python3
"""Build the paper-config CPU/DPU throughput model from checked-in results.

The DPU result is deliberately labelled as an instruction-level simulator/model
extrapolation.  It is not a hardware measurement.  CPU paper-config runs are
single-threaded; their 16-thread values reuse the measured scaling factor from
the repository's 1-thread/16-thread baseline instead of pretending that a new
16-thread measurement was made.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


BASELINE_CPU_1_RATE = 466.24400143522587
BASELINE_CPU_16_RATE = 3650.021928


def read_metric_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        return {row[0]: row[1] for row in list(csv.reader(f))[1:]}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fit_cycle_model(cycle_root: Path) -> dict:
    points = []
    for path in sorted(cycle_root.glob("*/summary.csv")):
        m = read_metric_csv(path)
        points.append({
            "path": str(path),
            "candidates": int(m["candidates"]),
            "expansions": int(m["expansions"]),
            "cycles": int(m["logic_cycles"]),
            "dpu_seconds": float(m["dpu_seconds"]),
            "beam": int(m["det_beam"]),
            "reverse_order": bool(int(m["reverse_order"])),
        })
    if len(points) < 3:
        raise ValueError(f"need at least 3 cycle traces, found {len(points)}")
    xs = [p["candidates"] for p in points]
    ys = [p["cycles"] for p in points]
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    denom = sum((x - xm) ** 2 for x in xs)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom
    intercept = ym - slope * xm
    constrained_to_origin = intercept < 0
    if constrained_to_origin:
        # A negative fixed launch cost is not physically meaningful and would
        # assign negative cycles to zero-candidate trials.  Use the non-negative
        # least-squares boundary solution (intercept == 0) in that case.
        intercept = 0.0
        slope = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - ym) ** 2 for y in ys)
    return {
        "trace_count": len(points),
        "intercept_cycles": intercept,
        "cycles_per_candidate": slope,
        "r_squared": 1 - ss_res / ss_tot if ss_tot else 1.0,
        "nonnegative_intercept_constraint_active": constrained_to_origin,
        "points": points,
    }


def sdk_host_cpu_per_trial(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    match = re.search(
        r"shots=(\d+).*?cpu_pack=([0-9.eE+-]+).*?cpu_h2d=([0-9.eE+-]+)"
        r".*?cpu_launch=([0-9.eE+-]+).*?cpu_d2h=([0-9.eE+-]+)",
        text,
    )
    if not match:
        raise ValueError(f"cannot parse host timing from {log_path}")
    trials = int(match.group(1))
    components = {
        "pack": float(match.group(2)),
        "h2d": float(match.group(3)),
        "launch": float(match.group(4)),
        "d2h": float(match.group(5)),
    }
    total = sum(components.values())
    return {
        "functional_sdk_trials": trials,
        "component_cpu_seconds": components,
        "total_cpu_seconds": total,
        "cpu_seconds_per_trial": total / trials,
    }


def aggregate_config(root: Path, name: str, cycle: dict, host: dict, args) -> dict:
    oracle_dir = root / f"{name}_oracle_10k"
    cpu_dir = root / f"{name}_cpu_10k"
    oracle_stats = json.loads((oracle_dir / "stats.json").read_text())
    cpu_stats = json.loads((cpu_dir / "stats.json").read_text())
    counts = {"resident": 0, "solved": 0, "low_confidence": 0, "fallback": 0,
              "cpu_restart": 0}
    resident_candidates = 0
    resident_expansions = 0
    cpu_restart_seconds = 0.0
    fallback_reasons: dict[str, int] = {}
    with (oracle_dir / "search_stats.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            kind = row["kind"]
            if kind == "resident_dpu":
                counts["resident"] += 1
                status = int(row["status"])
                if status == 0:
                    counts["solved"] += 1
                elif status == 1:
                    counts["low_confidence"] += 1
                elif status == 2:
                    counts["fallback"] += 1
                    reason = row["fallback_reason"]
                    fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
                resident_candidates += int(row["candidates"])
                resident_expansions += int(row["expansions"])
            elif kind == "cpu_search":
                counts["cpu_restart"] += 1
                cpu_restart_seconds += float(row["cpu_seconds"])

    shots = int(oracle_stats["num_shots"])
    trials_per_shot = int(oracle_stats["num_det_orders"])
    expected_trials = shots * trials_per_shot
    if counts["resident"] != expected_trials:
        raise ValueError(
            f"{name}: expected {expected_trials} resident trials, got {counts['resident']}"
        )
    cycles_total = (
        cycle["intercept_cycles"] * counts["resident"]
        + cycle["cycles_per_candidate"] * resident_candidates
    )
    cycles_per_shot = cycles_total / shots
    dpu_capacity = args.dpus * args.dpu_mhz * 1e6 / cycles_per_shot
    host_cpu_per_shot = (
        trials_per_shot * host["cpu_seconds_per_trial"]
        + cpu_restart_seconds / shots
    )
    host_capacity = args.cpu_workers / host_cpu_per_shot
    dpu_system_rate = min(dpu_capacity, host_capacity)

    cpu_1_rate = shots / float(cpu_stats["total_time_seconds"])
    measured_scaling = BASELINE_CPU_16_RATE / BASELINE_CPU_1_RATE
    cpu_16_estimate = cpu_1_rate * measured_scaling
    cpu_hash = sha256(cpu_dir / "predictions.01")
    dpu_hash = sha256(oracle_dir / "predictions.01")

    return {
        "name": name,
        "settings": {
            "beam_climbing": bool(oracle_stats["beam_climbing"]),
            "max_beam": int(oracle_stats["det_beam"]),
            "orders": trials_per_shot,
            "no_revisit_dets": bool(oracle_stats["no_revisit_dets"]),
            "pqlimit": int(oracle_stats["pqlimit"]),
        },
        "validation": {
            "shots": shots,
            "resident_trials": counts["resident"],
            "resident_solved": counts["solved"],
            "resident_low_confidence": counts["low_confidence"],
            "resident_fallbacks": counts["fallback"],
            "fallback_rate_per_trial": counts["fallback"] / counts["resident"],
            "fallback_reasons": fallback_reasons,
            "cpu_restarts": counts["cpu_restart"],
            "cpu_restart_seconds": cpu_restart_seconds,
            "resident_candidates": resident_candidates,
            "resident_expansions": resident_expansions,
            "cpu_logical_errors": int(cpu_stats["num_errors"]),
            "resident_logical_errors": int(oracle_stats["num_errors"]),
            "cpu_prediction_sha256": cpu_hash,
            "resident_prediction_sha256": dpu_hash,
            "predictions_identical": cpu_hash == dpu_hash,
        },
        "cpu": {
            "measured_1_thread_seconds": float(cpu_stats["total_time_seconds"]),
            "measured_1_thread_rate": cpu_1_rate,
            "baseline_measured_1_to_16_scaling": measured_scaling,
            "estimated_16_thread_rate": cpu_16_estimate,
        },
        "dpu_model": {
            "resident_cycles_per_shot": cycles_per_shot,
            "resident_seconds_per_shot_on_one_dpu": cycles_per_shot / (args.dpu_mhz * 1e6),
            "dpu_capacity_shots_per_second": dpu_capacity,
            "host_cpu_seconds_per_shot": host_cpu_per_shot,
            "host_capacity_shots_per_second": host_capacity,
            "system_rate_shots_per_second": dpu_system_rate,
            "binding_resource": "dpu" if dpu_capacity <= host_capacity else "host_cpu",
            "speedup_vs_estimated_cpu16": dpu_system_rate / cpu_16_estimate,
        },
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# Paper beam/order DPU validation and throughput",
        "",
        "The DPU numbers below are **uPIMulator cycle-simulator/model extrapolations**, "
        "not measurements on physical DPU hardware. Speedup is 2,560-DPU full-system "
        "throughput divided by the matching 16-thread CPU full-system throughput; it is "
        "not a per-core speed ratio.",
        "",
        "| paper configuration | CPU-only | 2,560-DPU offload | full-system speedup |",
        "|---|---:|---:|---:|",
    ]
    for c in result["configurations"]:
        s = c["settings"]
        lines.append(
            f"| {c['name']} (beam 0..{s['max_beam']}, {s['orders']} orders, "
            f"pqlimit {s['pqlimit']:,}) | {c['cpu']['estimated_16_thread_rate']:,.1f} shot/s "
            f"(model) | {c['dpu_model']['system_rate_shots_per_second']:,.1f} shot/s "
            f"(cycle model) | {c['dpu_model']['speedup_vs_estimated_cpu16']:.2f}× |"
        )
    lines += [
        "",
        "CPU paper-config values use measured single-thread runs and the repository's "
        "previously measured 1-to-16-thread scaling factor; they are estimates, not new "
        "16-thread measurements. The DPU model combines the 10,000-shot resident search "
        "population, functional SDK host timing, and the instruction-level cycle fit.",
        "",
        f"Cycle fit: {result['cycle_model']['trace_count']} traces, "
        f"cycles = {result['cycle_model']['intercept_cycles']:.1f} + "
        f"{result['cycle_model']['cycles_per_candidate']:.1f} × candidates, "
        f"R²={result['cycle_model']['r_squared']:.5f}.",
        "",
        "| configuration | validation | fallbacks | prediction equality | logical errors (CPU/DPU) |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in result["configurations"]:
        v = c["validation"]
        lines.append(
            f"| {c['name']} | {v['shots']:,} shots / {v['resident_trials']:,} trials | "
            f"{v['resident_fallbacks']:,} ({100*v['fallback_rate_per_trial']:.5f}%) | "
            f"{'yes' if v['predictions_identical'] else 'NO'} | "
            f"{v['cpu_logical_errors']}/{v['resident_logical_errors']} |"
        )
    lines += [
        "",
        "The cycle fit has only a small stratified trace set. These throughput values are "
        "therefore suitable as the current simulator-based estimate, but not as a hardware "
        "benchmark or a tightly bounded performance claim.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    p.add_argument("--dpus", type=int, default=2560)
    p.add_argument("--dpu-mhz", type=float, default=350.0)
    p.add_argument("--cpu-workers", type=int, default=16)
    args = p.parse_args()

    cycle = fit_cycle_model(args.results_root / "cycles")
    host = sdk_host_cpu_per_trial(args.results_root / "short_sdk4" / "dpu_live.log")
    result = {
        "method": {
            "dpu_result": "uPIMulator cycle-simulator/model extrapolation",
            "physical_dpu_hardware": False,
            "dpus": args.dpus,
            "dpu_mhz": args.dpu_mhz,
            "cpu_workers": args.cpu_workers,
            "speedup_definition": (
                "2560-DPU full-system throughput / matching 16-thread CPU full-system throughput"
            ),
        },
        "cycle_model": cycle,
        "functional_sdk_host": host,
        "configurations": [
            aggregate_config(args.results_root, "short", cycle, host, args),
            aggregate_config(args.results_root, "long", cycle, host, args),
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    args.output_md.write_text(render_markdown(result))


if __name__ == "__main__":
    main()
