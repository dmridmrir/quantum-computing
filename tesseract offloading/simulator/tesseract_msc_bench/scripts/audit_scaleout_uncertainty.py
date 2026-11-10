#!/usr/bin/env python3
"""Audit the small-DPU scale-out runs and the paper-config DPU model.

This is a reproducible *model sensitivity* analysis, not a physical UPMEM
utilization measurement.  In particular, the 64-shot functional-simulator
wall times and the 10,000-shot cycle model are never fitted as one series.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
import statistics
from pathlib import Path

from fit_scaleout import fit_nonnegative_line


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
OLD_RUN = "scaleout_extrapolation_64shot_lower_bound_t16_2026-09-16"
PAPER = RESULTS / "paper_dpu_validation"
POWER = RESULTS / "power" / "final" / "energy_comparison.json"


def old_replicates() -> list[dict]:
    rows = []
    for run, suffix in enumerate(("", "_r2", "_r3"), start=1):
        path = RESULTS / f"{OLD_RUN}{suffix}" / "measurements.csv"
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                rows.append(
                    {
                        "run": run,
                        "cpu_workers": int(row["cpu_threads"]),
                        "dpus": int(row["dpu_count"]),
                        "shots": int(row["shots"]),
                        "wall_seconds": float(row["process_wall_seconds"]),
                        "throughput": float(
                            row["hybrid_end_to_end_throughput_per_second"]
                        ),
                        "predictions_match_cpu": row["predictions_match_cpu"] == "true",
                    }
                )
    return rows


def summarize_replicates(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["cpu_workers"], row["dpus"]), []).append(row)
    summary = []
    for (cpu, dpus), group in sorted(groups.items()):
        rates = [row["throughput"] for row in group]
        assert len(group) == 3
        assert all(row["shots"] == 64 and row["predictions_match_cpu"] for row in group)
        mean = statistics.mean(rates)
        summary.append(
            {
                "cpu_workers": cpu,
                "dpus": dpus,
                "runs": len(group),
                "mean_throughput": mean,
                "median_throughput": statistics.median(rates),
                "min_throughput": min(rates),
                "max_throughput": max(rates),
                "sample_sd": statistics.stdev(rates),
                "coefficient_of_variation_percent": 100 * statistics.stdev(rates) / mean,
            }
        )
    return summary


def legacy_extrapolations(rows: list[dict], target_dpus: int = 2560) -> list[dict]:
    """Reproduce the old fit and expose its run/window sensitivity."""
    cpu16 = [row for row in rows if row["cpu_workers"] == 16]
    runs = sorted({row["run"] for row in cpu16})
    medians = []
    for dpus in (4, 8, 16, 32, 64):
        group = [row for row in cpu16 if row["dpus"] == dpus]
        medians.append(
            {"dpus": dpus, "shots": 64, "wall_seconds": statistics.median(row["wall_seconds"] for row in group)}
        )

    def fit(source: list[dict], label: str) -> dict:
        xs = [1 / row["dpus"] for row in source]
        ys = [row["wall_seconds"] / row["shots"] for row in source]
        intercept, slope = fit_nonnegative_line(xs, ys)
        residual = sum((actual - intercept - slope * x) ** 2 for actual, x in zip(ys, xs))
        total = sum((actual - statistics.mean(ys)) ** 2 for actual in ys)
        return {
            "fit": label,
            "dpus_used": ";".join(str(row["dpus"]) for row in source),
            "serial_seconds_per_shot": intercept,
            "parallel_seconds_per_shot_at_one_dpu": slope,
            "r_squared_in_sample": 1 - residual / total if total else 1.0,
            "extrapolated_2560_syndromes_per_second": 1 / (intercept + slope / target_dpus),
        }

    result = [fit(medians, "five_point_median")]
    result.extend(fit([row for row in cpu16 if row["run"] == run], f"run_{run}") for run in runs)
    result.append(fit([row for row in medians if row["dpus"] in (4, 8, 16)], "low_count_window"))
    result.append(fit([row for row in medians if row["dpus"] in (16, 32, 64)], "high_count_window"))
    return result


def read_shot_candidates(path: Path, trials_per_shot: int) -> tuple[list[int], dict]:
    shots = []
    trial_candidates = []
    total = 0
    resident_trials = 0
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["kind"] != "resident_dpu":
                continue
            order_index = int(row["order_index"])
            if order_index != resident_trials % trials_per_shot:
                raise ValueError(
                    f"trial order mismatch at resident row {resident_trials}: {order_index}"
                )
            candidates = int(row["candidates"])
            total += candidates
            trial_candidates.append(candidates)
            resident_trials += 1
            if resident_trials % trials_per_shot == 0:
                shots.append(total)
                total = 0
    if total or resident_trials != len(shots) * trials_per_shot:
        raise ValueError("incomplete final shot")
    return shots, {
        "shots": len(shots),
        "resident_trials": resident_trials,
        "max_trial_candidates": max(trial_candidates),
        "max_shot_candidates": max(shots),
        "trial_candidates": trial_candidates,
    }


def fit_cycles(points: list[dict]) -> tuple[float, float]:
    x = [float(point["candidates"]) for point in points]
    y = [float(point["cycles"]) for point in points]
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 0:
        raise ValueError("cycle trace candidates have zero variance")
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denominator
    intercept = y_mean - slope * x_mean
    if intercept < 0:
        intercept = 0.0
        slope = sum(a * b for a, b in zip(x, y)) / sum(a * a for a in x)
    return intercept, slope


def schedule(jobs: list[float], dpus: int) -> dict:
    """Optimistic free-DPU dispatch, with no host/rank/transfer overhead."""
    if not 1 <= dpus <= len(jobs):
        raise ValueError("DPU count must be within the shot count")
    finish_times = [0.0] * dpus
    heapq.heapify(finish_times)
    for duration in jobs:
        heapq.heapreplace(finish_times, finish_times[0] + duration)
    makespan = max(finish_times)
    busy = sum(jobs)
    return {
        "dpus": dpus,
        "shots": len(jobs),
        "compute_busy_seconds": busy,
        "compute_only_makespan_seconds": makespan,
        "mean_active_dpu_fraction": busy / (dpus * makespan),
        "compute_only_throughput_shots_per_second": len(jobs) / makespan,
    }


def quantile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return sorted_values[low] + (position - low) * (sorted_values[high] - sorted_values[low])


def predictive_bootstrap(
    shot_times: list[float], dpus: int, target_shots: int, repetitions: int, seed: int
) -> dict:
    rng = random.Random(seed)
    utilization = []
    throughput = []
    makespan = []
    for _ in range(repetitions):
        sampled_jobs = rng.choices(shot_times, k=target_shots)
        result = schedule(sampled_jobs, dpus)
        utilization.append(result["mean_active_dpu_fraction"])
        throughput.append(result["compute_only_throughput_shots_per_second"])
        makespan.append(result["compute_only_makespan_seconds"])
    def interval(values: list[float]) -> dict:
        ordered = sorted(values)
        return {
            "p025": quantile(ordered, 0.025),
            "median": quantile(ordered, 0.5),
            "p975": quantile(ordered, 0.975),
        }
    return {
        "method": "predictive resampling from observed 10k shot mix; fixed four-trace cycle fit",
        "seed": seed,
        "replications": repetitions,
        "target_shots": target_shots,
        "dpus": dpus,
        "utilization": interval(utilization),
        "throughput_shots_per_second": interval(throughput),
        "compute_only_makespan_seconds": interval(makespan),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "scaleout_statistical_audit")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--target-shots", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if args.bootstrap < 20 or args.target_shots < 2560:
        parser.error("bootstrap must be >=20 and target shots >=2560")

    timing = json.loads((PAPER / "timing_model_paper.json").read_text(encoding="utf-8"))
    short = next(row for row in timing["configurations"] if row["name"] == "short")
    trials = short["settings"]["orders"]
    candidates, shot_meta = read_shot_candidates(
        PAPER / "short_oracle_10k" / "search_stats.csv", trials
    )
    assert len(candidates) == short["validation"]["shots"]
    assert sum(candidates) == short["validation"]["resident_candidates"]
    cycle = timing["cycle_model"]
    frequency = timing["method"]["dpu_mhz"] * 1e6
    intercept = cycle["intercept_cycles"]
    slope = cycle["cycles_per_candidate"]
    shot_times = [(trials * intercept + value * slope) / frequency for value in candidates]
    assert math.isclose(statistics.mean(shot_times), short["dpu_model"]["resident_seconds_per_shot_on_one_dpu"], rel_tol=1e-10)

    legacy_syndromes = (PROJECT / "workloads" / "msc_grafted_d3_d7" / "syndromes.01").read_text(encoding="ascii").splitlines()
    if len(legacy_syndromes) != 64 or len(set(legacy_syndromes)) != 64:
        raise ValueError("legacy fixture must contain 64 distinct syndromes")
    old_rows = old_replicates()
    repetitions = summarize_replicates(old_rows)
    old_fit = legacy_extrapolations(old_rows)
    scaling = []
    for dpus in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 2560):
        row = schedule(shot_times, dpus)
        row["ideal_average_work_throughput_shots_per_second"] = dpus / statistics.mean(shot_times)
        scaling.append(row)
    replay_jobs = shot_times * (args.target_shots // len(shot_times))
    replay_jobs += shot_times[: args.target_shots % len(shot_times)]
    replay_fcfs = schedule(replay_jobs, 2560)
    # This ordering needs advance knowledge of each shot's candidate count;
    # it is a scheduling sensitivity check, not the implemented host policy.
    replay_clairvoyant_lpt = schedule(sorted(replay_jobs, reverse=True), 2560)
    replay_lower_bound_seconds = max(sum(replay_jobs) / 2560, max(replay_jobs))
    power = json.loads(POWER.read_text(encoding="utf-8"))
    cpu = power["cpu_context"]
    dpu_power = next(row["power_w"] for row in power["rows"] if row["configuration"].startswith("dpu"))
    break_even_rate = cpu["throughput_shots_per_s"] * dpu_power / cpu["full_system_power_w"]
    calibration = []
    for omit, point in enumerate(cycle["points"]):
        remaining = [other for i, other in enumerate(cycle["points"]) if i != omit]
        loo_intercept, loo_slope = fit_cycles(remaining)
        mean_seconds = (trials * loo_intercept + statistics.mean(candidates) * loo_slope) / frequency
        calibration.append(
            {
                "omitted_trace": Path(point["path"]).parent.name,
                "omitted_candidate_count": point["candidates"],
                "traces_used": len(remaining),
                "intercept_cycles": loo_intercept,
                "cycles_per_candidate": loo_slope,
                "ideal_2560_dpu_rate_shots_per_second": 2560 / mean_seconds,
                "above_fixed_power_energy_break_even": (2560 / mean_seconds) > break_even_rate,
            }
        )

    trial_values = shot_meta.pop("trial_candidates")
    max_fit_candidate = max(point["candidates"] for point in cycle["points"])
    above_fit = [value for value in trial_values if value > max_fit_candidate]
    shot_meta.update(
        {
            "cycle_fit_trace_count": len(cycle["points"]),
            "cycle_fit_candidate_range": [
                min(point["candidates"] for point in cycle["points"]),
                max_fit_candidate,
            ],
            "trials_above_cycle_fit_max": len(above_fit),
            "fraction_trials_above_cycle_fit_max": len(above_fit) / len(trial_values),
            "fraction_candidates_from_trials_above_cycle_fit_max": sum(above_fit) / sum(trial_values),
            "mean_estimated_dpu_seconds_per_shot": statistics.mean(shot_times),
            "max_estimated_dpu_seconds_per_shot": max(shot_times),
        }
    )
    predictive = predictive_bootstrap(
        shot_times, 2560, args.target_shots, args.bootstrap, args.seed
    )
    result = {
        "status": "model_sensitivity_only_not_measured_dpu_utilization",
        "old_functional_simulator": {
            "workload": "same set of 64 distinct syndromes across settings; 3 wall-time runs per point",
            "replicates": repetitions,
            "extrapolation_sensitivity": old_fit,
            "invalid_inference": "Do not fit these process-wall observations jointly with paper cycle results or claim physical DPU occupancy.",
        },
        "paper_short_model": {
            "workload": "10,000 distinct accepted shots, 16 sequential trials per DPU-assigned shot",
            "shot_population": shot_meta,
            "scheduling_assumptions": "Each shot stays on one DPU; immediate free-DPU dispatch; no host/rank transfer, launch, fallback, or allocation delay.",
            "scaling": scaling,
            "replay_target_shots": {
                "workload": "10k observed shot times repeated to target_shots; not a new experiment",
                "target_shots": args.target_shots,
                "first_come_first_served": replay_fcfs,
                "clairvoyant_longest_first": replay_clairvoyant_lpt,
                "compute_only_lower_bound_seconds": replay_lower_bound_seconds,
                "clairvoyant_caveat": "Requires per-shot work to be known before DPU execution; not a measured or current scheduling policy.",
            },
            "predictive_bootstrap_target_shots": predictive,
            "calibration_leave_one_out": calibration,
            "fixed_power_energy_break_even_shots_per_second": break_even_rate,
            "cycle_fit_caveat": "Four traces only; 12% of trials exceed the largest calibrated candidate count. Leave-one-out is sensitivity, not a confidence interval.",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "functional_simulator_replicates.csv", repetitions)
    write_csv(args.output / "functional_simulator_extrapolation.csv", old_fit)
    write_csv(args.output / "paper_short_compute_schedule.csv", scaling)
    write_csv(args.output / "cycle_calibration_leave_one_out.csv", calibration)
    (args.output / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"audit written to {args.output}")
    print(f"old runs: {len(repetitions)} settings × 3 replicates; paper shots: {len(candidates)}")
    print(f"100k modeled utilization median: {predictive['utilization']['median']:.3%}")
    print(f"fixed-power energy break-even: {break_even_rate:.1f} shots/s")


if __name__ == "__main__":
    main()
