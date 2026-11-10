#!/usr/bin/env python3
"""Fit a non-negative Amdahl-style model to multi-DPU throughput samples."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def fit_nonnegative_line(xs: list[float], ys: list[float]) -> tuple[float, float]:
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("at least two distinct DPU counts are required")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean

    # Non-negative least squares for two coefficients. If the unconstrained
    # optimum crosses an axis, the optimum on that axis has a closed form.
    if intercept < 0 and slope < 0:
        return max(y_mean, 0.0), 0.0
    if intercept < 0:
        slope = max(sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs), 0.0)
        intercept = 0.0
    elif slope < 0:
        intercept = max(y_mean, 0.0)
        slope = 0.0
    return intercept, slope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-dpus", type=int, default=2560)
    parser.add_argument(
        "--cap-by-shots",
        action="store_true",
        help="Limit useful concurrent DPUs to the fixed number of independent shots",
    )
    parser.add_argument(
        "--cpu-baseline-summary",
        type=Path,
        help="Use an independently repeated default CPU baseline instead of paired CPU samples",
    )
    args = parser.parse_args()
    if args.target_dpus <= 0:
        parser.error("--target-dpus must be positive")

    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    with args.measurements.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["predictions_match_cpu"].lower() != "true":
                raise ValueError(
                    f"CPU/DPU prediction mismatch at CPU={row['cpu_threads']} "
                    f"DPU={row['dpu_count']}"
                )
            groups[int(row["cpu_threads"])].append(row)

    if not groups:
        raise ValueError("measurement CSV has no rows")

    independent_cpu_rates: dict[int, float] = {}
    if args.cpu_baseline_summary:
        with args.cpu_baseline_summary.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                if row["configuration"] != "default":
                    continue
                cpu_threads = int(row["cpu_threads"])
                rate = float(row["median_process_wall_throughput_per_second"])
                if cpu_threads in independent_cpu_rates or not math.isfinite(rate) or rate <= 0:
                    raise ValueError(f"duplicate or invalid CPU baseline for {cpu_threads} threads")
                independent_cpu_rates[cpu_threads] = rate

    fieldnames = [
        "measurement_backend",
        "cpu_threads",
        "target_dpus",
        "measured_dpu_counts",
        "fit_model",
        "serial_seconds_per_shot",
        "parallel_seconds_per_shot_at_one_dpu",
        "r_squared",
        "cpu_only_full_system_throughput_median_per_second",
        "predicted_hybrid_end_to_end_throughput_per_second",
        "predicted_full_system_speedup_vs_cpu_only",
    ]
    if args.cap_by_shots:
        fieldnames += [
            "fixed_shots",
            "effective_parallel_dpus",
            "maximum_instantaneous_fraction_of_target_dpus",
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for cpu_threads in sorted(groups):
            rows = sorted(groups[cpu_threads], key=lambda row: int(row["dpu_count"]))
            shot_counts = {int(row["shots"]) for row in rows}
            if len(shot_counts) != 1:
                raise ValueError(f"mixed shot counts for CPU={cpu_threads}: {sorted(shot_counts)}")
            shots = next(iter(shot_counts))
            backends = {row["backend"] for row in rows}
            if len(backends) != 1:
                raise ValueError(f"mixed backends for CPU={cpu_threads}: {sorted(backends)}")
            xs = [1.0 / int(row["dpu_count"]) for row in rows]
            ys = [float(row["process_wall_seconds"]) / int(row["shots"]) for row in rows]
            intercept, slope = fit_nonnegative_line(xs, ys)
            predictions = [intercept + slope * x for x in xs]
            residual = sum((actual - predicted) ** 2 for actual, predicted in zip(ys, predictions))
            total = sum((actual - statistics.fmean(ys)) ** 2 for actual in ys)
            r_squared = 1.0 - residual / total if total != 0 else 1.0

            effective_dpus = min(args.target_dpus, shots) if args.cap_by_shots else args.target_dpus
            predicted_seconds = intercept + slope / effective_dpus
            if predicted_seconds <= 0 or not math.isfinite(predicted_seconds):
                raise ValueError(f"invalid fitted service time for CPU={cpu_threads}")
            predicted_throughput = 1.0 / predicted_seconds
            cpu_rates = []
            for row in rows:
                raw_cpu_rate = (
                    row.get("cpu_only_full_system_throughput_per_second")
                    or row.get("cpu_only_throughput_per_second")
                )
                if not raw_cpu_rate:
                    continue  # Invalid paired CPU wall-time samples are left blank.
                value = float(raw_cpu_rate)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"invalid CPU rate for CPU={cpu_threads}")
                cpu_rates.append(value)
            if args.cpu_baseline_summary:
                if cpu_threads not in independent_cpu_rates:
                    raise ValueError(f"no independent CPU baseline for CPU={cpu_threads}")
                cpu_rate = independent_cpu_rates[cpu_threads]
            else:
                if not cpu_rates:
                    raise ValueError(f"no valid CPU wall-time samples for CPU={cpu_threads}")
                cpu_rate = statistics.median(cpu_rates)
            output_row = {
                    "measurement_backend": next(iter(backends)),
                    "cpu_threads": cpu_threads,
                    "target_dpus": args.target_dpus,
                    "measured_dpu_counts": ";".join(row["dpu_count"] for row in rows),
                    "fit_model": (
                        "seconds_per_shot=serial+parallel/min(target_dpus,shots)"
                        if args.cap_by_shots
                        else "seconds_per_shot=serial+parallel/dpus"
                    ),
                    "serial_seconds_per_shot": f"{intercept:.12g}",
                    "parallel_seconds_per_shot_at_one_dpu": f"{slope:.12g}",
                    "r_squared": f"{r_squared:.9f}",
                    "cpu_only_full_system_throughput_median_per_second": f"{cpu_rate:.9f}",
                    "predicted_hybrid_end_to_end_throughput_per_second": f"{predicted_throughput:.9f}",
                    "predicted_full_system_speedup_vs_cpu_only": f"{predicted_throughput / cpu_rate:.9f}",
                }
            if args.cap_by_shots:
                output_row.update(
                    {
                        "fixed_shots": shots,
                        "effective_parallel_dpus": effective_dpus,
                        "maximum_instantaneous_fraction_of_target_dpus": f"{effective_dpus / args.target_dpus:.9f}",
                    }
                )
            writer.writerow(output_row)

    print(f"wrote {args.target_dpus}-DPU extrapolation to {args.output}")


if __name__ == "__main__":
    main()
