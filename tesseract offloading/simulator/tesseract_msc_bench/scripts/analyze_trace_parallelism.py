#!/usr/bin/env python3
"""Report DPU-parallelism opportunities in a Tesseract candidate trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def greedy_candidate_colors(detectors: list[int], conflicts: list[set[int]]) -> int:
    colors: dict[int, int] = {}
    for detector in sorted(detectors, key=lambda item: len(conflicts[item]), reverse=True):
        used = {colors[other] for other in conflicts[detector] if other in colors}
        color = 0
        while color in used:
            color += 1
        colors[detector] = color
    return max(colors.values(), default=-1) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()

    with args.trace.open(encoding="utf-8") as source:
        trace = json.load(source)

    num_detectors = int(trace["num_detectors"])
    e2d_offsets = trace["error_detector_offsets"]
    e2d_ids = trace["error_detector_ids"]
    candidate_offsets = trace["candidate_detector_offsets"]
    candidate_ids = trace["candidate_detector_ids"]
    num_errors = len(trace["error_cost_q"])

    d2e: list[list[int]] = [[] for _ in range(num_detectors)]
    error_degrees: list[int] = []
    conflicts: list[set[int]] = [set() for _ in range(num_detectors)]
    for error in range(num_errors):
        detectors = e2d_ids[e2d_offsets[error] : e2d_offsets[error + 1]]
        error_degrees.append(len(detectors))
        for detector in detectors:
            d2e[detector].append(error)
            conflicts[detector].update(other for other in detectors if other != detector)

    weights: list[int] = []
    repeated_checks: list[int] = []
    unique_checks: list[int] = []
    color_counts: list[int] = []
    max_error_contention: list[int] = []
    for candidate in range(len(candidate_offsets) - 1):
        detectors = candidate_ids[candidate_offsets[candidate] : candidate_offsets[candidate + 1]]
        weights.append(len(detectors))
        repeated_checks.append(
            sum(error_degrees[error] for detector in detectors for error in d2e[detector])
        )
        incident_errors = {error for detector in detectors for error in d2e[detector]}
        unique_checks.append(sum(error_degrees[error] for error in incident_errors))
        color_counts.append(greedy_candidate_colors(detectors, conflicts))
        residual = set(detectors)
        max_error_contention.append(
            max(
                (sum(detector in residual for detector in e2d_ids[e2d_offsets[error] : e2d_offsets[error + 1]])
                 for error in incident_errors),
                default=0,
            )
        )

    blocked_counts: list[int] = []
    blocked_words = trace.get("candidate_blocked_error_words", [])
    if blocked_words:
        stride = math.ceil(num_errors / 64)
        for candidate in range(len(candidate_offsets) - 1):
            begin = candidate * stride
            blocked_counts.append(sum(int(word).bit_count() for word in blocked_words[begin : begin + stride]))

    def distribution(name: str, values: list[int]) -> None:
        print(
            f"{name}: min={min(values, default=0)} p50={percentile(values, 0.50)} "
            f"p90={percentile(values, 0.90)} p99={percentile(values, 0.99)} "
            f"max={max(values, default=0)}"
        )

    print(f"trace={args.trace}")
    print(f"candidates={len(weights)} detectors={num_detectors} errors={num_errors}")
    distribution("candidate_detectors", weights)
    distribution("repeated_e2d_membership_checks", repeated_checks)
    distribution("unique_error_e2d_membership_checks", unique_checks)
    repeated_total = sum(repeated_checks)
    unique_total = sum(unique_checks)
    print(
        "overlap_reuse_ratio="
        f"{(repeated_total / unique_total if unique_total else 1):.6f} "
        f"repeated_total={repeated_total} unique_total={unique_total}"
    )
    distribution("greedy_conflict_colors", color_counts)
    distribution("max_residual_detectors_sharing_one_error", max_error_contention)
    if blocked_counts:
        distribution("blocked_errors", blocked_counts)


if __name__ == "__main__":
    main()
