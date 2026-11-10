#!/usr/bin/env python3
"""Select a reproducible, population-aware cycle-trace set from a full run.

Only ``resident_dpu`` rows advance the shot index.  A fallback may append a
second ``cpu_resume`` row for the same shot, so treating every CSV row as one
shot would silently corrupt the mapping back to the original b8 stream.

The stratified reservoir is suitable for weighted population estimates.
Extra top-tail and fallback samples are coverage cases and deliberately have
no sampling weight unless they were also selected by the reservoir.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Sample:
    global_shot_index: int
    chunk: str
    candidates: int
    expansions: int
    dpu_candidates: int
    dpu_expansions: int
    status: int
    fallback_reason: str
    heap_size: int
    arena_used: int
    export_bytes: int
    candidate_bin: str
    tags: set[str] = field(default_factory=set)


def stable_priority(seed: int, shot: int, namespace: str) -> int:
    payload = f"{seed}:{namespace}:{shot}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def candidate_bin(candidates: int) -> str:
    if candidates <= 0:
        return "0"
    power = candidates.bit_length() - 1
    return f"2^{power}..2^{power + 1}-1"


def parse_int(row: dict[str, str], field_name: str, default: int = 0) -> int:
    text = row.get(field_name, "")
    return default if text == "" else int(text)


def histogram_add(histogram: array, value: int) -> None:
    if value >= len(histogram):
        histogram.extend(array("Q", [0]) * (value + 1 - len(histogram)))
    histogram[value] += 1


def histogram_quantiles(histogram: array, count: int, total: int) -> dict[str, float | int]:
    targets = {
        "p50": count // 2,
        "p90": int(0.90 * (count - 1)),
        "p99": int(0.99 * (count - 1)),
    }
    result: dict[str, float | int] = {"mean": total / count}
    cumulative = 0
    unresolved = dict(targets)
    maximum = 0
    for value, frequency in enumerate(histogram):
        if frequency == 0:
            continue
        maximum = value
        cumulative += frequency
        for name, index in list(unresolved.items()):
            if cumulative > index:
                result[name] = value
                del unresolved[name]
    result["max"] = maximum
    return result


def bounded_sample(
    reservoirs: dict[str, list[tuple[int, int, Sample]]],
    namespace: str,
    limit: int,
    seed: int,
    sample: Sample,
) -> None:
    if limit <= 0:
        return
    priority = stable_priority(seed, sample.global_shot_index, namespace)
    heap = reservoirs[namespace]
    item = (-priority, -sample.global_shot_index, sample)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--per-bin", type=int, default=8)
    parser.add_argument("--top", type=int, default=32)
    parser.add_argument("--per-fallback-reason", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0x54455353)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    if min(args.per_bin, args.top, args.per_fallback_reason) < 0:
        parser.error("sample counts must be non-negative")
    if not args.allow_incomplete and not (args.run_root / "complete").is_file():
        raise SystemExit(
            f"{args.run_root}/complete is absent; use --allow-incomplete only for a provisional set"
        )

    bin_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    reservoirs: dict[str, list[tuple[int, int, Sample]]] = defaultdict(list)
    top_heap: list[tuple[int, int, Sample]] = []
    chunks_read = 0
    shots_read = 0
    cpu_shots = 0
    cpu_seconds_sum = 0.0
    cpu_candidate_sum = 0
    dpu_candidate_sum = 0
    resume_rows = 0
    resume_cpu_seconds_sum = 0.0
    export_bytes_sum = 0
    export_bytes_max = 0
    cpu_logical_errors = 0
    oracle_logical_errors = 0
    prediction_hash_chunks = 0
    prediction_mismatches = 0
    quantization_bound_violations = 0
    nonzero_solution_cost_deltas = 0
    max_absolute_solution_cost_delta = 0.0
    max_solution_cost_delta_over_q16_bound = 0.0
    candidate_histogram = array("Q")
    population_max_candidates = -1
    population_max_shot = -1

    for chunk_dir in sorted((args.run_root / "chunks").glob("[0-9]*-[0-9]*")):
        if not (chunk_dir / "complete").is_file():
            continue
        try:
            begin_text, end_text = chunk_dir.name.split("-", 1)
            begin, end = int(begin_text), int(end_text)
        except ValueError as exc:
            raise SystemExit(f"invalid chunk directory name: {chunk_dir.name}") from exc
        expected = end - begin
        cpu_candidates = array("Q")
        cpu_expansions = array("Q")
        with (chunk_dir / "cpu" / "search_stats.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                candidates = parse_int(row, "candidates")
                cpu_candidates.append(candidates)
                cpu_expansions.append(parse_int(row, "expansions"))
                cpu_seconds_sum += float(row["cpu_seconds"])
                cpu_candidate_sum += candidates
                histogram_add(candidate_histogram, candidates)
        if len(cpu_candidates) != expected:
            raise SystemExit(
                f"{chunk_dir.name}: expected {expected} CPU rows, got {len(cpu_candidates)}"
            )
        cpu_shots += len(cpu_candidates)

        with (chunk_dir / "cpu" / "stats.json").open() as f:
            cpu_stats = json.load(f)
        with (chunk_dir / "oracle" / "stats.json").open() as f:
            oracle_stats = json.load(f)
        comparison_path = chunk_dir / "comparison.json"
        if not comparison_path.is_file():
            raise SystemExit(f"{chunk_dir.name}: comparison.json is absent")
        comparison = json.loads(comparison_path.read_text())
        cpu_logical_errors += int(cpu_stats["num_errors"])
        oracle_logical_errors += int(oracle_stats["num_errors"])
        prediction_mismatches += int(comparison["prediction_mismatches"])
        quantization_bound_violations += int(comparison["quantization_bound_violations"])
        nonzero_solution_cost_deltas += int(comparison["nonzero_solution_cost_deltas"])
        max_absolute_solution_cost_delta = max(
            max_absolute_solution_cost_delta,
            float(comparison["max_absolute_solution_cost_delta"]),
        )
        max_solution_cost_delta_over_q16_bound = max(
            max_solution_cost_delta_over_q16_bound,
            float(comparison["max_solution_cost_delta_over_q16_bound"]),
        )
        if int(comparison["quantization_bound_violations"]) != 0:
            raise SystemExit(f"{chunk_dir.name}: solution cost exceeds Q16 error bound")
        hashes = [line.split()[0] for line in
                  (chunk_dir / "prediction_sha256.txt").read_text().splitlines() if line.strip()]
        if len(hashes) != 2:
            raise SystemExit(f"{chunk_dir.name}: prediction hashes are absent")
        prediction_hash_chunks += int(hashes[0] == hashes[1])

        stats_path = chunk_dir / "oracle" / "search_stats.csv"
        resident_index = 0
        with stats_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("kind") != "resident_dpu":
                    resume_rows += 1
                    resume_cpu_seconds_sum += float(row["cpu_seconds"])
                    continue
                shot = begin + resident_index
                candidates = int(cpu_candidates[resident_index])
                expansions = int(cpu_expansions[resident_index])
                resident_index += 1
                dpu_candidates = parse_int(row, "candidates")
                dpu_candidate_sum += dpu_candidates
                status = parse_int(row, "status", -1)
                reason = row.get("fallback_reason", "none") or "none"
                bin_name = candidate_bin(candidates)
                export_bytes = parse_int(row, "export_bytes")
                export_bytes_sum += export_bytes
                export_bytes_max = max(export_bytes_max, export_bytes)
                sample = Sample(
                    global_shot_index=shot,
                    chunk=chunk_dir.name,
                    candidates=candidates,
                    expansions=expansions,
                    dpu_candidates=dpu_candidates,
                    dpu_expansions=parse_int(row, "expansions"),
                    status=status,
                    fallback_reason=reason,
                    heap_size=parse_int(row, "heap_size"),
                    arena_used=parse_int(row, "arena_used"),
                    export_bytes=export_bytes,
                    candidate_bin=bin_name,
                )
                if candidates > population_max_candidates:
                    population_max_candidates = candidates
                    population_max_shot = shot
                bin_counts[bin_name] += 1
                status_counts[str(status)] += 1
                if status == 2 or reason != "none":
                    fallback_counts[reason] += 1
                    bounded_sample(
                        reservoirs,
                        f"fallback:{reason}",
                        args.per_fallback_reason,
                        args.seed,
                        sample,
                    )
                bounded_sample(
                    reservoirs, f"bin:{bin_name}", args.per_bin, args.seed, sample
                )
                if args.top > 0:
                    top_item = (candidates, -shot, sample)
                    if len(top_heap) < args.top:
                        heapq.heappush(top_heap, top_item)
                    elif top_item > top_heap[0]:
                        heapq.heapreplace(top_heap, top_item)

        if resident_index != expected:
            raise SystemExit(
                f"{chunk_dir.name}: expected {expected} resident rows, got {resident_index}"
            )
        chunks_read += 1
        shots_read += resident_index

    if shots_read == 0:
        raise SystemExit("no complete chunks with resident_dpu rows found")

    selected: dict[int, Sample] = {}
    bin_reservoir_sizes: Counter[str] = Counter()
    for namespace, heap in reservoirs.items():
        tag = "stratified" if namespace.startswith("bin:") else namespace
        for _, _, sample in heap:
            chosen = selected.setdefault(sample.global_shot_index, sample)
            chosen.tags.add(tag)
            if tag == "stratified":
                bin_reservoir_sizes[sample.candidate_bin] += 1
    for _, _, sample in top_heap:
        chosen = selected.setdefault(sample.global_shot_index, sample)
        chosen.tags.add("top_tail")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "selected_traces.csv"
    fieldnames = [
        "global_shot_index",
        "chunk",
        "candidates",
        "expansions",
        "dpu_candidates",
        "dpu_expansions",
        "status",
        "fallback_reason",
        "heap_size",
        "arena_used",
        "export_bytes",
        "candidate_bin",
        "selection_tags",
        "population_bin_count",
        "stratified_bin_sample_count",
        "sampling_weight",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in sorted(selected.values(), key=lambda x: x.global_shot_index):
            stratified = "stratified" in sample.tags
            sample_count = bin_reservoir_sizes[sample.candidate_bin]
            writer.writerow(
                {
                    **{
                        name: getattr(sample, name)
                        for name in fieldnames[:12]
                    },
                    "selection_tags": ";".join(sorted(sample.tags)),
                    "population_bin_count": bin_counts[sample.candidate_bin],
                    "stratified_bin_sample_count": sample_count,
                    "sampling_weight": (
                        f"{bin_counts[sample.candidate_bin] / sample_count:.12g}"
                        if stratified
                        else ""
                    ),
                }
            )

    summary = {
        "run_root": str(args.run_root.resolve()),
        "complete_run_required": not args.allow_incomplete,
        "chunks_read": chunks_read,
        "shots_read": shots_read,
        "selected_unique_shots": len(selected),
        "seed": args.seed,
        "selection": {
            "stratified_per_log2_candidate_bin": args.per_bin,
            "largest_candidate_shots": args.top,
            "per_fallback_reason": args.per_fallback_reason,
        },
        "candidate_bins": dict(sorted(bin_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "fallback_reason_counts": dict(sorted(fallback_counts.items())),
        "max_candidates": population_max_candidates,
        "max_candidates_shot": population_max_shot,
        "cpu": {
            "shots": cpu_shots,
            "cpu_seconds_sum": cpu_seconds_sum,
            "mean_cpu_seconds_per_shot": cpu_seconds_sum / cpu_shots,
            "candidate_sum": cpu_candidate_sum,
            "candidate_quantiles": histogram_quantiles(
                candidate_histogram, cpu_shots, cpu_candidate_sum
            ),
            "logical_errors": cpu_logical_errors,
        },
        "resident": {
            "dpu_attempts": shots_read,
            "dpu_candidate_sum": dpu_candidate_sum,
            "mean_dpu_candidates_per_shot": dpu_candidate_sum / shots_read,
            "resume_rows": resume_rows,
            "resume_cpu_seconds_sum": resume_cpu_seconds_sum,
            "export_bytes_sum": export_bytes_sum,
            "export_bytes_max": export_bytes_max,
            "logical_errors": oracle_logical_errors,
        },
        "predictions": {
            "hash_equal_chunks": prediction_hash_chunks,
            "prediction_mismatches": prediction_mismatches,
            "prediction_mismatch_rate": prediction_mismatches / shots_read,
            "quantization_bound_violations": quantization_bound_violations,
            "all_solution_cost_deltas_within_q16_bound": quantization_bound_violations == 0,
            "nonzero_solution_cost_deltas": nonzero_solution_cost_deltas,
            "max_absolute_solution_cost_delta": max_absolute_solution_cost_delta,
            "max_solution_cost_delta_over_q16_bound": max_solution_cost_delta_over_q16_bound,
        },
    }
    with (args.output_dir / "population_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
