#!/usr/bin/env python3
"""Summarize per-launch candidate counts emitted by Tesseract live-DPU runs."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


BATCH_SIZES_RE = re.compile(r"(?:^|\s)batch_sizes=([0-9,]*)")


def percentile(values: list[int], fraction: float) -> int:
    index = max(0, math.ceil(fraction * len(values)) - 1)
    return values[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="dpu_live.log from run_live_dem_comparison.sh")
    parser.add_argument("--output", type=Path, help="optional CSV output path")
    args = parser.parse_args()

    sizes: list[int] = []
    for line in args.log.read_text(encoding="utf-8").splitlines():
        match = BATCH_SIZES_RE.search(line)
        if match and match.group(1):
            sizes.extend(int(value) for value in match.group(1).split(","))
    if not sizes:
        raise SystemExit(f"no batch_sizes fields found in {args.log}")

    sizes.sort()
    lines = ["metric,value"]
    lines.append(f"launches,{len(sizes)}")
    lines.append(f"candidates,{sum(sizes)}")
    lines.append(f"mean,{sum(sizes) / len(sizes):.6f}")
    lines.append(f"min,{sizes[0]}")
    for label, fraction in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99)):
        lines.append(f"{label},{percentile(sizes, fraction)}")
    lines.append(f"max,{sizes[-1]}")
    lines.append(f"launches_lt_16,{sum(size < 16 for size in sizes)}")
    for limit in (8, 16, 32, 64, 128):
        count = sum(size <= limit for size in sizes)
        lines.append(f"launches_le_{limit},{count}")
    lines.append(f"launches_gt_128,{sum(size > 128 for size in sizes)}")
    output = "\n".join(lines) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
