#!/usr/bin/env python3
"""Record a running Linux process's elapsed lifetime using /proc uptime.

Intended as an independent cross-check when a wall-clock-based benchmark
timer may have jumped. This is not a replacement for a fresh benchmark run.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def process_start_uptime(pid: int) -> float:
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    fields_after_comm = stat.rsplit(") ", 1)[1].split()
    start_ticks = int(fields_after_comm[19])  # Linux stat field 22.
    return start_ticks / os.sysconf("SC_CLK_TCK")


def uptime() -> float:
    return float(Path("/proc/uptime").read_text(encoding="ascii").split()[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    process = Path(f"/proc/{args.pid}")
    start = process_start_uptime(args.pid)
    while process.exists():
        time.sleep(0.25)
    elapsed = uptime() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"{elapsed:.2f}\n", encoding="ascii")
    print(f"pid={args.pid} uptime_elapsed_seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
