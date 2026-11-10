#!/usr/bin/env python3
"""Repeat a paired Tesseract 01 syndrome/observable workload deterministically."""

from __future__ import annotations

import argparse
from pathlib import Path


def read_records(path: Path) -> list[str]:
    records = path.read_text(encoding="utf-8").splitlines()
    if not records:
        raise ValueError(f"workload is empty: {path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--syndromes", type=Path, required=True)
    parser.add_argument("--observables", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shots", type=int, required=True)
    args = parser.parse_args()
    if args.shots <= 0:
        raise ValueError("--shots must be positive")

    syndromes = read_records(args.syndromes)
    observables = read_records(args.observables)
    if len(syndromes) != len(observables):
        raise ValueError(
            f"syndrome/observable counts differ: {len(syndromes)} != {len(observables)}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    syndrome_out = args.out_dir / "syndromes.01"
    observable_out = args.out_dir / "observables.01"
    syndrome_out.write_text(
        "\n".join(syndromes[index % len(syndromes)] for index in range(args.shots)) + "\n",
        encoding="utf-8",
    )
    observable_out.write_text(
        "\n".join(observables[index % len(observables)] for index in range(args.shots)) + "\n",
        encoding="utf-8",
    )
    print(f"repeated {len(syndromes)} paired records to {args.shots} shots in {args.out_dir}")


if __name__ == "__main__":
    main()
