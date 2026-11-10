#!/usr/bin/env python3
"""Build a Tesseract-ready DEM and postselected syndromes from an MSC circuit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import BinaryIO

import numpy as np
import stim


_BASIS_BY_ANNOTATION = "XXXZZZXZ"
_COLOR_BY_ANNOTATION = "rgbrgb__"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xor_detector_set(instruction: stim.DemInstruction) -> frozenset[int]:
    result: set[int] = set()
    for target in instruction.targets_copy():
        if target.is_relative_detector_id():
            if target.val in result:
                result.remove(target.val)
            else:
                result.add(target.val)
    return frozenset(result)


def _observable_mask(instruction: stim.DemInstruction) -> int:
    result = 0
    for target in instruction.targets_copy():
        if target.is_logical_observable_id():
            result ^= 1 << target.val
    return result


def find_postselected_detectors(
    dem: stim.DetectorErrorModel,
) -> tuple[frozenset[int], frozenset[int], frozenset[int]]:
    """Reproduce the postselection rules in cultiv's desaturation sampler."""

    flat_dem = dem.flattened()
    coordinates = flat_dem.get_detector_coordinates()
    bases: list[str] = []
    hidden: set[int] = set()
    visible: set[int] = set()

    for detector_id in range(flat_dem.num_detectors):
        coords = coordinates.get(detector_id, [])
        if len(coords) <= 4 or coords[4] == -9:
            hidden.add(detector_id)
            bases.append("!")
            continue

        annotation = int(coords[4])
        if not 0 <= annotation < len(_BASIS_BY_ANNOTATION):
            raise ValueError(
                f"detector {detector_id} has unsupported coordinate annotation "
                f"{annotation}"
            )
        basis = _BASIS_BY_ANNOTATION[annotation]
        color = _COLOR_BY_ANNOTATION[annotation]
        bases.append(basis)
        if (color == "r" and basis == "X") or (
            color == "g" and basis == "Z"
        ):
            visible.add(detector_id)

    for instruction in flat_dem:
        if instruction.type != "error":
            continue
        detectors = _xor_detector_set(instruction)
        detector_bases = {bases[d] for d in detectors}
        if (
            len(detectors) == 2
            and len(detector_bases) == 1
            and _observable_mask(instruction)
        ):
            hidden.update(detectors)

    return frozenset(hidden | visible), frozenset(hidden), frozenset(visible)


def _write_samples(output: BinaryIO, samples: np.ndarray, output_format: str) -> None:
    """Append a dense sample chunk in a Stim-compatible shot format."""
    if samples.shape[0] == 0:
        return
    if output_format == "b8":
        output.write(np.packbits(samples, axis=1, bitorder="little").tobytes())
        return
    encoded = np.empty((samples.shape[0], samples.shape[1] + 1), dtype=np.uint8)
    encoded[:, :-1] = samples
    encoded[:, :-1] += ord("0")
    encoded[:, -1] = ord("\n")
    output.write(encoded.tobytes())


def prepare(args: argparse.Namespace) -> dict[str, object]:
    circuit_path = args.circuit.resolve()
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        source_circuit = circuit_path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        source_circuit = str(circuit_path)

    circuit = stim.Circuit.from_file(str(circuit_path))
    dem = circuit.detector_error_model(
        decompose_errors=False,
        flatten_loops=False,
        allow_gauge_detectors=True,
        approximate_disjoint_errors=1,
        ignore_decomposition_failures=False,
        block_decomposition_from_introducing_remnant_edges=False,
    )
    postselected, hidden, visible = find_postselected_detectors(dem)

    discard_mask = np.zeros(circuit.num_detectors, dtype=np.bool_)
    discard_mask[list(postselected)] = True

    dem_path = output_dir / "model.dem"
    syndrome_path = output_dir / f"syndromes.{args.output_format}"
    observable_path = output_dir / f"observables.{args.output_format}"
    metadata_path = output_dir / "metadata.json"
    dem.to_file(str(dem_path))

    sampler = circuit.compile_detector_sampler(seed=args.seed)
    keep_limit = args.target_kept if args.target_kept is not None else args.max_kept
    raw_shots_sampled = 0
    surviving_shots = 0
    kept_shots = 0
    accepted_index_sample: list[int] = []
    weight_sample: list[int] = []
    weight_histogram: Counter[int] = Counter()
    recommended = 0
    recommended_weight: int | None = None
    chunks_sampled = 0

    with syndrome_path.open("wb") as syndrome_output, observable_path.open(
        "wb"
    ) as observable_output:
        while (
            kept_shots < keep_limit
            if args.target_kept is not None
            else raw_shots_sampled < args.raw_shots
        ):
            if args.target_kept is None:
                remaining_raw = args.raw_shots - raw_shots_sampled
                if remaining_raw <= 0:
                    break
                chunk_shots = min(args.sample_chunk_shots, remaining_raw)
            else:
                if args.max_raw_shots and raw_shots_sampled >= args.max_raw_shots:
                    break
                chunk_shots = args.sample_chunk_shots
                if args.max_raw_shots:
                    chunk_shots = min(
                        chunk_shots, args.max_raw_shots - raw_shots_sampled
                    )

            detectors, observables = sampler.sample(
                shots=chunk_shots,
                separate_observables=True,
                bit_packed=False,
            )
            if circuit.num_observables == 0:
                observables = np.zeros((chunk_shots, 0), dtype=np.bool_)

            keep = ~np.any(detectors & discard_mask, axis=1)
            chunk_accepted = np.flatnonzero(keep)
            surviving_shots += int(chunk_accepted.size)
            remaining_kept = max(keep_limit - kept_shots, 0)
            selected = chunk_accepted[:remaining_kept]
            kept_detectors = detectors[selected]
            kept_observables = observables[selected]
            weights = np.count_nonzero(kept_detectors, axis=1)

            _write_samples(syndrome_output, kept_detectors, args.output_format)
            _write_samples(observable_output, kept_observables, args.output_format)

            sample_remaining = max(
                args.metadata_sample_limit - len(accepted_index_sample), 0
            )
            if sample_remaining:
                accepted_index_sample.extend(
                    int(raw_shots_sampled + index)
                    for index in selected[:sample_remaining]
                )
                weight_sample.extend(int(weight) for weight in weights[:sample_remaining])
            weight_histogram.update(int(weight) for weight in weights)

            nonzero = np.flatnonzero(weights)
            if nonzero.size:
                local = int(nonzero[np.argmin(weights[nonzero])])
                local_weight = int(weights[local])
                if recommended_weight is None or local_weight < recommended_weight:
                    recommended = kept_shots + local
                    recommended_weight = local_weight

            kept_shots += int(selected.size)
            raw_shots_sampled += chunk_shots
            chunks_sampled += 1
            if (
                args.progress_every_chunks
                and chunks_sampled % args.progress_every_chunks == 0
            ):
                print(
                    "progress",
                    f"chunks={chunks_sampled}",
                    f"raw_shots={raw_shots_sampled}",
                    f"kept_shots={kept_shots}",
                    flush=True,
                )

    if kept_shots == 0:
        raise RuntimeError("no shots survived MSC postselection")
    if args.target_kept is not None and kept_shots != args.target_kept:
        raise RuntimeError(
            f"only {kept_shots} shots survived after sampling "
            f"{raw_shots_sampled} raw shots; target was {args.target_kept}"
        )

    flat_dem = dem.flattened()
    metadata: dict[str, object] = {
        "format": "tesseract-msc-workload-v2",
        "shot_data_format": args.output_format,
        "source_circuit": source_circuit,
        "source_circuit_sha256": _sha256(circuit_path),
        "dem_sha256": _sha256(dem_path),
        "dem_options": {
            "decompose_errors": False,
            "flatten_loops": False,
            "allow_gauge_detectors": True,
            "approximate_disjoint_errors": 1,
            "ignore_decomposition_failures": False,
            "block_decomposition_from_introducing_remnant_edges": False,
        },
        "seed": args.seed,
        "raw_shots": raw_shots_sampled,
        "requested_raw_shots": (
            args.raw_shots if args.target_kept is None else None
        ),
        "target_kept_shots": args.target_kept,
        "sample_chunk_shots": args.sample_chunk_shots,
        "surviving_shots_before_limit": surviving_shots,
        "kept_shots": kept_shots,
        "accepted_raw_shot_indices": accepted_index_sample,
        "accepted_raw_shot_indices_truncated": kept_shots
        > len(accepted_index_sample),
        "syndrome_weights": weight_sample,
        "syndrome_weights_truncated": kept_shots > len(weight_sample),
        "syndrome_weight_histogram": {
            str(weight): count for weight, count in sorted(weight_histogram.items())
        },
        "recommended_trace_shot_index": recommended,
        "num_qubits": circuit.num_qubits,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "num_dem_errors": flat_dem.num_errors,
        "postselected_detector_count": len(postselected),
        "postselected_hidden_detector_count": len(hidden),
        "postselected_visible_detector_count": len(visible),
        "postselected_detectors": sorted(postselected),
        "files": {
            "dem": dem_path.name,
            "syndromes": syndrome_path.name,
            "observables": observable_path.name,
        },
        "file_sha256": {
            "dem": _sha256(dem_path),
            "syndromes": _sha256(syndrome_path),
            "observables": _sha256(observable_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--raw-shots", type=int, default=4096)
    parser.add_argument("--max-kept", type=int, default=64)
    parser.add_argument(
        "--target-kept",
        type=int,
        help="sample until exactly this many postselection survivors are written",
    )
    parser.add_argument(
        "--max-raw-shots",
        type=int,
        default=0,
        help="safety cap for --target-kept mode (0 means no cap)",
    )
    parser.add_argument("--sample-chunk-shots", type=int, default=100_000)
    parser.add_argument("--metadata-sample-limit", type=int, default=100_000)
    parser.add_argument(
        "--progress-every-chunks",
        type=int,
        default=0,
        help="print progress after every N completed chunks (0 disables progress)",
    )
    parser.add_argument("--output-format", choices=("01", "b8"), default="01")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.raw_shots <= 0 or args.max_kept <= 0:
        parser.error("--raw-shots and --max-kept must be positive")
    if args.target_kept is not None and args.target_kept <= 0:
        parser.error("--target-kept must be positive")
    if args.max_raw_shots < 0:
        parser.error("--max-raw-shots must be non-negative")
    if args.sample_chunk_shots <= 0:
        parser.error("--sample-chunk-shots must be positive")
    if args.metadata_sample_limit < 0:
        parser.error("--metadata-sample-limit must be non-negative")
    if args.progress_every_chunks < 0:
        parser.error("--progress-every-chunks must be non-negative")

    metadata = prepare(args)
    print(
        "prepared",
        metadata["kept_shots"],
        "postselected MSC shots; recommended trace shot",
        metadata["recommended_trace_shot_index"],
    )


if __name__ == "__main__":
    main()
