#!/usr/bin/env python3
"""Estimate CPU/DPU energy efficiency from a matched throughput model.

This script deliberately keeps measured/specification/model values separate.  It
does not pretend that uPIMulator measures power; DPU power is an externally
sourced parameter applied to the modeled throughput.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parent
DEFAULT_TIMING_MODEL = (
    BENCH_ROOT
    / "results"
    / "paper_dpu_validation"
    / "timing_model_paper.json"
)
DEFAULT_OUTPUT_DIR = BENCH_ROOT / "results" / "power" / "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-model", type=Path, default=DEFAULT_TIMING_MODEL)
    parser.add_argument("--configuration", default="short", help="paper model configuration")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cpu-power-w",
        type=float,
        default=65.0,
        help="CPU package power used for the baseline (default: i9-10900 TDP)",
    )
    parser.add_argument(
        "--cpu-power-kind",
        default="Intel specification TDP; not workload power measurement",
    )
    parser.add_argument(
        "--cpu-power-summary",
        type=Path,
        action="append",
        default=[],
        help=(
            "RAPL power_summary.json from run_cpu_power_baseline.ps1; may be "
            "passed repeatedly and is aggregated by the median"
        ),
    )
    parser.add_argument(
        "--cpu-power-domain",
        choices=("package", "package-plus-dram"),
        default="package-plus-dram",
    )
    parser.add_argument(
        "--cpu-power-transfer-note",
        help=(
            "Required when applying RAPL samples from a different decoder setting "
            "to a paper throughput configuration; printed in every result"
        ),
    )
    parser.add_argument(
        "--cpu-system-overhead-w",
        type=float,
        default=115.0,
        help=(
            "Chassis, fans, common modules and PSU-loss "
            "allowance used to align CPU with the full-server DPU boundary"
        ),
    )
    parser.add_argument(
        "--dpu-dimm-power-w",
        type=float,
        default=23.22,
        help="Power per 128-DPU PIM DIMM",
    )
    parser.add_argument("--dpu-dimms", type=int, default=20)
    parser.add_argument(
        "--dpu-server-power-w",
        type=float,
        default=770.0,
        help="Reference full-server power estimate including 20 PIM DIMMs",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=10_000,
        help="Work quantity used for total-time/energy projections",
    )
    return parser.parse_args()


def make_row(
    *,
    name: str,
    boundary: str,
    evidence_kind: str,
    power_w: float,
    throughput: float,
    shots: int,
) -> dict[str, Any]:
    energy_j_per_shot = power_w / throughput
    seconds_for_shots = shots / throughput
    return {
        "configuration": name,
        "power_boundary": boundary,
        "power_evidence": evidence_kind,
        "power_w": power_w,
        "throughput_shots_per_s": throughput,
        "energy_j_per_shot": energy_j_per_shot,
        "energy_mj_per_shot": energy_j_per_shot * 1000.0,
        "shots_per_joule": 1.0 / energy_j_per_shot,
        "projected_shots": shots,
        "projected_seconds": seconds_for_shots,
        "projected_energy_j": power_w * seconds_for_shots,
        "projected_energy_wh": power_w * seconds_for_shots / 3600.0,
    }


def main() -> None:
    args = parse_args()
    if args.cpu_power_w <= 0 or args.dpu_dimm_power_w <= 0:
        raise SystemExit("power values must be positive")
    if args.cpu_system_overhead_w < 0:
        raise SystemExit("CPU system overhead must be nonnegative")
    if args.dpu_dimms <= 0 or args.shots <= 0:
        raise SystemExit("DPU DIMM count and shots must be positive")

    with args.timing_model.open("r", encoding="utf-8") as f:
        timing = json.load(f)

    paper_model = "configurations" in timing
    if paper_model:
        try:
            config = next(
                item for item in timing["configurations"]
                if item["name"] == args.configuration
            )
        except StopIteration as exc:
            raise SystemExit(f"paper configuration not found: {args.configuration}") from exc
        cpu_rate = float(config["cpu"]["estimated_16_thread_rate"])
        primary_rate = float(config["dpu_model"]["system_rate_shots_per_second"])
        cpu_workers = timing["method"]["cpu_workers"]
        dpus = timing["method"]["dpus"]
        dpu_mhz = timing["method"]["dpu_mhz"]
        throughput_result_kind = timing["method"]["dpu_result"]
        validation_shots = int(config["validation"]["shots"])
        cpu_rate_basis = "1-thread measured time scaled to 16 threads; not measured at 16 threads"
        dpu_rate_basis = "uPIMulator cycle and SDK host-cost model; not physical DPU throughput"
    else:
        config = None
        cpu_rate = float(timing["inputs"]["cpu_only_full_system_rate_per_s"])
        primary_rate = float(timing["throughput_primary"]["system_per_s"])
        cpu_workers = timing["inputs"]["cpu_workers"]
        dpus = timing["inputs"]["dpus"]
        dpu_mhz = timing["inputs"]["dpu_mhz"]
        throughput_result_kind = timing["inputs"]["result_kind"]
        validation_shots = None
        cpu_rate_basis = "resident timing model or matching RAPL run throughput"
        dpu_rate_basis = "uPIMulator/SDK resident throughput model"
    if cpu_workers != 16 or dpus != 2560:
        raise SystemExit("this report requires 16 CPU workers and 2560 DPUs")
    if paper_model and args.cpu_power_summary and not args.cpu_power_transfer_note:
        raise SystemExit("paper model with RAPL summaries requires --cpu-power-transfer-note")
    cpu_power_w = args.cpu_power_w
    cpu_power_kind = args.cpu_power_kind
    cpu_boundary = "CPU package only"
    cpu_power_label = "CPU assumed component power"
    cpu_power_summaries: list[dict[str, Any]] = []
    if args.cpu_power_summary:
        for path in args.cpu_power_summary:
            with path.open("r", encoding="utf-8-sig") as f:
                cpu_power_summaries.append(json.load(f))
        if args.cpu_power_domain == "package-plus-dram":
            power_key = "average_package_and_dram_power_w"
            cpu_boundary = "RAPL CPU package + DRAM"
        else:
            power_key = "average_package_power_w"
            cpu_boundary = "RAPL CPU package only"
        cpu_power_w = statistics.median(
            float(summary[power_key]) for summary in cpu_power_summaries
        )
        if not paper_model:
            cpu_rate = statistics.median(
                float(summary["process_wall_throughput_shots_per_s"])
                for summary in cpu_power_summaries
            )
        cpu_power_kind = (
            f"median of {len(cpu_power_summaries)} simultaneous Windows RAPL run(s)"
        )
        cpu_power_label = "CPU measured component power"
        if args.cpu_power_transfer_note:
            cpu_power_kind += f"; {args.cpu_power_transfer_note}"
    cpu_component_power_w = cpu_power_w
    cpu_system_power_w = cpu_component_power_w + args.cpu_system_overhead_w
    cpu_system_power_kind = (
        f"{cpu_power_kind} + {args.cpu_system_overhead_w:.2f} W literature "
        "system-overhead allowance"
    )
    dpu_array_power_w = args.dpu_dimm_power_w * args.dpu_dimms
    rows = [
        make_row(
            name="cpu16_i9_10900",
            boundary=(
                f"complete CPU system model: {cpu_boundary} + chassis/fans/"
                "common modules/PSU loss"
            ),
            evidence_kind=cpu_system_power_kind,
            power_w=cpu_system_power_w,
            throughput=cpu_rate,
            shots=args.shots,
        ),
        make_row(
            name="dpu2560_full_offload_system",
            boundary=(
                "complete reference server: host CPUs, host DRAM, transfers/launch, "
                "20 PIM DIMMs, chassis, fans, and PSU loss"
            ),
            evidence_kind="literature component-power estimate",
            power_w=args.dpu_server_power_w,
            throughput=primary_rate,
            shots=args.shots,
        )
    ]
    cpu_context = {
        "component_power_w": cpu_component_power_w,
        "system_overhead_w": args.cpu_system_overhead_w,
        "full_system_power_w": cpu_system_power_w,
        "throughput_shots_per_s": cpu_rate,
        "full_system_energy_j_per_shot": cpu_system_power_w / cpu_rate,
        "component_power_boundary": cpu_boundary,
        "power_evidence": cpu_system_power_kind,
        "comparable_to_full_dpu_system": True,
        "qualification": (
            "aligned component-estimate boundary, not CPU wall-power measurement; "
            + (args.cpu_power_transfer_note or "CPU power is not paired with paper throughput")
            if paper_model else
            "aligned component-estimate boundary, not CPU wall-power measurement"
        ),
    }
    cpu_energy_j_per_shot = rows[0]["energy_j_per_shot"]
    dpu_energy_j_per_shot = rows[1]["energy_j_per_shot"]
    comparison = {
        "direct_energy_ratio_valid_for_model": True,
        "dpu_power_over_cpu": args.dpu_server_power_w / cpu_system_power_w,
        "cpu_time_over_dpu": rows[0]["projected_seconds"] / rows[1]["projected_seconds"],
        "dpu_energy_over_cpu": dpu_energy_j_per_shot / cpu_energy_j_per_shot,
        "dpu_energy_saving_percent": (
            1.0 - dpu_energy_j_per_shot / cpu_energy_j_per_shot
        ) * 100.0,
        "qualification": (
            "both rows use an aligned whole-system component-estimate boundary; "
            "neither is a matched pair of wall-power measurements; "
            + (args.cpu_power_transfer_note or "CPU power is not paired with paper throughput")
            if paper_model else
            "both rows use an aligned whole-system component-estimate boundary; "
            "neither is a matched pair of wall-power measurements"
        ),
    }
    provenance = {
        "timing_model": str(args.timing_model.resolve()),
        "configuration": config["name"] if config else "resident_default",
        "configuration_settings": config["settings"] if config else None,
        "throughput_validation_shots": validation_shots,
        "throughput_result_kind": throughput_result_kind,
        "cpu_rate_basis": cpu_rate_basis,
        "dpu_rate_basis": dpu_rate_basis,
        "cpu_workers": cpu_workers,
        "dpus": dpus,
        "dpu_mhz": dpu_mhz,
        "power_assumptions": {
            "cpu_component_power_w": cpu_component_power_w,
            "cpu_system_overhead_w": args.cpu_system_overhead_w,
            "cpu_full_system_power_w": cpu_system_power_w,
            "cpu_power_kind": cpu_power_kind,
            "cpu_power_domain": args.cpu_power_domain,
            "cpu_power_summaries": [
                str(path.resolve()) for path in args.cpu_power_summary
            ],
            "cpu_power_transfer_note": args.cpu_power_transfer_note,
            "dpu_dimm_power_w": args.dpu_dimm_power_w,
            "dpu_dimms": args.dpu_dimms,
            "dpu_array_power_w": dpu_array_power_w,
            "dpu_reference_server_power_w": args.dpu_server_power_w,
        },
        "references": {
            "intel_i9_10900": (
                "https://www.intel.com/content/www/us/en/products/sku/199328/"
                "intel-core-i910900-processor-20m-cache-up-to-5-20-ghz/"
                "specifications.html"
            ),
            "upmem_reference_platform": "https://www.upmem.com/technology/",
            "pim_dimm_and_server_power": (
                "https://link.springer.com/chapter/10.1007/978-3-031-48803-0_13"
            ),
            "windows_power_meter_units": (
                "https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/"
                "pmi/ns-pmi-_pmi_reported_capabilities"
            ),
        },
        "limitations": [
            "uPIMulator does not model or measure electrical power",
            "CPU TDP is not measured workload power unless RAPL summaries are supplied",
            "DPU throughput is an extrapolation, not a 2560-DPU measurement",
            "CPU non-RAPL system power uses a generic 115 W literature allowance",
            "The 770 W value is a component estimate, not this workload's measured wall power",
            "One-time DPU program/model loading is not included in steady-state throughput",
            "Paper CPU 16-thread throughput and DPU 2560 throughput are both modeled, not measured at those scales" if paper_model else "Resident model and RAPL throughput may differ by run",
        ],
    }
    if args.cpu_power_transfer_note:
        provenance["limitations"].append(args.cpu_power_transfer_note)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "energy_comparison.csv"
    json_path = args.output_dir / "energy_comparison.json"
    report_path = args.output_dir / "ENERGY_REPORT.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "provenance": provenance,
                "comparison": comparison,
                "cpu_context": cpu_context,
                "rows": rows,
            },
            f,
            indent=2,
        )
        f.write("\n")

    lines = [
        "# CPU/DPU power and energy table (provisional)",
        "",
        f"Decoder configuration: `{args.configuration if paper_model else 'resident_default'}`; 16 CPU threads vs 2,560 modeled DPUs; {args.shots:,} projected shots.",
        (
            f"The paper short throughput model was validated on {validation_shots:,} shots; "
            f"{args.shots:,}-shot time and energy are projections, not measured runs."
            if paper_model else
            "Times and energies are calculated from the supplied throughput and power."
        ),
        "",
        "> Both rows use the same whole-system accounting categories. CPU combines",
        "> CPU component power with a 115 W system-overhead allowance; DPU uses",
        "> the literature's complete 770 W reference-server estimate.",
        f"> CPU RAPL qualification: {args.cpu_power_transfer_note or 'see power evidence below'}.",
        "",
        f"| configuration | boundary | W | shot/s | mJ/shot | shot/J | {args.shots:,}-shot time | {args.shots:,}-shot energy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {configuration} | {power_boundary} | {power_w:.2f} | "
            "{throughput_shots_per_s:.2f} | {energy_mj_per_shot:.3f} | "
            "{shots_per_joule:.3f} | {projected_seconds:.3f} s | "
            "{projected_energy_j:.3f} J |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Comparison status",
            "",
            (
                f"DPU draws {comparison['dpu_power_over_cpu']:.2f}× the modeled CPU "
                f"system power, finishes the {args.shots:,}-shot job {comparison['cpu_time_over_dpu']:.2f}× "
                "faster, and uses "
                f"{comparison['dpu_energy_saving_percent']:.2f}% less modeled total energy."
            ),
            "This is a conditional model result, not a matched wall-power measurement or a demonstrated hardware energy saving.",
            "",
            "## Interpretation limits",
            "",
            "- The DPU throughput already includes steady-state host packing, H2D, launch, D2H, and fallback service costs.",
            "- One-time server startup and DPU program/model loading are not represented.",
            "- A final DPU claim requires wall-power or board telemetry on real hardware.",
            f"- CPU throughput basis: {cpu_rate_basis}.",
            f"- CPU power transfer: {args.cpu_power_transfer_note or 'none recorded'}.",
            "",
            "## Inputs and provenance",
            "",
            f"- {cpu_power_label}: {cpu_component_power_w:.2f} W; boundary: {cpu_boundary}.",
            f"- CPU common-system allowance: {args.cpu_system_overhead_w:.2f} W; full-system model: {cpu_system_power_w:.2f} W.",
            f"- DPU array: {args.dpu_dimms} DIMMs × {args.dpu_dimm_power_w:.2f} W = {dpu_array_power_w:.2f} W.",
            "- UPMEM topology: 128 DPUs per DIMM and 20 DIMMs per 2,560-DPU reference platform.",
            f"- DPU performance: `{args.timing_model}` at {dpu_mhz:.0f} MHz; {dpu_rate_basis}.",
            "- Intel specification: https://www.intel.com/content/www/us/en/products/sku/199328/intel-core-i910900-processor-20m-cache-up-to-5-20-ghz/specifications.html",
            "- UPMEM reference platform: https://www.upmem.com/technology/",
            "- PIM DIMM/server power method: https://link.springer.com/chapter/10.1007/978-3-031-48803-0_13",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(report_path)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
