#!/usr/bin/env python3
"""Render the final two-row throughput table after full validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("timing_model", type=Path)
    parser.add_argument("population_summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--required-shots", type=int, default=100_000_000)
    parser.add_argument("--minimum-cycle-traces", type=int, default=50)
    args = parser.parse_args()

    model = json.loads(args.timing_model.read_text())
    population = json.loads(args.population_summary.read_text())
    inputs = model["inputs"]
    primary = model["throughput_primary"]
    sensitivity = model["throughput_pessimistic_slope"]
    cpu = population["cpu"]
    resident = population["resident"]
    predictions = population["predictions"]

    if int(cpu["shots"]) != args.required_shots:
        raise SystemExit(f"refusing final report: {cpu['shots']} != {args.required_shots} shots")
    if int(resident["dpu_attempts"]) != args.required_shots:
        raise SystemExit("refusing final report: DPU-attempt count is incomplete")
    if not predictions["all_solution_cost_deltas_within_q16_bound"]:
        raise SystemExit("refusing final report: CPU/resident cost exceeds the Q16 error bound")
    cycle_traces = int(inputs["cycle_traces_in_weighted_calibration"])
    if cycle_traces < args.minimum_cycle_traces:
        raise SystemExit(
            f"refusing final report: only {cycle_traces} weighted cycle traces; "
            f"need {args.minimum_cycle_traces}"
        )

    cpu_rate = float(inputs["cpu_only_full_system_rate_per_s"])
    dpu_rate = float(primary["system_per_s"])
    speedup = float(primary["speedup"])
    dpu_ms = float(primary["mean_dpu_seconds_per_shot"]) * 1e3
    host_bound = float(primary["cpu_bound_per_s"])
    dpu_bound = float(primary["dpu_bound_per_s"])
    conservative_rate = float(sensitivity["system_per_s"])
    conservative_speedup = float(sensitivity["speedup"])

    lines = [
        "# 최종 처리량 결과",
        "",
        "| 구성 | 산출 방식 | shot/s |",
        "|---|---|---:|",
        (
            "| CPU-only (16 thread) | i9-10900, 100,000-shot wall-time 3회 중앙값 "
            f"(실측) | {cpu_rate:,.0f} |"
        ),
        (
            f"| DPU 오프로딩 (2,560 DPU) | {args.required_shots:,}-shot 전수 작업량 + "
            f"층화·가중 cycle trace {cycle_traces}개; shot당 평균 DPU 시간 "
            f"{dpu_ms:.3f} ms; min(host {host_bound:,.0f}, DPU {dpu_bound:,.0f}) "
            f"(시뮬레이터 외삽) | **{dpu_rate:,.0f} ({speedup:.2f}×)** |"
        ),
        "",
        (f"민감도 분석(가중 선형 slope): **{conservative_rate:,.0f} shot/s "
         f"({conservative_speedup:.2f}×)**."),
        "",
        ("speedup은 DPU 코어와 CPU 코어의 단일 코어 성능비가 아니라, "
         "**2,560-DPU 모델의 full-system 처리량 / CPU 16-thread 실측 처리량**이다."),
        "",
        (f"CPU와 resident oracle은 {args.required_shots:,}개 shot에서 solution-cost 차이가 "
         f"Q16 누적 반올림 상한 이하였다. 동률 correction의 observable prediction 불일치는 "
         f"{predictions['prediction_mismatches']:,}건이며, logical error는 CPU "
         f"{cpu['logical_errors']:,}건, resident {resident['logical_errors']:,}건이다. DPU 수치는 실제 "
         "2,560-DPU 장비 측정이 아니라 350 MHz cycle-level 시뮬레이터 외삽이다."),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
