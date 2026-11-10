#!/usr/bin/env python3
"""Regression test for the finite-shot DPU cap in the scale-out fit."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ScaleoutFitTests(unittest.TestCase):
    def test_2560_dpus_cannot_run_more_than_1000_shots_at_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            measurements = root / "measurements.csv"
            output = root / "fit.csv"
            cpu_summary = root / "cpu_summary.csv"
            with cpu_summary.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["configuration", "cpu_threads", "median_process_wall_throughput_per_second"],
                )
                writer.writeheader()
                writer.writerow({
                    "configuration": "default",
                    "cpu_threads": 16,
                    "median_process_wall_throughput_per_second": 250,
                })
            with measurements.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "backend", "cpu_threads", "dpu_count", "shots",
                        "process_wall_seconds", "cpu_only_full_system_throughput_per_second",
                        "predictions_match_cpu",
                    ],
                )
                writer.writeheader()
                for dpus in (4, 8, 16):
                    writer.writerow(
                        {
                            "backend": "simulator",
                            "cpu_threads": 16,
                            "dpu_count": dpus,
                            "shots": 1000,
                            "process_wall_seconds": 1000 * (0.001 + 1 / dpus),
                            "cpu_only_full_system_throughput_per_second": "" if dpus == 8 else 100,
                            "predictions_match_cpu": "true",
                        }
                    )
            subprocess.run(
                [
                    sys.executable, str(Path(__file__).with_name("fit_scaleout.py")),
                    str(measurements), str(output), "--target-dpus", "2560",
                    "--cap-by-shots", "--cpu-baseline-summary", str(cpu_summary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with output.open(newline="", encoding="utf-8") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(int(row["fixed_shots"]), 1000)
            self.assertEqual(int(row["effective_parallel_dpus"]), 1000)
            self.assertAlmostEqual(float(row["maximum_instantaneous_fraction_of_target_dpus"]), 1000 / 2560)
            self.assertAlmostEqual(float(row["predicted_hybrid_end_to_end_throughput_per_second"]), 500)
            self.assertAlmostEqual(float(row["cpu_only_full_system_throughput_median_per_second"]), 250)
            self.assertAlmostEqual(float(row["predicted_full_system_speedup_vs_cpu_only"]), 2)


if __name__ == "__main__":
    unittest.main()
