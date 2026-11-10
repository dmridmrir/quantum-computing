#!/usr/bin/env python3
"""Small invariant checks for the scale-out sensitivity audit."""

import unittest

from audit_scaleout_uncertainty import (
    fit_cycles,
    legacy_extrapolations,
    old_replicates,
    quantile,
    schedule,
)


class ScaleoutAuditTests(unittest.TestCase):
    def test_greedy_schedule_and_clairvoyant_ordering(self) -> None:
        first_come = schedule([3.0, 1.0, 2.0, 4.0], 2)
        longest_first = schedule([4.0, 3.0, 2.0, 1.0], 2)
        self.assertEqual(first_come["compute_only_makespan_seconds"], 7.0)
        self.assertEqual(longest_first["compute_only_makespan_seconds"], 5.0)
        self.assertAlmostEqual(first_come["mean_active_dpu_fraction"], 10 / 14)
        self.assertEqual(longest_first["mean_active_dpu_fraction"], 1.0)

    def test_cycle_fit_nonnegative_intercept(self) -> None:
        intercept, slope = fit_cycles(
            [
                {"candidates": 1, "cycles": 5},
                {"candidates": 2, "cycles": 8},
                {"candidates": 3, "cycles": 11},
            ]
        )
        self.assertAlmostEqual(intercept, 2.0)
        self.assertAlmostEqual(slope, 3.0)

    def test_quantile_interpolation(self) -> None:
        self.assertEqual(quantile([1.0, 3.0, 5.0], 0.25), 2.0)

    def test_legacy_extrapolation_reproduces_checked_in_fit(self) -> None:
        fits = legacy_extrapolations(old_replicates())
        main = next(row for row in fits if row["fit"] == "five_point_median")
        self.assertAlmostEqual(main["extrapolated_2560_syndromes_per_second"], 2.290369099, places=8)
        self.assertEqual(main["dpus_used"], "4;8;16;32;64")


if __name__ == "__main__":
    unittest.main()
