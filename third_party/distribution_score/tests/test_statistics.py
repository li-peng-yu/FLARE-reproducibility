from __future__ import annotations

import unittest

import numpy as np

from distribution_score.statistics import (
    calibration_summary,
    density_ratio_summary,
    probability_rank_from_distances,
    same_condition_score,
)


class DensityRatioTests(unittest.TestCase):
    def test_equal_density_is_perfect(self) -> None:
        result = density_ratio_summary(np.array([0.2, 0.7]), np.array([0.2, 0.7]))
        self.assertAlmostEqual(result["symmetric_ratio_score"], 1.0)
        np.testing.assert_allclose(result["per_truth_symmetric_score"], 1.0)

    def test_reciprocal_ratios_have_equal_penalty(self) -> None:
        result = density_ratio_summary(np.array([2.0, 0.5]), np.ones(2))
        np.testing.assert_allclose(result["per_truth_symmetric_score"], [0.5, 0.5])
        self.assertAlmostEqual(result["symmetric_ratio_score"], 0.5)

    def test_unequal_sample_counts_are_normalized(self) -> None:
        # Every kernel value is identical. N=500 and M=30 must still produce ratio one.
        cross = np.full((30, 500), 0.25)
        self_distance = np.full((30, 30), 0.25)
        np.fill_diagonal(self_distance, 0.0)
        result = same_condition_score(cross, self_distance, sigma=0.25)
        self.assertAlmostEqual(result["symmetric_ratio_score"], 1.0)
        np.testing.assert_allclose(result["density_ratio"], 1.0)


class ProbabilityCalibrationTests(unittest.TestCase):
    def test_probability_rank_uses_truth_as_last_query(self) -> None:
        reference_self = np.array(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        )
        # The truth (last row) is denser than both ranking rows, so u=1/(2+1).
        query = np.array([[2.0, 2.0, 2.0], [1.5, 1.5, 1.5], [0.1, 0.1, 0.1]])
        result = probability_rank_from_distances(reference_self, query, sigma=1.0)
        self.assertAlmostEqual(result["probability_rank_u"], 1.0 / 3.0)

    def test_exact_decile_grid_is_calibrated(self) -> None:
        values = np.arange(0.05, 1.0, 0.1)
        summary = calibration_summary(values)
        self.assertAlmostEqual(summary["calibration_score"], 1.0)
        np.testing.assert_allclose(summary["coverage"], np.arange(0.1, 1.0, 0.1))


if __name__ == "__main__":
    unittest.main()
