"""Tests for E-L4-1 prediction evaluation helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from l4_prediction_evaluation import binary_state_metrics, persistence_forecast, regression_metrics


class L4PredictionEvaluationTests(unittest.TestCase):
    def test_regression_metrics_exact(self):
        metrics = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)

    def test_regression_metrics_ignores_missing(self):
        metrics = regression_metrics(np.array([1.0, np.nan]), np.array([2.0, 3.0]))
        self.assertEqual(metrics["mae"], 1.0)

    def test_persistence_baseline(self):
        values = np.arange(20.0).reshape(10, 2, 1)
        pred = persistence_forecast(values, np.array([0, 2]), history=3, horizon=2)
        np.testing.assert_array_equal(pred[:, 0], values[[2, 4]])
        np.testing.assert_array_equal(pred[:, 1], values[[2, 4]])

    def test_binary_state_metrics(self):
        metrics = binary_state_metrics(np.array([0.0, 2.0, 2.0, 0.0]), np.array([0.0, 2.0, 0.0, 2.0]), 1.0)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)

    def test_high_state_not_scalar_combination(self):
        demand = binary_state_metrics(np.array([0.0, 2.0]), np.array([0.0, 2.0]), 1.0)
        efficiency = binary_state_metrics(np.array([2.0, 0.0]), np.array([2.0, 0.0]), 1.0)
        self.assertEqual(demand["recall"], 1.0)
        self.assertEqual(efficiency["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()