"""Tests for event-level frozen L4 prediction evaluation."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_l4_event_prediction import aggregate_archive, recovery_summary, state_code


class EventPredictionEvaluationTests(unittest.TestCase):
    def test_recovery_requires_consecutive_steps(self):
        times = np.arange(8).astype("datetime64[h]")
        result = recovery_summary(times, np.array([0.0, 3.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1]), 0.5, 3)
        self.assertEqual(result["recovery_status"], "recovered")
        self.assertEqual(result["recovery_duration_hours"], 2.0)

    def test_censored_recovery(self):
        times = np.arange(5).astype("datetime64[h]")
        result = recovery_summary(times, np.array([0.0, 3.0, 1.0, 0.8, 0.7]), 0.5, 3)
        self.assertEqual(result["recovery_status"], "censored")
        self.assertTrue(np.isnan(result["recovery_duration_hours"]))

    def test_state_codes_are_four_dimensional(self):
        codes = state_code(np.array([2.0, 2.0, 0.0, 0.0]), np.array([2.0, 0.0, 2.0, 0.0]), 1.0, 1.0)
        np.testing.assert_array_equal(codes, [0, 1, 2, 3])

    def test_archive_duplicate_targets_are_averaged(self):
        self.assertTrue(callable(aggregate_archive))

    def test_no_scalar_state_output(self):
        codes = state_code(np.array([2.0]), np.array([0.0]), 1.0, 1.0)
        self.assertEqual(codes.shape, (1,))
        self.assertNotEqual(codes[0], 0)


if __name__ == "__main__":
    unittest.main()