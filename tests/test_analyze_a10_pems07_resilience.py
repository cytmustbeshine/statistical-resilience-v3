import unittest

import numpy as np
import pandas as pd

from analyze_a10_pems07_resilience import aggregate_overlapping_forecasts, gate_decision


class AnalyzeA10Pems07ResilienceTests(unittest.TestCase):
    def test_aggregate_overlapping_forecasts(self):
        values = np.array([1.0, 2.0, 4.0, 6.0]).reshape(2, 2, 1, 1)
        targets, aggregated = aggregate_overlapping_forecasts(values, np.array([0, 1]), history=1)
        np.testing.assert_array_equal(targets, np.array([1, 2, 3]))
        np.testing.assert_allclose(aggregated[:, 0], np.array([1.0, 3.0, 6.0]))

    def test_gate_requires_majority_and_mean_improvement(self):
        frame = pd.DataFrame(
            {
                "deficit_mae_ratio": [0.9, 0.95, 1.02],
                "deficit_rmse_ratio": [0.9, 0.98, 1.01],
                "q90_tail_mae_ratio": [0.8, 0.9, 1.1],
                "a10_high_state_f1": [0.7, 0.8, 0.75],
                "dcrnn_high_state_f1": [0.6, 0.7, 0.7],
            }
        )
        self.assertTrue(gate_decision(frame)["resilience_gate_passed"])

    def test_gate_rejects_lower_mean_f1(self):
        frame = pd.DataFrame(
            {
                "deficit_mae_ratio": [0.9, 0.95, 1.02],
                "deficit_rmse_ratio": [0.9, 0.98, 1.01],
                "q90_tail_mae_ratio": [0.8, 0.9, 1.1],
                "a10_high_state_f1": [0.5, 0.5, 0.5],
                "dcrnn_high_state_f1": [0.6, 0.6, 0.6],
            }
        )
        self.assertFalse(gate_decision(frame)["resilience_gate_passed"])


if __name__ == "__main__":
    unittest.main()
