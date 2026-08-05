import unittest

import numpy as np

from evaluate_a6_outer import window_losses
from statistical_tests import moving_block_bootstrap_difference


class AnalyzeA8PemsExternalTests(unittest.TestCase):
    def test_window_loss_and_bootstrap_detect_improvement(self):
        truth = np.arange(24, dtype=float).reshape(2, 3, 4, 1)
        candidate = truth + 0.1
        baseline = truth + 1.0
        candidate_losses = window_losses(truth, candidate)
        baseline_losses = window_losses(truth, baseline)
        result = moving_block_bootstrap_difference(
            candidate_losses["mae"],
            baseline_losses["mae"],
            block_length=1,
            repetitions=100,
            seed=42,
        )
        self.assertLess(result["ci_upper_95"], 0.0)


if __name__ == "__main__":
    unittest.main()
