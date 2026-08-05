import unittest

import numpy as np

from evaluate_a6_outer import window_losses
from statistical_tests import moving_block_bootstrap_difference


class AnalyzeA9Pems03ExternalTests(unittest.TestCase):
    def test_strict_bootstrap_signal(self):
        truth = np.arange(96, dtype=float).reshape(2, 12, 4, 1)
        a9 = truth + 0.05
        dcrnn = truth + 0.5
        a9_loss = window_losses(truth, a9)["rmse"]
        dcrnn_loss = window_losses(truth, dcrnn)["rmse"]
        result = moving_block_bootstrap_difference(
            a9_loss, dcrnn_loss, block_length=1, repetitions=100, seed=42
        )
        self.assertLess(result["ci_upper_95"], 0.0)


if __name__ == "__main__":
    unittest.main()
