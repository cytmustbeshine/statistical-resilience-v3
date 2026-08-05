import unittest

import numpy as np

from analyze_a10_pems07_external import horizon_mae


class AnalyzeA10Pems07ExternalTests(unittest.TestCase):
    def test_horizon_mae(self):
        truth = np.zeros((2, 3, 4, 1), dtype=float)
        prediction = np.ones_like(truth)
        self.assertTrue(np.array_equal(horizon_mae(truth, prediction), np.ones(3)))


if __name__ == "__main__":
    unittest.main()
