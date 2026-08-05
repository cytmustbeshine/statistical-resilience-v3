import tempfile
import unittest
from pathlib import Path

import numpy as np

from hierarchical_statistical_calibration import (
    HierarchicalCalibration,
    fit_hierarchical_calibration,
)
from l4_prediction_evaluation import regression_metrics


class HierarchicalStatisticalCalibrationTests(unittest.TestCase):
    def test_calibration_improves_biased_forecast(self):
        rng = np.random.default_rng(42)
        truth = rng.uniform(10.0, 100.0, size=(40, 3, 4, 1))
        prediction = truth - 4.0
        persistence = truth - 8.0
        calibration = fit_hierarchical_calibration(truth, prediction, persistence)
        calibrated = calibration.apply(prediction, persistence)
        self.assertLess(
            regression_metrics(truth, calibrated)["mae"],
            regression_metrics(truth, prediction)["mae"],
        )

    def test_roundtrip(self):
        calibration = HierarchicalCalibration(
            blend_weight=0.9,
            correction=np.ones((2, 3, 1)),
            correction_shrinkage=0.25,
            calibration_metric_ratios={"mae": 0.9, "rmse": 0.95, "smape": 0.8, "wape": 0.9},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "calibration.npz"
            calibration.save(path)
            loaded = HierarchicalCalibration.load(path)
        self.assertEqual(loaded.blend_weight, calibration.blend_weight)
        self.assertTrue(np.array_equal(loaded.correction, calibration.correction))
        self.assertEqual(loaded.calibration_metric_ratios, calibration.calibration_metric_ratios)


if __name__ == "__main__":
    unittest.main()
