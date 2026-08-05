import unittest

import numpy as np
import pandas as pd
import torch

from statistical_identity_model import (
    StatisticalIdentityResidualForecaster,
    fit_shrunk_seasonal_baseline,
)


class StatisticalIdentityModelTests(unittest.TestCase):
    def test_initial_prediction_equals_seasonal_baseline(self):
        model = StatisticalIdentityResidualForecaster(5)
        x = torch.randn(2, 12, 5, 1)
        baseline = torch.randn(2, 12, 5, 1)
        prediction = model(x, torch.eye(5), baseline, torch.tensor([1, 2]), torch.tensor([0, 1]))
        self.assertTrue(torch.allclose(prediction, baseline))

    def test_identity_embeddings_receive_gradients(self):
        model = StatisticalIdentityResidualForecaster(5)
        with torch.no_grad():
            model.output_projection.weight.normal_(0.0, 0.01)
        x = torch.randn(2, 12, 5, 1)
        baseline = torch.zeros(2, 12, 5, 1)
        prediction = model(x, torch.eye(5), baseline, torch.tensor([1, 2]), torch.tensor([0, 1]))
        prediction.square().mean().backward()
        self.assertGreater(float(model.node_embedding.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.time_of_day_embedding.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.day_of_week_embedding.weight.grad.abs().sum()), 0.0)

    def test_seasonal_baseline_uses_training_rows_only(self):
        timestamps = pd.date_range("2020-01-01", periods=600, freq="5min").to_numpy()
        values = np.arange(600 * 3, dtype=float).reshape(600, 3, 1)
        first, first_meta = fit_shrunk_seasonal_baseline(values, timestamps, 400)
        changed = values.copy()
        changed[400:] += 100000.0
        second, second_meta = fit_shrunk_seasonal_baseline(changed, timestamps, 400)
        self.assertTrue(np.allclose(first, second))
        self.assertEqual(first_meta["baseline_sha256"], second_meta["baseline_sha256"])


if __name__ == "__main__":
    unittest.main()
