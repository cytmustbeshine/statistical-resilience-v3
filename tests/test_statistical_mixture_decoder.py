import unittest

import torch

from run_a4_capacity_validation import model_for_baseline
from statistical_mixture_decoder import StatisticalMixtureForecastWrapper


class StatisticalMixtureDecoderTests(unittest.TestCase):
    def test_shape_weights_and_gradients(self):
        base = model_for_baseline(5, 3, 64, 2)
        model = StatisticalMixtureForecastWrapper(base, hidden_dim=64, horizon=3)
        x = torch.randn(2, 6, 5, 1)
        prediction, aux = model(x, torch.eye(5), return_aux=True)
        self.assertEqual(tuple(prediction.shape), (2, 3, 5, 1))
        self.assertEqual(tuple(aux["expert_predictions"].shape), (2, 3, 5, 1, 3))
        self.assertEqual(tuple(aux["mixture_weights"].shape), (2, 3, 5, 3))
        self.assertTrue(torch.allclose(aux["mixture_weights"].sum(dim=-1), torch.ones(2, 3, 5)))
        prediction.mean().backward()
        self.assertIsNotNone(model.output_projection.weight.grad)
        self.assertIsNotNone(model.gate[-1].weight.grad)
        self.assertIsNotNone(model.base_model.output[-1].weight.grad)

    def test_initial_gate_is_equal_weighted(self):
        base = model_for_baseline(4, 2, 64, 2)
        model = StatisticalMixtureForecastWrapper(base, hidden_dim=64, horizon=2)
        _, aux = model(torch.randn(1, 6, 4, 1), torch.eye(4), return_aux=True)
        expected = torch.full_like(aux["mixture_weights"], 1.0 / 3.0)
        self.assertTrue(torch.allclose(aux["mixture_weights"], expected))

    def test_constant_history_produces_finite_state_features(self):
        base = model_for_baseline(4, 2, 64, 2)
        model = StatisticalMixtureForecastWrapper(base, hidden_dim=64, horizon=2)
        _, aux = model(torch.ones(1, 6, 4, 1), torch.eye(4), return_aux=True)
        self.assertTrue(torch.isfinite(aux["traffic_state_features"]).all())
        self.assertTrue(torch.isfinite(aux["mixture_weights"]).all())


if __name__ == "__main__":
    unittest.main()
