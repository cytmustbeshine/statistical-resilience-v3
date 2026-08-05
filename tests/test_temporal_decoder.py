import unittest

import torch

from run_a4_capacity_validation import model_for_baseline
from temporal_decoder import AutoregressiveForecastWrapper


class AutoregressiveDecoderTests(unittest.TestCase):
    def test_decoder_shape_and_gradients(self):
        base = model_for_baseline(5, 3, 64, 2)
        model = AutoregressiveForecastWrapper(base, hidden_dim=64, horizon=3)
        x = torch.randn(2, 6, 5, 1)
        adjacency = torch.eye(5)
        prediction, aux = model(x, adjacency, return_aux=True)
        self.assertEqual(tuple(prediction.shape), (2, 3, 5, 1))
        self.assertEqual(tuple(aux["final_state"].shape), (2, 5, 64))
        prediction.mean().backward()
        self.assertIsNotNone(model.output_projection.weight.grad)

    def test_decoder_predictions_are_finite(self):
        base = model_for_baseline(5, 3, 64, 2)
        model = AutoregressiveForecastWrapper(base, hidden_dim=64, horizon=3)
        prediction = model(torch.randn(1, 6, 5, 1), torch.eye(5))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_teacher_forcing_changes_training_path(self):
        torch.manual_seed(7)
        base = model_for_baseline(5, 3, 64, 2)
        model = AutoregressiveForecastWrapper(base, hidden_dim=64, horizon=3)
        model.train()
        x = torch.randn(2, 6, 5, 1)
        labels = torch.full((2, 3, 5, 1), 4.0)
        torch.manual_seed(19)
        free_running = model(x, torch.eye(5), labels=labels, teacher_forcing_ratio=0.0)
        torch.manual_seed(19)
        forced = model(x, torch.eye(5), labels=labels, teacher_forcing_ratio=1.0)
        self.assertTrue(torch.allclose(free_running[:, 0], forced[:, 0]))
        self.assertFalse(torch.allclose(free_running[:, 1:], forced[:, 1:]))

    def test_nonfinite_teacher_targets_fall_back_to_predictions(self):
        torch.manual_seed(11)
        base = model_for_baseline(5, 3, 64, 2)
        model = AutoregressiveForecastWrapper(base, hidden_dim=64, horizon=3)
        model.train()
        x = torch.randn(1, 6, 5, 1)
        labels = torch.full((1, 3, 5, 1), float("nan"))
        torch.manual_seed(23)
        free_running = model(x, torch.eye(5), teacher_forcing_ratio=0.0)
        torch.manual_seed(23)
        forced = model(x, torch.eye(5), labels=labels, teacher_forcing_ratio=1.0)
        self.assertTrue(torch.allclose(free_running, forced))

    def test_invalid_teacher_forcing_inputs_are_rejected(self):
        base = model_for_baseline(5, 3, 64, 2)
        model = AutoregressiveForecastWrapper(base, hidden_dim=64, horizon=3)
        x = torch.randn(1, 6, 5, 1)
        with self.assertRaises(ValueError):
            model(x, torch.eye(5), teacher_forcing_ratio=1.1)
        with self.assertRaises(ValueError):
            model(x, torch.eye(5), labels=torch.randn(1, 2, 5, 1), teacher_forcing_ratio=1.0)


if __name__ == "__main__":
    unittest.main()
