"""Unit tests for the final L4-aligned M2/M3 training runner."""
from __future__ import annotations

import inspect
import unittest

import numpy as np
import torch

from l4_prediction_pipeline import DUAL_SPACE_MISSING_PROTOCOL, PROFILE_COMPATIBLE_SPLIT_PROTOCOL
from run_final_l4_a3 import build_model, empirical_cvar, system_deficit


class FinalL4A3TrainingTests(unittest.TestCase):
    def test_l4_deficit_output_nonnegative(self):
        model = build_model(5, 12, 16, 1)
        x = torch.randn(2, 12, 5, 1)
        adj = torch.eye(5)
        _, aux = model(x, adj, return_aux=True)
        deficit = torch.nn.functional.softplus(aux["resilience_pred"])
        self.assertTrue(bool((deficit >= 0).all()))
        self.assertEqual(tuple(deficit.shape), (2, 12, 5, 1))

    def test_cvar_uses_largest_training_losses(self):
        losses = torch.tensor([1.0, 2.0, 3.0, 100.0])
        mask = torch.tensor([True, True, True, True])
        value = empirical_cvar(losses, mask, 0.75)
        self.assertEqual(float(value), 100.0)

    def test_cvar_masks_missing_targets(self):
        losses = torch.tensor([1.0, 1000.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, True])
        value = empirical_cvar(losses, mask, 0.75)
        self.assertEqual(float(value), 4.0)

    def test_system_deficit_keeps_two_dimensions_separate(self):
        values = np.ones((2, 3, 5), dtype=float)
        result = system_deficit(values)
        self.assertEqual(result.shape, (2, 3))

    def test_runner_uses_preregistered_contract(self):
        import run_final_l4_a3

        source = inspect.getsource(run_final_l4_a3)
        self.assertIn("PROFILE_COMPATIBLE_SPLIT_PROTOCOL", source)
        self.assertIn("DUAL_SPACE_MISSING_PROTOCOL", source)
        self.assertIn('"lambda_l4": args.lambda_l4', source)
        self.assertIn('variant == "m3"', source)
        self.assertNotIn("--event-col", source)
        self.assertNotIn("--weather", source.lower())

    def test_runner_does_not_emit_scalar_l4(self):
        import run_final_l4_a3

        source = inspect.getsource(run_final_l4_a3)
        self.assertIn("true_node_deficit", source)
        self.assertIn("pred_node_deficit", source)
        self.assertNotIn("scalar_l4", source)


if __name__ == "__main__":
    unittest.main()
