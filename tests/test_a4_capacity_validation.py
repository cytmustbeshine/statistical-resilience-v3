import unittest

import pandas as pd

from run_a4_capacity_validation import (
    ResidualForecastWrapper,
    candidate_summary,
    model_for_baseline,
    parse_float_list,
    parse_int_list,
    zero_initialize_residual_head,
)

import torch


class A4CapacityValidationTests(unittest.TestCase):
    def test_candidate_parser_deduplicates(self):
        self.assertEqual(parse_int_list("64, 128,64"), [64, 128])
        self.assertEqual(parse_float_list("0.001,0.0005,0.001"), [0.001, 0.0005])

    def test_candidate_summary_applies_preregistered_gate(self):
        tasks = [("bridge", "flow"), ("rainstorm", "flow"), ("rainstorm", "speed"), ("typhoon", "flow"), ("typhoon", "speed")]
        rows = []
        for dataset, variable in tasks:
            rows.append({"dataset": dataset, "variable": variable, "seed": 42, "hidden_dim": 64, "lr": 0.001, "best_val_mae": 10.0, "finite": True, "parameter_count": 1_000_000})
            rows.append({"dataset": dataset, "variable": variable, "seed": 42, "hidden_dim": 160, "lr": 0.001, "best_val_mae": 9.5, "finite": True, "parameter_count": 3_400_000})
        summary = candidate_summary(pd.DataFrame(rows))
        selected = summary.loc[summary["hidden_dim"] == 160].iloc[0]
        self.assertTrue(bool(selected["advances"]))
        self.assertEqual(int(selected["improved_tasks"]), 5)

    def test_zero_initialized_residual_starts_at_persistence(self):
        base = model_for_baseline(5, 3, 64, 2)
        zero_initialize_residual_head(base)
        self.assertEqual(torch.count_nonzero(base.output[-1].weight).item(), 0)
        wrapper = ResidualForecastWrapper(base, horizon=3, residual_scale=1.0)
        wrapper.eval()
        x = torch.randn(2, 6, 5, 1)
        adjacency = torch.eye(5)
        with torch.no_grad():
            prediction = wrapper(x, adjacency)
        expected = x[:, -1:, :, :].expand(-1, 3, -1, -1)
        self.assertTrue(torch.allclose(prediction, expected, atol=1e-6))

if __name__ == "__main__":
    unittest.main()