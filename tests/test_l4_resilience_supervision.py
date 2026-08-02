"""Tests for frozen node-level L4 auxiliary supervision."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_latent_traffic_performance import load_saved_profile
from audit_l4_prediction_pipeline import DATASETS, ordered_frame
from flow_speed_resilience import compute_directional_probabilistic_deficit
from l4_prediction_pipeline import (
    load_ordered_univariate_series_physical,
    paired_column_plan,
    profile_compatible_event_external_split,
)
from l4_resilience_supervision import build_frozen_l4_supervision


class FrozenL4SupervisionTests(unittest.TestCase):
    def load_case(self, dataset: str, variable: str):
        config = DATASETS[dataset]
        frame = ordered_frame(config)
        plan = paired_column_plan(list(frame.columns), config["flow_suffix"], config["speed_suffix"], 41)
        if variable == "flow":
            columns = plan["selected_flow_columns"]
            root = Path(r"D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4") / dataset
            label = "demand"
        else:
            columns = plan["paired_speed_columns"]
            root = Path(r"D:\TrafficGNN\outputs\latent_traffic_performance_l3") / dataset
            label = "speed"
        values, timestamps, names = load_ordered_univariate_series_physical(
            str(config["csv"]), str(config["time_col"]), columns, config[f"{variable}_suffix"]
        )
        train_end = profile_compatible_event_external_split(dataset)["train_time_end_exclusive"]
        profile = load_saved_profile(root, label, values[..., 0], train_end, "log1p_nonnegative")
        self.assertIsNotNone(profile)
        return values, timestamps, names, train_end, profile

    def test_demand_target_matches_frozen_l4(self):
        values, timestamps, names, train_end, bundle = self.load_case("rainstorm", "flow")
        result = build_frozen_l4_supervision(
            values, timestamps, names, "flow", bundle["profile"], bundle["ecdf"], train_end, 12, 12
        )
        reference = compute_directional_probabilistic_deficit(values[..., 0], bundle, timestamps, "lower")
        np.testing.assert_allclose(
            result["deficit_targets"][0, :, :, 0],
            reference["probabilistic_deficit"][12:24],
            equal_nan=True,
        )

    def test_efficiency_target_matches_frozen_l4(self):
        values, timestamps, names, train_end, bundle = self.load_case("typhoon", "speed")
        result = build_frozen_l4_supervision(
            values, timestamps, names, "speed", bundle["profile"], bundle["ecdf"], train_end, 12, 12
        )
        self.assertEqual(result["deficit_targets"].shape[2], 16)
        self.assertEqual(result["metadata"]["dimension"], "efficiency")

    def test_missing_target_remains_nan(self):
        values, timestamps, names, train_end, bundle = self.load_case("rainstorm", "speed")
        result = build_frozen_l4_supervision(
            values, timestamps, names, "speed", bundle["profile"], bundle["ecdf"], train_end, 12, 12
        )
        self.assertGreater(int(np.isnan(result["deficit_targets"]).sum()), 0)
        self.assertEqual(int(np.isnan(result["deficit_targets"]).sum()), int((~result["valid_mask"]).sum()))

    def test_train_boundary_must_match_frozen_profile(self):
        values, timestamps, names, train_end, bundle = self.load_case("bridge", "flow")
        with self.assertRaises(ValueError):
            build_frozen_l4_supervision(
                values, timestamps, names, "flow", bundle["profile"], bundle["ecdf"], train_end + 1, 12, 12
            )

    def test_q90_q99_are_train_only_reference_values(self):
        values, timestamps, names, train_end, bundle = self.load_case("typhoon", "flow")
        result = build_frozen_l4_supervision(
            values, timestamps, names, "flow", bundle["profile"], bundle["ecdf"], train_end, 12, 12
        )
        reference = compute_directional_probabilistic_deficit(values[..., 0], bundle, timestamps, "lower")
        self.assertAlmostEqual(result["q90"], reference["train_deficit_quantiles"]["q90"], places=12)
        self.assertAlmostEqual(result["q99"], reference["train_deficit_quantiles"]["q99"], places=12)

    def test_npz_roundtrip_preserves_nan_and_masks(self):
        values, timestamps, names, train_end, bundle = self.load_case("typhoon", "speed")
        result = build_frozen_l4_supervision(
            values, timestamps, names, "speed", bundle["profile"], bundle["ecdf"], train_end, 12, 12
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "supervision.npz"
            np.savez_compressed(path, deficits=result["deficit_targets"], valid=result["valid_mask"])
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(np.isnan(archive["deficits"]), np.isnan(result["deficit_targets"]))
                np.testing.assert_array_equal(archive["valid"], result["valid_mask"])

    def test_no_scalar_l4_output(self):
        values, timestamps, names, train_end, bundle = self.load_case("bridge", "flow")
        result = build_frozen_l4_supervision(
            values, timestamps, names, "flow", bundle["profile"], bundle["ecdf"], train_end, 12, 12
        )
        self.assertEqual(result["deficit_targets"].ndim, 4)
        self.assertNotIn("scalar_l4", result)


if __name__ == "__main__":
    unittest.main()
