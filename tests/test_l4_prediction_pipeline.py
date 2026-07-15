"""Unittest coverage for the non-training E-L4-0 prediction-pipeline audit."""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_l4_prediction_pipeline import (
    DATASETS,
    audit_split,
    bridge_four_state_available,
    checkpoint_name,
    dataset_matches,
    frozen_profile_available,
    inverse_predictions,
    l4_prediction_schema,
    model_smoke,
    numeric_columns,
    profile_paths,
    save_prediction_npz,
    scaler_audit_row,
    separate_event_segments,
    strict_chronological_window_splits,
    target_bounds,
    target_timestamp_matrix,
)
from data import StandardScaler
from flow_speed_resilience import match_node_variables


class TestL4PredictionPipeline(unittest.TestCase):
    def test_split_windows_do_not_cross_boundaries(self):
        train, val, test, bounds = strict_chronological_window_splits(100, 4, 3)
        for indices, low, high in (
            (train, 0, bounds["train_time_end_exclusive"]),
            (val, bounds["train_time_end_exclusive"], bounds["val_time_end_exclusive"]),
            (test, bounds["val_time_end_exclusive"], 100),
        ):
            for index in indices:
                start, end = target_bounds(index, 4, 3)
                self.assertGreaterEqual(start, low)
                self.assertLess(end, high)

    def test_current_split_leak_is_detected(self):
        audit = audit_split("synthetic", 100, 4, 3)
        self.assertFalse(audit["current_split_leak_free"])
        self.assertTrue(audit["strict_split_leak_free"])

    def test_scaler_fit_train_only(self):
        values = np.concatenate([np.zeros((6, 2)), np.full((4, 2), 100.0)])
        row = scaler_audit_row("x", "flow", values, 6)
        self.assertEqual(row["mean"], 0.0)

    def test_flow_speed_scalers_independent(self):
        flow = scaler_audit_row("x", "flow", np.arange(20.0).reshape(10, 2), 6)
        speed = scaler_audit_row("x", "speed", np.arange(20.0, 40.0).reshape(10, 2), 6)
        self.assertNotEqual(flow["mean"], speed["mean"])
        self.assertNotEqual(flow["scaler_object_scope"], speed["scaler_object_scope"])

    def test_scaler_roundtrip(self):
        row = scaler_audit_row("x", "flow", np.arange(20.0).reshape(10, 2), 6)
        self.assertTrue(row["roundtrip_pass"])

    def test_target_timestamp_alignment(self):
        timestamps = np.arange(20)
        matrix = target_timestamp_matrix(timestamps, [2], history=4, horizon=3)
        np.testing.assert_array_equal(matrix, [[6, 7, 8]])

    def test_multihorizon_alignment(self):
        timestamps = np.arange(30)
        matrix = target_timestamp_matrix(timestamps, [0, 5], history=4, horizon=3)
        self.assertEqual(matrix.shape, (2, 3))
        np.testing.assert_array_equal(matrix[1], [9, 10, 11])

    def test_node_matching_by_name(self):
        matches = match_node_variables(
            ["b_speed", "a_volume", "a_speed", "b_volume"], "_volume", "_speed"
        )
        by_node = {row["node"]: row for row in matches}
        self.assertEqual(by_node["a"]["speed_col"], "a_speed")
        self.assertEqual(by_node["b"]["speed_col"], "b_speed")

    def test_typhoon_fixed_matched_nodes(self):
        columns = []
        for index in range(41):
            columns.append(f"n{index}_volume")
            if index < 16:
                columns.append(f"n{index}_speed")
        frame = pd.DataFrame({column: [1.0] for column in columns})
        frame.insert(0, "Time", ["2020-01-01"])
        config = dict(DATASETS["typhoon"])
        matches, _, _ = dataset_matches(frame, config, 41)
        self.assertEqual(sum(row["speed_col"] is not None for row in matches), 16)

    def test_bridge_speed_unavailable(self):
        self.assertFalse(bridge_four_state_available("bridge", False))

    def test_missing_speed_is_not_fabricated(self):
        matches = match_node_variables(["a_volume"], "_volume", "_speed")
        self.assertIsNone(matches[0]["speed_col"])

    def test_adjacency_matches_node_order(self):
        smoke = model_smoke(num_nodes=4, history=12, horizon=3)
        self.assertEqual(smoke["input_shape"][2], smoke["output_shape"][2])
        self.assertTrue(smoke["shape_pass"])

    def test_prediction_npz_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.npz"
            expected = np.arange(12).reshape(2, 3, 2)
            save_prediction_npz(path, predictions=expected)
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["predictions"], expected)

    def test_checkpoint_names_do_not_collide(self):
        names = {
            checkpoint_name("typhoon", "flow", 42),
            checkpoint_name("typhoon", "speed", 42),
            checkpoint_name("rainstorm", "speed", 42),
        }
        self.assertEqual(len(names), 3)

    def _profile_package(self, directory: str) -> tuple[Path, Path, Path]:
        root = Path(directory)
        hierarchy = root / "x_profile_hierarchical.npz"
        ecdf = root / "x_profile_ecdf.npz"
        report = root / "x_profile_report.json"
        np.savez_compressed(hierarchy, location=np.array([1.0]))
        np.savez_compressed(ecdf, values=np.array([1.0]))
        report.write_text(json.dumps({"train_only": True}), encoding="utf-8")
        return hierarchy, ecdf, report

    def test_l4_profile_not_refit_on_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._profile_package(directory)
            before = tuple(path.stat().st_mtime_ns for path in paths)
            _ = np.full(20, 999.0)
            self.assertTrue(frozen_profile_available(paths))
            self.assertEqual(before, tuple(path.stat().st_mtime_ns for path in paths))

    def test_l4_profile_not_refit_on_test(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._profile_package(directory)
            self.assertTrue(frozen_profile_available(paths))
            metadata = json.loads(paths[2].read_text(encoding="utf-8"))
            self.assertTrue(metadata["train_only"])

    def test_physical_space_used_for_l4(self):
        scaler = StandardScaler()
        raw = np.arange(12.0).reshape(6, 2, 1)
        scaler.fit(raw[:4])
        scaled = scaler.transform(raw)
        np.testing.assert_allclose(inverse_predictions(scaled, scaler), raw)

    def test_missing_prediction_remains_nan(self):
        scaler = StandardScaler()
        scaler.fit(np.arange(8.0).reshape(4, 2, 1))
        values = np.array([[[np.nan]]])
        self.assertTrue(np.isnan(inverse_predictions(values, scaler)[0, 0, 0]))

    def test_standardized_negative_not_clipped(self):
        values = np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]])
        row = scaler_audit_row("x", "speed", values, 2)
        self.assertTrue(row["negative_values_preserved"])
        self.assertEqual(row["negative_value_count"], 2)

    def test_event_signal_not_in_training_features(self):
        frame = pd.DataFrame(
            {"Time": ["2020-01-01"], "a_volume": [1.0], "typhoon_intensity": [5.0]}
        )
        self.assertEqual(numeric_columns(frame, "_volume", "Time"), ["a_volume"])

    def test_event_signal_not_in_loss(self):
        source = inspect.getsource(model_smoke)
        self.assertNotIn("event_signal=", source)
        self.assertNotIn("resilience_aux=True", source)

    def test_prediction_and_truth_use_same_profile(self):
        l4 = Path("l4")
        source = Path("source")
        self.assertEqual(
            profile_paths("rainstorm", "efficiency", l4, source),
            profile_paths("rainstorm", "efficiency", l4, source),
        )

    def test_candidate_uses_train_threshold(self):
        values = np.array([0.0, 1.0, 2.0, 1000.0])
        train_q90 = float(np.quantile(values[:3], 0.9))
        changed_test = values.copy()
        changed_test[3] = -1000.0
        self.assertEqual(train_q90, float(np.quantile(changed_test[:3], 0.9)))

    def test_bridge_skips_four_state_metrics(self):
        self.assertFalse(bridge_four_state_available("bridge", False))
        self.assertTrue(bridge_four_state_available("rainstorm", True))

    def test_typhoon_segments_remain_separate(self):
        segments = [(3194, 3430), (3482, 3718), (3770, 4006)]
        self.assertEqual(separate_event_segments(segments), segments)
        self.assertEqual(len(separate_event_segments(segments)), 3)

    def test_no_scalar_l4_score(self):
        schema = l4_prediction_schema()
        self.assertNotIn("joint_score", schema)
        self.assertNotIn("combined_deficit", schema)
        self.assertIn("demand_deficit_pred", schema)
        self.assertIn("efficiency_deficit_pred", schema)


if __name__ == "__main__":
    unittest.main()
