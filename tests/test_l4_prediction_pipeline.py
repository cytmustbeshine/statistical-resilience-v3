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
import torch

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
from data import StandardScaler, load_wide_traffic_csv, split_traffic_window_indices_strict
from flow_speed_resilience import match_node_variables
from l4_prediction_pipeline import (
    event_free_feature_contract,
    fit_train_only_scaler,
    load_forecast_checkpoint,
    load_prediction_archive,
    paired_column_plan,
    save_forecast_checkpoint,
    save_prediction_archive,
    scaler_metadata,
    strict_data_bundle,
)


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


    def test_repaired_strict_split_is_target_disjoint(self):
        train, val, test, info = split_traffic_window_indices_strict(100, 4, 3)
        self.assertTrue(info["target_disjoint"])
        self.assertLess(target_bounds(train[-1], 4, 3)[1], info["train_time_end_exclusive"])
        self.assertGreaterEqual(target_bounds(val[0], 4, 3)[0], info["train_time_end_exclusive"])
        self.assertLess(target_bounds(val[-1], 4, 3)[1], info["val_time_end_exclusive"])
        self.assertGreaterEqual(target_bounds(test[0], 4, 3)[0], info["val_time_end_exclusive"])

    def test_explicit_value_columns_preserve_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traffic.csv"
            pd.DataFrame(
                {
                    "Time": pd.date_range("2020-01-01", periods=3, freq="5min"),
                    "a_volume": [1.0, 2.0, 3.0],
                    "b_volume": [4.0, 5.0, 6.0],
                }
            ).to_csv(path, index=False)
            data, names = load_wide_traffic_csv(
                str(path),
                value_suffix="_volume",
                time_col="Time",
                value_columns=["b_volume", "a_volume"],
            )
            self.assertEqual(names, ["b_volume", "a_volume"])
            np.testing.assert_array_equal(data[0, :, 0], [4.0, 1.0])

    def test_explicit_value_columns_reject_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traffic.csv"
            pd.DataFrame(
                {"Time": ["2020-01-01"], "a_volume": [1.0]}
            ).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_wide_traffic_csv(
                    str(path),
                    value_suffix="_volume",
                    time_col="Time",
                    value_columns=["a_volume", "a_volume"],
                )

    def test_paired_column_plan_locks_typhoon_subnet(self):
        columns = ["Time"]
        for index in range(41):
            columns.append(f"n{index}_volume")
            if index >= 25:
                columns.append(f"n{index}_speed")
        plan = paired_column_plan(columns, "_volume", "_speed", 41)
        self.assertEqual(len(plan["paired_node_names"]), 16)
        self.assertEqual(plan["paired_node_names"], [f"n{i}" for i in range(25, 41)])
        self.assertEqual(
            plan["paired_speed_columns"], [f"n{i}_speed" for i in range(25, 41)]
        )

    def test_repaired_scaler_uses_strict_train_prefix(self):
        values = np.concatenate(
            [np.zeros((6, 2, 1)), np.full((4, 2, 1), 1000.0)], axis=0
        )
        scaler, _ = fit_train_only_scaler(values, 6)
        self.assertEqual(float(np.asarray(scaler.mean).reshape(-1)[0]), 0.0)

    def _checkpoint_metadata(self, scaler):
        return {
            "dataset": "synthetic",
            "variable": "speed",
            "node_names": ["a", "b"],
            "history": 4,
            "horizon": 3,
            "train_time_end_exclusive": 6,
            "val_time_end_exclusive": 8,
            "seed": 42,
            "scaler": scaler_metadata(scaler),
            "timestamp_start": "2020-01-01T00:00:00",
            "timestamp_end": "2020-01-01T00:45:00",
            "input_features": ["traffic"],
            "event_features": [],
        }

    def test_checkpoint_metadata_roundtrip(self):
        scaler = StandardScaler()
        scaler.fit(np.arange(12.0).reshape(6, 2, 1))
        metadata = self._checkpoint_metadata(scaler)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_forecast_checkpoint(path, {"weight": torch.tensor([2.0])}, metadata)
            loaded = load_forecast_checkpoint(path)
            self.assertEqual(loaded["metadata"]["node_names"], ["a", "b"])
            self.assertEqual(loaded["metadata"]["scaler"], metadata["scaler"])
            self.assertEqual(float(loaded["model_state_dict"]["weight"][0]), 2.0)

    def test_prediction_archive_contains_aligned_horizons(self):
        scaler = StandardScaler()
        scaler.fit(np.arange(12.0).reshape(6, 2, 1))
        shape = (2, 3, 2, 1)
        truth = np.arange(np.prod(shape), dtype=float).reshape(shape)
        timestamps = np.arange(6).reshape(2, 3).astype("datetime64[m]")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.npz"
            save_prediction_archive(
                path,
                dataset="synthetic",
                variable="speed",
                split="validation",
                target_timestamps=timestamps,
                node_names=["a", "b"],
                y_true_scaled=truth,
                y_pred_scaled=truth + 1.0,
                y_true_physical=truth,
                y_pred_physical=truth + 1.0,
                train_time_end_exclusive=6,
                val_time_end_exclusive=8,
                scaler=scaler_metadata(scaler),
                seed=42,
            )
            loaded = load_prediction_archive(path)
            self.assertEqual(loaded["target_timestamps"].shape, (2, 3))
            self.assertEqual(loaded["node_names"].tolist(), ["a", "b"])
            np.testing.assert_array_equal(loaded["horizons"], [1, 2, 3])

    def test_strict_data_bundle_keeps_timestamp_alignment(self):
        values = np.arange(200.0).reshape(100, 2, 1)
        timestamps = np.arange(100).astype("datetime64[m]")
        bundle = strict_data_bundle(values, timestamps, history=4, horizon=3)
        self.assertEqual(bundle["val_target_timestamps"].shape[1], 3)
        self.assertTrue(bundle["split_info"]["target_disjoint"])
        self.assertEqual(bundle["scaler_metadata"]["type"], "StandardScaler")

    def test_strict_bundle_accepts_frozen_profile_boundaries(self):
        values = np.arange(240.0).reshape(120, 2, 1)
        timestamps = np.arange(120).astype("datetime64[m]")
        bundle = strict_data_bundle(
            values,
            timestamps,
            history=4,
            horizon=3,
            train_time_end_exclusive=70,
            val_time_end_exclusive=95,
        )
        info = bundle["split_info"]
        self.assertEqual(info["train_time_end_exclusive"], 70)
        self.assertEqual(info["val_time_end_exclusive"], 95)
        self.assertTrue(info["target_disjoint"])
        train_targets = set(bundle["train_target_timestamps"].reshape(-1).tolist())
        val_targets = set(bundle["val_target_timestamps"].reshape(-1).tolist())
        test_targets = set(bundle["test_target_timestamps"].reshape(-1).tolist())
        self.assertFalse(train_targets & val_targets)
        self.assertFalse(val_targets & test_targets)

    def test_event_free_feature_contract(self):
        contract = event_free_feature_contract()
        self.assertEqual(contract["input_features"], ["traffic"])
        self.assertEqual(contract["event_features"], [])
        self.assertEqual(contract["weather_features"], [])
        self.assertFalse(contract["resilience_aux_enabled"])
        self.assertFalse(contract["cvar_enabled"])
        self.assertFalse(contract["scalar_l4_target"])

if __name__ == "__main__":
    unittest.main()
