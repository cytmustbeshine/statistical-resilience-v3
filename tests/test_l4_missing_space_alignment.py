"""Tests for the E-L4-2A-R3 dual-space missing-data contract."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_final_l4_a3_alignment import (
    EVENT_WINDOWS,
    MODEL_SHA256_AT_START,
    TRAIN_SHA256_AT_START,
    canonical_l4_threshold_map,
    file_sha256,
    partial_output_inventory,
)
from audit_l4_missing_space_alignment import profile_reference, timestamp_keys
from audit_l4_prediction_pipeline import DATASETS, ordered_frame
from data import split_traffic_window_indices_strict
from l4_prediction_pipeline import (
    event_free_feature_contract,
    fit_train_only_scaler,
    impute_model_inputs_train_only,
    load_ordered_univariate_series_physical,
    paired_column_plan,
    profile_compatible_event_external_split,
)


class L4MissingSpaceAlignmentTests(unittest.TestCase):
    L4_DIR = Path(r"D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4")
    L3_DIR = Path(r"D:\TrafficGNN\outputs\latent_traffic_performance_l3")

    def test_physical_loader_preserves_rainstorm_speed_missing(self):
        config = DATASETS["rainstorm"]
        frame = ordered_frame(config)
        plan = paired_column_plan(list(frame.columns), "_volume", "_speed", 41)
        values, _, names = load_ordered_univariate_series_physical(
            str(config["csv"]), str(config["time_col"]), plan["paired_speed_columns"], "_speed"
        )
        self.assertEqual(int(np.isnan(values).sum()), 578)
        self.assertEqual(names, plan["paired_node_names"])

    def test_physical_loader_preserves_typhoon_speed_missing(self):
        config = DATASETS["typhoon"]
        frame = ordered_frame(config)
        plan = paired_column_plan(list(frame.columns), "_volume", "_speed", 41)
        values, _, names = load_ordered_univariate_series_physical(
            str(config["csv"]), str(config["time_col"]), plan["paired_speed_columns"], "_speed"
        )
        self.assertEqual(values.shape[1], 16)
        self.assertEqual(int(np.isnan(values).sum()), 146)
        self.assertEqual(names, plan["paired_node_names"])

    def test_train_only_imputation_produces_finite_copy(self):
        physical = np.array([1.0, np.nan, 3.0, np.nan, 100.0])[:, None, None]
        before = physical.copy()
        model, metadata = impute_model_inputs_train_only(physical, 3)
        self.assertTrue(np.isfinite(model).all())
        self.assertEqual(metadata["node_medians"], [2.0])
        np.testing.assert_array_equal(np.isnan(physical), np.isnan(before))

    def test_imputation_parameters_ignore_post_train_values(self):
        physical = np.array([1.0, np.nan, 3.0, 10.0, np.nan])[:, None, None]
        changed = physical.copy()
        changed[3, 0, 0] = 1000000.0
        _, left = impute_model_inputs_train_only(physical, 3)
        _, right = impute_model_inputs_train_only(changed, 3)
        self.assertEqual(left["node_medians"], right["node_medians"])
        self.assertEqual(left["global_median"], right["global_median"])

    def test_imputation_rejects_all_missing_training_prefix(self):
        with self.assertRaises(ValueError):
            impute_model_inputs_train_only(np.full((5, 2, 1), np.nan), 3)

    def test_scaler_uses_imputed_training_prefix_only(self):
        physical = np.array([1.0, np.nan, 3.0, 10.0, np.nan])[:, None, None]
        changed = physical.copy()
        changed[3, 0, 0] = 1000000.0
        left_model, _ = impute_model_inputs_train_only(physical, 3)
        right_model, _ = impute_model_inputs_train_only(changed, 3)
        left, _ = fit_train_only_scaler(left_model, 3)
        right, _ = fit_train_only_scaler(right_model, 3)
        np.testing.assert_array_equal(left.mean, right.mean)
        np.testing.assert_array_equal(left.std, right.std)

    def test_raw_physical_values_reproduce_all_canonical_thresholds(self):
        canonical = canonical_l4_threshold_map(self.L4_DIR)
        for dataset in ("bridge", "rainstorm", "typhoon"):
            config = DATASETS[dataset]
            frame = ordered_frame(config)
            plan = paired_column_plan(list(frame.columns), config["flow_suffix"], config["speed_suffix"], 41)
            specs = [("flow", "demand", plan["selected_flow_columns"])]
            if dataset != "bridge":
                specs.append(("speed", "efficiency", plan["paired_speed_columns"]))
            for variable, dimension, columns in specs:
                values, timestamps, _ = load_ordered_univariate_series_physical(
                    str(config["csv"]), str(config["time_col"]), columns, config[f"{variable}_suffix"]
                )
                quantiles, _, _ = profile_reference(
                    dataset, dimension, values, timestamps, self.L4_DIR, self.L3_DIR
                )
                self.assertAlmostEqual(quantiles["q90"], canonical[(dataset, dimension)]["q90"], places=12)
                self.assertAlmostEqual(quantiles["q99"], canonical[(dataset, dimension)]["q99"], places=12)

    def test_missing_truth_remains_invalid_for_l4_metrics(self):
        physical = np.array([1.0, np.nan, 2.0])[:, None, None]
        model, _ = impute_model_inputs_train_only(physical, 2)
        self.assertTrue(np.isnan(physical[1, 0, 0]))
        self.assertTrue(np.isfinite(model[1, 0, 0]))
        valid = np.isfinite(physical)
        self.assertEqual(int(valid.sum()), 2)

    def test_profile_compatible_split_keeps_events_external(self):
        lengths = {"bridge": 5184, "rainstorm": 12096, "typhoon": 4320}
        for dataset, length in lengths.items():
            split = profile_compatible_event_external_split(dataset)
            train, val, test, info = split_traffic_window_indices_strict(
                length, 12, 12,
                train_time_end_exclusive=split["train_time_end_exclusive"],
                val_time_end_exclusive=split["val_time_end_exclusive"],
            )
            timestamps = np.arange(length).astype("datetime64[m]")
            train_keys = timestamp_keys(timestamps, train, 12, 12)
            val_keys = timestamp_keys(timestamps, val, 12, 12)
            test_keys = timestamp_keys(timestamps, test, 12, 12)
            self.assertTrue(info["target_disjoint"])
            for start, end in EVENT_WINDOWS[dataset]:
                event = set(int(v) for v in timestamps[start:end + 1].astype("datetime64[ns]").astype("int64"))
                self.assertFalse(event & train_keys)
                self.assertFalse(event & val_keys)
                self.assertTrue(event.issubset(test_keys))

    def test_feature_contract_excludes_event_and_weather(self):
        contract = event_free_feature_contract()
        self.assertEqual(contract["input_features"], ["traffic"])
        self.assertEqual(contract["event_features"], [])
        self.assertEqual(contract["weather_features"], [])

    def test_model_and_train_remain_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(file_sha256(root / "model.py"), MODEL_SHA256_AT_START)
        self.assertEqual(file_sha256(root / "train.py"), TRAIN_SHA256_AT_START)

    def test_partial_formal_outputs_remain_unchanged(self):
        baseline = json.loads(
            Path(r"D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\e_l4_2a_r2\_partial_integrity_before.json").read_text(encoding="utf-8-sig")
        )
        current = partial_output_inventory(Path(baseline["root"]))
        self.assertEqual((current["result_count"], current["file_count"], current["total_bytes"]), (24, 120, 492153111))
        self.assertEqual(
            {row["relative_path"]: row["sha256"] for row in current["result_files"]},
            {row["relative_path"]: row["sha256"] for row in baseline["result_files"]},
        )

    def test_r3_decision_json_roundtrip(self):
        payload = {"stage": "E-L4-2A-R3", "training_run": False, "stage_passed": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)


if __name__ == "__main__":
    unittest.main()