"""Regression tests for the read-only E-L4-2B-0 runner contract audit."""
from __future__ import annotations
import json
import unittest
from pathlib import Path
import pandas as pd

class RunnerContractAuditTests(unittest.TestCase):
    ROOT = Path(r"D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\e_l4_2b_0_contract")

    def test_b0_decision_is_read_only_and_blocks_training(self):
        decision = json.loads((self.ROOT / "e_l4_2b_0_decision.json").read_text(encoding="utf-8"))
        self.assertEqual(decision["stage"], "E-L4-2B-0")
        self.assertFalse(decision["training_run"])
        self.assertFalse(decision["optimizer_step_called"])
        self.assertFalse(decision["backward_called"])
        self.assertTrue(decision["stage_passed"])
        self.assertTrue(decision["e_l4_2b_training_authorized"])

    def test_b0_runner_wiring_is_repaired(self):
        decision = json.loads((self.ROOT / "e_l4_2b_0_decision.json").read_text(encoding="utf-8"))
        self.assertEqual(decision["blockers"], [])
        self.assertTrue(decision["runner_contract_passed"])
        self.assertTrue(decision["fairness_contract_passed"])
        self.assertTrue(decision["checkpoint_schema_passed"])
        self.assertTrue(decision["prediction_schema_passed"])

    def test_b0_forward_smoke_is_finite_without_updates(self):
        frame = pd.read_csv(self.ROOT / "forward_schema_audit.csv", encoding="utf-8-sig")
        self.assertTrue(frame["finite"].all())
        self.assertFalse(frame["optimizer_step_called"].any())
        self.assertFalse(frame["backward_called"].any())

    def test_b0_dual_space_data_contract_is_available(self):
        frame = pd.read_csv(self.ROOT / "dual_space_contract_audit.csv", encoding="utf-8-sig")
        self.assertTrue(frame["model_input_finite"].all())
        self.assertTrue((frame["model_input_nan_count"] == 0).all())
        self.assertTrue(frame["physical_model_arrays_independent"].all())
        self.assertEqual(int(frame.loc[(frame.dataset == "rainstorm") & (frame.variable == "speed"), "physical_nan_count"].iloc[0]), 578)
        self.assertEqual(int(frame.loc[(frame.dataset == "typhoon") & (frame.variable == "speed"), "physical_nan_count"].iloc[0]), 146)

    def test_b0_outputs_roundtrip(self):
        for path in self.ROOT.glob("*.csv"):
            frame = pd.read_csv(path, encoding="utf-8-sig")
            self.assertFalse(frame.empty, path.name)
        report = (self.ROOT / "e_l4_2b_0_contract_report.md").read_text(encoding="utf-8")
        self.assertIn("E-L4-2B-0", report)

    def test_b0_schema_contains_dual_space_masks(self):
        frame = pd.read_csv(self.ROOT / "checkpoint_npz_schema_audit.csv", encoding="utf-8-sig")
        self.assertTrue(frame["checkpoint_has_imputation"].all())
        self.assertTrue(frame["npz_has_physical_truth_valid_mask"].all())
        self.assertTrue(frame["npz_has_l4_valid_mask"].all())
        self.assertTrue(frame["truth_mask_shape_matches"].all())
        self.assertTrue(frame["l4_mask_shape_matches"].all())

    def test_b0_runner_sources_expose_explicit_dual_contract(self):
        frame = pd.read_csv(self.ROOT / "runner_contract_audit.csv", encoding="utf-8-sig")
        self.assertTrue(frame["dual_contract_available"].all())
        self.assertTrue(frame["event_weather_excluded_in_dual_protocol"].all())

if __name__ == "__main__":
    unittest.main()
