import tempfile
import unittest
from pathlib import Path

import pandas as pd

from run_a4_capacity_validation import TASKS
from run_a7_curriculum_validation import (
    linear_teacher_forcing_ratio,
    parse_tasks,
    validation_decision,
)


class A7CurriculumValidationTests(unittest.TestCase):
    def test_locked_linear_schedule(self):
        self.assertEqual(linear_teacher_forcing_ratio(0, 100), 1.0)
        self.assertAlmostEqual(linear_teacher_forcing_ratio(40, 100), 0.5)
        self.assertEqual(linear_teacher_forcing_ratio(80, 100), 0.0)
        self.assertEqual(linear_teacher_forcing_ratio(99, 100), 0.0)

    def test_task_parser_rejects_unknown_task(self):
        self.assertEqual(parse_tasks("bridge:flow"), [("bridge", "flow")])
        with self.assertRaises(ValueError):
            parse_tasks("bridge:speed")

    def test_full_validation_gate(self):
        rows = []
        controls = []
        for seed in (42, 2024, 3407):
            for dataset, variable in TASKS:
                rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "variable": variable,
                        "best_val_mae": 8.5,
                        "finite": True,
                    }
                )
                controls.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "variable": variable,
                        "m1_val_mae": 10.0,
                        "a6_val_mae": 9.0,
                    }
                )
        with tempfile.TemporaryDirectory() as temporary_directory:
            control_path = Path(temporary_directory) / "controls.csv"
            pd.DataFrame(controls).to_csv(control_path, index=False)
            comparisons, decision = validation_decision(pd.DataFrame(rows), control_path)
        self.assertEqual(len(comparisons), 15)
        self.assertTrue(decision["full_protocol"])
        self.assertTrue(decision["accepted"])
        self.assertTrue(decision["external_confirmation_authorized"])
        self.assertFalse(decision["outer_evaluation_authorized"])


if __name__ == "__main__":
    unittest.main()
