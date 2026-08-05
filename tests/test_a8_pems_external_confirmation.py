import tempfile
import unittest
from pathlib import Path

import pandas as pd

from run_a8_pems_external_confirmation import parse_tasks, resolve_value_columns


class A8PemsExternalConfirmationTests(unittest.TestCase):
    def test_locked_task_parser(self):
        self.assertEqual(parse_tasks("pems04:flow"), [("pems04", "flow")])
        with self.assertRaises(ValueError):
            parse_tasks("pems04:occupancy")

    def test_value_columns_follow_csv_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.csv"
            pd.DataFrame(
                {
                    "Time": ["2018-01-01 00:00:00"],
                    "node_b_volume": [1.0],
                    "node_a_speed": [2.0],
                    "node_a_volume": [3.0],
                    "node_b_speed": [4.0],
                }
            ).to_csv(path, index=False)
            self.assertEqual(
                resolve_value_columns(path, "flow", 2),
                ["node_b_volume", "node_a_volume"],
            )


if __name__ == "__main__":
    unittest.main()
