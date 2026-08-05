import unittest

import pandas as pd

from run_a9_pems03_external_confirmation import PEMS03_SHAPE, pems03_timestamps


class A9Pems03ExternalConfirmationTests(unittest.TestCase):
    def test_locked_calendar(self):
        timestamps = pems03_timestamps(PEMS03_SHAPE[0])
        self.assertEqual(pd.Timestamp(timestamps[0]), pd.Timestamp("2018-09-01 00:00:00"))
        self.assertEqual(pd.Timestamp(timestamps[-1]), pd.Timestamp("2018-11-30 23:55:00"))
        self.assertEqual(len(timestamps), 26208)


if __name__ == "__main__":
    unittest.main()
