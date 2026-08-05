import unittest

from run_a10_pems07_external_confirmation import PEMS07_MD5, PEMS07_SHAPE


class A10Pems07ExternalConfirmationTests(unittest.TestCase):
    def test_locked_dataset_contract(self):
        self.assertEqual(PEMS07_MD5, "978d3d9b85fe640a446983a34271a48d")
        self.assertEqual(PEMS07_SHAPE, (28224, 883, 1))


if __name__ == "__main__":
    unittest.main()
