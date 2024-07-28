import unittest
import numpy as np
from configr import configr

class TestConfigr(unittest.TestCase):
    def setUp(self):
        self.Args = {
            "l1": 0.5,
            "l2": 0.5,
            "l3": 0.3,
            "La": 2.0,
            "Lb": 2.0
        }

    def test_configr(self):
        print("\n","test_configr ... ")
        expected_lj = np.array([1.3, 1.3, 1.0, 1.0])
        expected_XL = np.array([
            [-1.0, -1.0, 1.0, 1.0],
            [-2.0, 0.0, 0.0, -2.0]
        ])
        lj, XL = configr(self.Args)
        np.testing.assert_almost_equal(lj, expected_lj, decimal=6)
        np.testing.assert_almost_equal(XL, expected_XL, decimal=6)
    """
        def test_configr_invalid_lengths(self):
        Args_invalid = {
            "l1": 0.5,
            "l2": 0.5,
            "l3": 1.5,
            "La": 2.0,
            "Lb": 2.0
        }
        lj, XL = configr(Args_invalid)
        print('---2>',lj, XL)
        self.assertIsNone(lj)
        self.assertIsNone(XL)
    """

if __name__ == '__main__':
    unittest.main()
