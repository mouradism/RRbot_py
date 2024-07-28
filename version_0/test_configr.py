import unittest
import numpy as np
from configr import configr  # Replace 'your_module' with the actual module name

class TestConfigrFunction(unittest.TestCase):
    def test_valid_input(self):
        l1, l2, l3 = 0.5, 0.5, 0.3
        La, Lb = 2.0, 2.0
        expected_lj = np.array([1.3, 1.3, 1.,  1. ])
        expected_XL = np.array([
            [-Lb / 2, -Lb / 2, +Lb / 2, +Lb / 2],
            [-La, 0, 0, -La]
        ])
        
        lj, XL = configr(l1, l2, l3, La, Lb)

        np.testing.assert_array_almost_equal(lj, expected_lj)
        np.testing.assert_array_almost_equal(XL, expected_XL)

if __name__ == '__main__':
    unittest.main()
