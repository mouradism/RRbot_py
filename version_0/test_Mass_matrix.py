import unittest
import sys


import numpy as np
from Mass_matrix import Mass_matrix


class TestMLassMatrix(unittest.TestCase):

    def test_MLass_matrix(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        l1, l2, l3 = 0.5, 0.5, 0.3
        m1, m2, m3 = 6, 4, 1
        J1 =(m1*l1**2)/3
        J2 =(m2*l2**2)/3
        J3 =(m3*l3**2)/3
        ivp = [x, J1, J2, J3, m1, m2, m3, l1, l2, l3]
        result = Mass_matrix(ivp)

        expected_xMc = np.array([
            [4.75587106, 1.83000243, 0.18996893],
            [1.83000243, 1.02913381, 0.12415024],
            [0.18996893, 0.12415024, 0.0525 ]
        ])

        expected_xMs = np.block([
            [np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), expected_xMc]
        ])

        np.testing.assert_almost_equal(result, expected_xMs, decimal=6)

if __name__ == '__main__':
    unittest.main()