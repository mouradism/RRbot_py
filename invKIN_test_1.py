import unittest
import numpy as np
from invKIN import invKIN

class TestInvKIN(unittest.TestCase):
    def setUp(self):
        self.Args = {
            "l1": 1.0,
            "l2": 1.0
        }

    def test1_invKIN(self):
        print("\n","test1_invKIN ... ")
        X2 = np.array([1.0, 1.0])
        X3 = np.array([2.0, 0.0])
        expected_q = np.array([ 0.,          1.57079633, -2.35619449])
        result_q = invKIN(X2, X3, self.Args)
        print(result_q)
        np.testing.assert_almost_equal(result_q, expected_q, decimal=6)

    def test2_invKIN(self):
        print("\n","test2_invKIN ... ")
        # Test case 1: Nominal case
        X2 = [1.0, 1.0]
        X3 = [2.0, 2.0]
        Args = {"l1": 1.0, "l2": 1.0}
        expected_q = np.array([0.      ,  1.570796, -0.785398])
        q = invKIN(X2, X3, Args)
        np.testing.assert_allclose(q, expected_q, rtol=1e-6)
        
        # Test case 2: Degenerate case (X2 = X3)
        X2 = [1.0, 1.0]
        X3 = [1.0, 1.0]
        Args = {"l1": 1.0, "l2": 1.0}
        expected_q = np.array([0.,         1.57079633, 1.57079633])
        q = invKIN(X2, X3, Args)
        np.testing.assert_allclose(q, expected_q, rtol=1e-6)

    def test3_invKIN(self):
        print("\n","test3_invKIN ... ")
        X2 = np.array([1.5, 0.5])
        X3 = np.array([2.0, 1.0])
        expected_q = np.array([-0.33730748,  1.31811607, -0.19541043])
        result_q = invKIN(X2, X3, self.Args)
        print(result_q)
        np.testing.assert_almost_equal(result_q, expected_q, decimal=6)

    

if __name__ == '__main__':
    unittest.main()
