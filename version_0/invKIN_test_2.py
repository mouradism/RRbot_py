import unittest
import numpy as np
from invKIN import invKIN

class TestInvKIN(unittest.TestCase):
    def test_invKIN(self):
    
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
        
        """
        # Test case 3: Invalid input (l1 + l2 < |X2|)
        X2 = [2.0, 2.0]
        X3 = [3.0, 3.0]
        Args = {"l1": 1.0, "l2": 1.0}
        with self.assertRaises(ValueError):
            q = invKIN(X2, X3, Args)
        """
     

if __name__ == '__main__':
    unittest.main()