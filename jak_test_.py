import unittest
import numpy as np
from jak import jak

class TestJak(unittest.TestCase):
    def test_jak(self):
        print("\n","test_jak ... ")
        l1, l2, l3 = 1.0, 1.0, 1.0
        q = np.array([np.pi/6, np.pi/4, np.pi/3])
        dq = np.array([0.1, 0.2, 0.3])
        
        expected_Jk = np.array([
            [-2.17303261, -1.67303261, -0.70710678],
            [ 0.41773767, -0.44828774 ,-0.70710678]
        ])
        
        expected_dt_Jk = np.array([
            [0.26001581 , 0.34661836,  0.42426407],
            [-0.76404182 ,-0.71404182, -0.42426405 ]
        ])
        
        expected_X = np.array([
            0.41773767 , 2.17303261
        ])

        Jk, dt_Jk, X = jak(l1, l2, l3, q, dq)

        np.testing.assert_almost_equal(Jk, expected_Jk, decimal=6)
        np.testing.assert_almost_equal(dt_Jk, expected_dt_Jk, decimal=6)
        np.testing.assert_almost_equal(X, expected_X, decimal=6)

if __name__ == '__main__':
    unittest.main()
