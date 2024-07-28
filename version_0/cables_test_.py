import unittest
import numpy as np
from cables import cables
from invKIN import invKIN

class TestCables(unittest.TestCase):
    def test_cables(self):
        alpha = -0.0 * np.pi
        l1, l2, l3 = 0.5, 0.5, 0.3
        Lb, La = 2.0, 2.0
        Xi = np.array([-0.5, -0.40])
        Xf = Xi + np.array([0.40, -0.40])

        def initialize_lj_and_XL(l1, l2, l3, Lb, La):
            lj = np.array([
                [1, 1, 1],
                [0.5, 0, 0],
                [1, 1, 0],
                [1, 0.5, 0]
            ]) @ np.array([l1, l2, l3])
            
            if any(lj > l1 + l2 + l3):
                return
            
            XL = np.array([
                [+Lb/2, -Lb/2, +Lb/2, -Lb/2],
                [-0*La, 0, 0, -0*La]
            ])
            return lj, XL

        lj, XL = initialize_lj_and_XL(l1, l2, l3, Lb, La)

        if lj is None or XL is None:
            print("Invalid configuration")
        else:
            target = Xi + l3 * np.array([np.cos(alpha), np.sin(alpha)])
            q = invKIN(target, Xi,l1, l2)
          
        dq = np.array([0, 0, 0])
        
        
        # Known outputs for the test case
        expected_X = np.array([
            [-5.000000e-01, -4.000000e-01],
            [-2.500000e-01, -3.061617e-17],
            [-2.000000e-01, -4.000000e-01],
            [-3.500000e-01, -2.000000e-01],
        ])
        
        expected_Jk = np.array([
            [ 4.00000000e-01,  4.00000000e-01,  3.67394040e-17],
            [-5.00000000e-01, -5.55111512e-17, -3.00000000e-01],
            [ 3.06161700e-17,  0.00000000e+00, -0.00000000e+00],
            [-2.50000000e-01,  0.00000000e+00, -0.00000000e+00],
            [ 4.00000000e-01,  4.00000000e-01, -0.00000000e+00],
            [-2.00000000e-01,  3.00000000e-01, -0.00000000e+00],
            [ 2.00000000e-01,  2.00000000e-01, -0.00000000e+00],
            [-3.50000000e-01,  1.50000000e-01, -0.00000000e+00]
        ])
        
        expected_dt_Jk = np.array([
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0., -0.],
            [ 0.,  0., -0.],
            [ 0.,  0., -0.],
            [ 0.,  0., -0.],
            [ 0.,  0., -0.],
            [ 0.,  0., -0.]
        ])
        
        expected_S_ = np.array([
            [ 0.96623494  ,0.          ,0.          ,0.        ],
            [ 0.25766265  ,0.          ,0.          ,0.        ],
            [ 0.         ,-1.          ,0.          ,0.        ],
            [ 0.         ,-0.          ,0.          ,0.        ],
            [ 0.          ,0.          ,0.9486833   ,0.        ],
            [ 0.          ,0.          ,0.31622777  ,0.        ],
            [ 0.          ,0.          ,0.         ,-0.95577901],
            [ 0.          ,0.          ,0.          ,0.29408585]
        ])
        
        X, Jk, dt_Jk, S_ = cables(lj, XL, q, dq, l1, l2, l3)
 
        np.testing.assert_almost_equal(X, expected_X, decimal=6)
        np.testing.assert_almost_equal(Jk, expected_Jk, decimal=6)
        np.testing.assert_almost_equal(dt_Jk, expected_dt_Jk, decimal=6)
        np.testing.assert_almost_equal(S_, expected_S_, decimal=6)



if __name__ == '__main__':
    unittest.main()
