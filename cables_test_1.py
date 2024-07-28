import unittest
import numpy as np

#the following functions are imported:
from cables import cables
from jak import jak
from TETA  import   TETA
from invKIN import invKIN

class TestCablesFunction(unittest.TestCase):
    
    def setUp(self):
        # Setting up common variables used in the tests
        self.Args = {
            "l1": 0.5,
            "l2": 0.5,
            "l3": 0.3,
            "La": 2.0,
            "Lb": 2.0,
            "lj": None,
            "Xi": np.array([-0.5, -0.40]),
            "Xf": None,
            "XL": None,
            "alpha": -0.0 * np.pi
        }

        self.Args["Xf"] = self.Args["Xi"] + np.array([0.40, -0.40])

        self.Args["lj"] = np.array([
                [1, 1, 1],
                [0.5, 0, 0],
                [1, 1, 0],
                [1, 0.5, 0]
            ]) @ np.array([self.Args["l1"], self.Args["l2"], self.Args["l3"]])
        
        self.Args["XL"] = np.array([
                [+self.Args["Lb"]/2, -self.Args["Lb"]/2, +self.Args["Lb"]/2, -self.Args["Lb"]/2],
                [-0*self.Args["La"], 0, 0, -0*self.Args["La"]]
            ])

        if self.Args["lj"] is None or self.Args["XL"] is None:
            print("Invalid configuration")
        else:
            target = self.Args["Xi"] + self.Args["l3"] * np.array([np.cos(self.Args["alpha"]), np.sin(self.Args["alpha"])])
            self.q = invKIN(target, self.Args["Xi"],self.Args)
          
        self.dq = np.array([0, 0, 0])

        #expected values for X, Jk, dt_Jk, S_ (these need to be calculated accurately)
        self.expected_X = np.array([[-5.000000e-01, -4.000000e-01, -2.500000e-01, -3.061617e-17,
                                    -2.000000e-01, -4.000000e-01, -3.500000e-01, -2.000000e-01]])
        self.expected_Jk = np.array([
            [ 4.00000000e-01,  4.00000000e-01,  3.67394040e-17],
            [-5.00000000e-01, -5.55111512e-17, -3.00000000e-01],
            [ 3.06161700e-17,  0.00000000e+00, -0.00000000e+00],
            [-2.50000000e-01,  0.00000000e+00, -0.00000000e+00],
            [ 4.00000000e-01,  4.00000000e-01, -0.00000000e+00],
            [-2.00000000e-01,  3.00000000e-01, -0.00000000e+00],
            [ 2.00000000e-01,  2.00000000e-01, -0.00000000e+00],
            [-3.50000000e-01,  1.50000000e-01, -0.00000000e+00]
        ])
        self.expected_dt_Jk = np.array([
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0., -0.],
            [ 0.,  0., -0.],
            [ 0.,  0., -0.],
            [ 0.,  0., -0.],
            [ 0.,  0., -0.],
            [ 0.,  0., -0.]
        ])
        self.expected_S_ = np.array([
            [ 0.96623494  ,0.          ,0.          ,0.        ],
            [ 0.25766265  ,0.          ,0.          ,0.        ],
            [ 0.         ,-1.          ,0.          ,0.        ],
            [ 0.         ,-0.          ,0.          ,0.        ],
            [ 0.          ,0.          ,0.9486833   ,0.        ],
            [ 0.          ,0.          ,0.31622777  ,0.        ],
            [ 0.          ,0.          ,0.         ,-0.95577901],
            [ 0.          ,0.          ,0.          ,0.29408585]
        ])
    
    def test_cables_function(self):
        print("\n","test_cables_function ... ")
        X, Jk, dt_Jk, S_ = cables(self.q, self.dq, self.Args)
        # Testing shapes of the outputs
        self.assertEqual(X.shape, (1, 8), "X should be a 4x2 matrix")
        self.assertEqual(Jk.shape, (8, 3), "Jk should be an 8x3 matrix")
        self.assertEqual(dt_Jk.shape, (8, 3), "dt_Jk should be an 8x3 matrix")
        self.assertEqual(S_.shape, (8, 4), "S_ should be a 2x4 matrix")
        
        # Additional checks can be added based on expected values
        # For example:
        # np.testing.assert_allclose(X[0], expected_X1, rtol=1e-5, atol=1e-8)
        # np.testing.assert_allclose(Jk[0], expected_Jk1, rtol=1e-5, atol=1e-8)
        # np.testing.assert_allclose(dt_Jk[0], expected_dt_Jk1, rtol=1e-5, atol=1e-8)
        #np.testing.assert_allclose(S_[:, 0], expected_S1, rtol=1e-5, atol=1e-8)

        # Testing values of the outputs
        
        np.testing.assert_allclose(X, self.expected_X, rtol=1e-5, atol=1e-8, err_msg="X values do not match expected values")
        np.testing.assert_allclose(Jk, self.expected_Jk, rtol=1e-5, atol=1e-8, err_msg="Jk values do not match expected values")
        np.testing.assert_allclose(dt_Jk, self.expected_dt_Jk, rtol=1e-5, atol=1e-8, err_msg="dt_Jk values do not match expected values")
        np.testing.assert_allclose(S_, self.expected_S_, rtol=1e-5, atol=1e-8, err_msg="S_ values do not match expected values")
    
    def test_invalid_lengths(self):
        print("\n","test_invalid_lengths ... ")
        invalid_args = {
            "l1": 2.0,
            "l2": 1.5,
            "l3": 1.0,
            "lj": [3.0, 3.5, 5.5, 4.0],  # One length is too long
            "XL": np.array([[2.5, 3.0, 1.5, 4.0], [1.5, 2.0, 2.5, 1.0]])
        }
    
        with self.assertRaises(ValueError) as context:
            cables(self.q, self.dq, invalid_args)
        
        self.assertTrue('One or more lengths exceed the total combined length.' in str(context.exception))
    
if __name__ == '__main__':
    unittest.main()
