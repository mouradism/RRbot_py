import unittest
import numpy as np
from Mass_matrix import Mass_matrix
from invKIN import invKIN

#ok
class TestMassMatrix(unittest.TestCase):
    def setUp(self):
        self.Args = {
            "J1": None,
            "J2": None,
            "J3": None,
            "m1": 6,
            "m2": 4,
            "m3": 1,
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

        # Calculate additional parameters
        self.Args["J1"] = (self.Args["m1"] * self.Args["l1"] ** 2) / 3
        self.Args["J2"] = (self.Args["m2"] * self.Args["l2"] ** 2) / 3
        self.Args["J3"] = (self.Args["m3"] * self.Args["l3"] ** 2) / 3

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

    def test_mass_matrix(self):
        print("\n","test_mass_matrix ... ")
        # Test input
        x = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        
        # Expected output (simplified)
        expected_xMs = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.75587106, 1.83000243, 0.18996893],
            [0.0, 0.0, 0.0, 1.83000243, 1.02913381, 0.12415024],
            [0.0, 0.0, 0.0, 0.18996893, 0.12415024, 0.0525]
        ])
        # Calculate mass matrix
        result_xMs,_ = Mass_matrix(x,self.Args)

        # Verify that the result is close to the expected value
        np.testing.assert_array_almost_equal(result_xMs, expected_xMs, decimal=5)

if __name__ == '__main__':
    unittest.main()