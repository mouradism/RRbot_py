import unittest
import numpy as np
from invKIN import invKIN
from RRode import RRode

# Unit test class for RRode
class TestRRode(unittest.TestCase):
    def setUp(self):
        self.Args = {
            "t_": 0,
            "E_": 0,
            "TT": [],
            "FF": [],
            "g": 9.8,
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
            "XL": np.array([
                [-0.5, -0.5, 0.5, 0.5],
                [-1.0, 0.0, 0.0, -1.0]
            ]),
            "r": 0.1,
            "ivp": None,
            "Xi": np.array([-0.5, -0.4]),
            "Xf": None,
            "alpha": -0.0 * np.pi,
            "lj": None,
            "XL": None# <---------------None type non aceptable----------------
        }
        
        self.Args["XL"] = np.array([[self.Args["Lb"] / 2, -self.Args["Lb"] / 2, self.Args["Lb"] / 2, -self.Args["Lb"] / 2], [0, 0, 0, 0]])
        self.Args["Xf"] = self.Args["Xi"] + np.array([0.40, -0.40])
        
        self.Args["J1"] = (self.Args["m1"] * self.Args["l1"] ** 2) / 3
        self.Args["J2"] = (self.Args["m2"] * self.Args["l2"] ** 2) / 3
        self.Args["J3"] = (self.Args["m3"] * self.Args["l3"] ** 2) / 3

        # Inverse Kinematics
        self.q = invKIN(self.Args["Xi"] + self.Args["l3"] * np.array([np.cos(self.Args["alpha"]), np.sin(self.Args["alpha"])]).T, self.Args["Xi"],self.Args)
        # Initial conditions
        self.dq1 = 0
        self.dq2 = 0
        self.dq3 = 0
        self.Args["ivp"] = np.array([self.q[0], self.q[1], self.q[2], self.dq1, self.dq2, self.dq3]).T
        self.x = self.Args["ivp"]
        self.t = 0
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

    def test_RRode(self):
        print("\n","test_RRode ... ")
        expected_xdot = np.array([ 0.  ,   0.  ,   0.  , -31.85,   7.35,  -1.47])
        xdot = RRode(self.t, self.x, self.Args)
        np.testing.assert_almost_equal(xdot, expected_xdot, decimal=3)

if __name__ == '__main__':
    unittest.main()