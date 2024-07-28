import unittest
import numpy as np

from RRode import computeForces
# Define the unit test class
class TestComputeForces(unittest.TestCase):
    def setUp(self):
        self.Args = {
            "g": 9.8,
            "m1": 6,
            "m2": 4,
            "m3": 1,
            "l1": 0.5,
            "l2": 0.5,
            "l3": 0.3
        }

    def test_computeForces(self):
        print("\n","test_computeForces ... ")
        q = np.array([0.1, 0.2, 0.3])
        dq = np.array([0.1, 0.2, 0.3])
        expected_vFc = np.array([54.230364, 15.252555,  1.215598])
        
        vFc = computeForces(q, dq, self.Args)
        np.testing.assert_almost_equal(vFc, expected_vFc, decimal=6)

    def test_computeForces_zero_velocities(self):
        print("\n","test_computeForces_zero_velocities ... ")
        q = np.array([0.1, 0.2, 0.3])
        dq = np.zeros(3)
        expected_vFc = np.array([54.26085302, 15.25668974 , 1.21324335])
        
        vFc = computeForces(q, dq, self.Args)
        np.testing.assert_almost_equal(vFc, expected_vFc, decimal=6)

if __name__ == '__main__':
    unittest.main()