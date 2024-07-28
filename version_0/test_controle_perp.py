import unittest
import numpy as np
from scipy.linalg import pinv, null_space
from jak import jak
from invKIN import invKIN
from v_0.cables_v0 import cables
from controle_perp import *

class TestFunctions(unittest.TestCase):
    def test_controle(self):
        # Set up test inputs
        t = 1.0
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        args = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, [10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0])

        # Call the function and assert outputs
        Tau, F = controle(t, x, args)
        self.assertIsInstance(Tau, np.ndarray)
        self.assertIsInstance(F, float)

    def test_dyn_0(self):
        # Set up test inputs
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        args = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)

        # Call the function and assert outputs
        xMc, vFc = dyn_0(x, args)
        self.assertIsInstance(xMc, np.ndarray)
        self.assertIsInstance(vFc, np.ndarray)

    def test_planning(self):
        # Set up test inputs
        t = 1.0
        lj = np.array([10.0, 11.0, 12.0, 13.0])
        q = np.array([0.1, 0.2, 0.3])
        dq = np.array([0.4, 0.5, 0.6])
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        Dt = 0.5
        args = (1.0, 2.0, 3.0, 4.0, 0.5, np.array([10.0, 10.0]), np.array([20.0, 20.0]))

        # Call the function and assert outputs
        Xd, V, A = planning(t, lj, q, dq, X, Dt, args)
        self.assertIsInstance(Xd, np.ndarray)
        self.assertIsInstance(V, np.ndarray)
        self.assertIsInstance(A, np.ndarray)

    def test_calculateSegments(self):
        # Set up test inputs
        lj = 5.0
        l1 = 2.0
        l2 = 3.0
        l3 = 4.0

        # Call the function and assert outputs
        l1_seg, l2_seg, l3_seg = calculateSegments(lj, l1, l2, l3)
        self.assertIsInstance(l1_seg, float)
        self.assertIsInstance(l2_seg, float)
        self.assertIsInstance(l3_seg, float)

    def test_computeTrajectoryCoefficients(self):
        # Set up test inputs
        Xi = np.array([1.0, 2.0])
        Xf = np.array([3.0, 4.0])
        Vi = np.array([0.1, 0.2])
        Vf = np.array([0.3, 0.4])
        tf = 1.0

        # Call the function and assert outputs
        a, b, c, d = computeTrajectoryCoefficients(Xi, Xf, Vi, Vf, tf)
        self.assertIsInstance(a, np.ndarray)
        self.assertIsInstance(b, np.ndarray)
        self.assertIsInstance(c, np.ndarray)
        self.assertIsInstance(d, np.ndarray)

    def test_evaluateTrajectory(self):
        # Set up test inputs
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        c = np.array([0.1, 0.2])
        d = np.array([0.3, 0.4])
        t = 0.5

        # Call the function and assert outputs
        Xc, Vc, Ac = evaluateTrajectory(a, b, c, d, t)
        self.assertIsInstance(Xc, np.ndarray)
        self.assertIsInstance(Vc, np.ndarray)
        self.assertIsInstance(Ac, np.ndarray)

    def test_computeJointVelocities(self):
        # Set up test inputs
        l1 = 1.0
        l2 = 2.0
        l3 = 3.0
        qc = np.array([0.1, 0.2, 0.3])
        Vc = np.array([0.4, 0.5])

        # Call the function and assert outputs
        dqc = computeJointVelocities(l1, l2, l3, qc, Vc)
        self.assertIsInstance(dqc, np.ndarray)

    def test_computeJointAccelerations(self):
        # Set up test inputs
        l1 = 1.0
        l2 = 2.0
        l3 = 3.0
        qc = np.array([0.1, 0.2, 0.3])
        dqc = np.array([0.4, 0.5, 0.6])
        Ac = np.array([0.7, 0.8])

        # Call the function and assert outputs
        ddqc = computeJointAccelerations(l1, l2, l3, qc, dqc, Ac)
        self.assertIsInstance(ddqc, np.ndarray)

    def test_computeJacobians(self):
        # Set up test inputs
        l11 = 1.0
        l12 = 2.0
        l13 = 3.0
        l21 = 4.0
        l22 = 5.0
        l23 = 6.0
        l31 = 7.0
        l32 = 8.0
        l33 = 9.0
        l41 = 10.0
        l42 = 11.0
        l43 = 12.0
        qc = np.array([0.1, 0.2, 0.3])
        dqc = np.array([0.4, 0.5, 0.6])

        # Call the function and assert outputs
        Jk, dt_Jk, Xd = computeJacobians(l11, l12, l13, l21, l22, l23, l31, l32, l33, l41, l42, l43, qc, dqc)
        self.assertIsInstance(Jk, np.ndarray)
        self.assertIsInstance(dt_Jk, np.ndarray)
        self.assertIsInstance(Xd, np.ndarray)

if __name__ == '__main__':
    unittest.main()