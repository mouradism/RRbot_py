import unittest
from TETA import TETA
import numpy as np

class TestTETA(unittest.TestCase):
    
    def test_teta(self):
        print("\n","test_teta ... ")
        X = [1.0, 1.0]
        x = 4.0
        y = 5.0
        expected_teta = np.arctan2(y - X[1], x - X[0]) - np.pi
        result = TETA(x, y, X)
        self.assertAlmostEqual(result, expected_teta, places=6)

    def test_teta_zero(self):
        print("\n","test_teta_zero ... ")
        X = [0.0, 0.0]
        x = 0.0
        y = 0.0
        expected_teta = -np.pi
        result = TETA(x, y, X)
        self.assertAlmostEqual(result, expected_teta, places=6)

    def test_teta_negative(self):
        print("\n","test_teta_negative ... ")
        X = [1.0, 1.0]
        x = -1.0
        y = -1.0
        expected_teta = np.arctan2(y - X[1], x - X[0]) - np.pi
        result = TETA(x, y, X)
        self.assertAlmostEqual(result, expected_teta, places=6)

if __name__ == '__main__':
    unittest.main()
