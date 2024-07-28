import unittest
import numpy as np

#imported calsses
from jak_test_ import  TestJak
from TETA_test_ import TestTETA
from invKIN_test_1 import TestInvKIN
from cables_test_1 import TestCablesFunction
from configr_test_ import TestConfigr
from Mass_matrix_test_ import TestMassMatrix
from controle_perp_test_ import TestControle
from RRode_computeForces_test_ import TestComputeForces
from RRode_test_ import TestRRode

if __name__ == '__main__':
    print("\n","FULL INTEGRATION TEST ... ")
    unittest.main()