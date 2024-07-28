import unittest
import numpy as np

#the following functions are imported:
from cables import cables
from jak import jak
from TETA  import   TETA
from invKIN import invKIN
from configr import configr
from Mass_matrix import Mass_matrix
from controle_perp import  controle
from RRode import computeForces
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

class FullIntegration(unittest.TestCase):
    def test_integration(self):
        """TestJak class Integration test."""
        integration_TestJak= TestJak(methodName='test_jak')
        integration_TestJak.test_jak()
        """TestTETA class Integration test."""
        integration_TestTETA= TestTETA(methodName='test_teta')
        integration_TestTETA= TestTETA(methodName='test_teta_zero')
        integration_TestTETA= TestTETA(methodName='test_teta_negative')
        integration_TestTETA.test_teta()
        integration_TestTETA.test_teta_zero()
        integration_TestTETA.test_teta_negative()
        """TestInvKIN class Integration test."""
        integration_TestInvKIN= TestInvKIN(methodName='setUp')
        integration_TestInvKIN= TestInvKIN(methodName='test1_invKIN')
        integration_TestInvKIN= TestInvKIN(methodName='test2_invKIN')
        integration_TestInvKIN= TestInvKIN(methodName='test3_invKIN')
        integration_TestInvKIN.setUp()
        integration_TestInvKIN.test1_invKIN()
        integration_TestInvKIN.test2_invKIN()
        integration_TestInvKIN.test3_invKIN()
        """TestCablesFunction class Integration test."""
        integration_TestCablesFunction= TestCablesFunction(methodName='setUp')
        integration_TestCablesFunction= TestCablesFunction(methodName='test_cables_function')
        integration_TestCablesFunction= TestCablesFunction(methodName='test_invalid_lengths')
        integration_TestCablesFunction.setUp()
        integration_TestCablesFunction.test_cables_function()
        integration_TestCablesFunction.test_invalid_lengths()
        """TestConfigr class Integration test."""
        integration_TestConfigr= TestConfigr(methodName='setUp')
        integration_TestConfigr= TestConfigr(methodName='test_configr')
        integration_TestConfigr.setUp()
        integration_TestConfigr.test_configr()
        """TestMassMatrix class Integration test."""
        integration_TestMassMatrix= TestMassMatrix(methodName='setUp')
        integration_TestMassMatrix= TestMassMatrix(methodName='test_mass_matrix')
        integration_TestMassMatrix.setUp()
        integration_TestMassMatrix.test_mass_matrix()
        """TestControle class Integration test."""
        integration_TestControle= TestControle(methodName='setUp')
        integration_TestControle= TestControle(methodName='test_controle')
        integration_TestControle.setUp()
        integration_TestControle.test_controle()
        """TestComputeForces class Integration test."""
        integration_TestComputeForces= TestComputeForces(methodName='setUp')
        integration_TestComputeForces= TestComputeForces(methodName='test_computeForces')
        integration_TestComputeForces= TestComputeForces(methodName='test_computeForces_zero_velocities')
        integration_TestComputeForces.setUp()
        integration_TestComputeForces.test_computeForces()
        integration_TestComputeForces.test_computeForces_zero_velocities()
        """TestRRode class Integration test."""
        integration_TestRRode= TestRRode(methodName='setUp')
        integration_TestRRode= TestRRode(methodName='test_RRode')
        integration_TestRRode.setUp()
        integration_TestRRode.test_RRode()

if __name__ == '__main__':
    unittest.main()
