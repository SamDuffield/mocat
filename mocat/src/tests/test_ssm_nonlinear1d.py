########################################################################################################################
# Module: ssm/tests/test_ssm_linear_gaussian.py
# Description: Tests for 1D non-linear benchmark model.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


import unittest

from mocat.src.tests.test_ssm import TestSSM
from mocat.src.ssm.scenarios.nonlinear1d import NonLinear1DBenchmark


class Test1DBootstrap(TestSSM):

    ssm_scenario = NonLinear1DBenchmark()

    def test_simulate(self):
        super()._test_simulate()

    def test_bootstrap(self):
        super()._test_bootstrap()

    def test_backward(self):
        super()._test_backward()


if __name__ == '__main__':
    unittest.main()
