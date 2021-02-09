########################################################################################################################
# Module: ssm/tests/test_ssm_linear_gaussian.py
# Description: Tests for linear TemporalGaussian state space models (and Kalman filtering/smoothing).
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import unittest

import jax.numpy as np
from mocat.src.tests.test_ssm import TestSSM
from mocat.src.ssm.linear_gaussian.linear_gaussian import TimeHomogenousLinearGaussian


def create_eye_lgssm(dim: int):
    return TimeHomogenousLinearGaussian(initial_mean=np.zeros(dim),
                                        initial_covariance=np.eye(dim),
                                        transition_matrix=np.eye(dim),
                                        transition_covariance=np.eye(dim),
                                        likelihood_matrix=np.eye(dim),
                                        likelihood_covariance=np.eye(dim),
                                        name=f'{dim}D-eye LGSSM')


class Test1D(TestSSM):

    ssm_scenario = create_eye_lgssm(1)

    def test_simulate(self):
        super()._test_simulate()

    def test_bootstrap(self):
        super()._test_bootstrap()

    def test_backward(self):
        super()._test_backward_no_bound()


class Test5D(TestSSM):

    ssm_scenario = create_eye_lgssm(5)

    def test_simulate(self):
        super()._test_simulate()

    def test_bootstrap(self):
        super()._test_bootstrap()

    def test_backward(self):
        super()._test_backward_no_bound()


if __name__ == '__main__':
    unittest.main()
