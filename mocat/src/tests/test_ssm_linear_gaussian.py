########################################################################################################################
# Module: ssm/tests/test_ssm_linear_gaussian.py
# Description: Tests for linear TemporalGaussian state space models (and Kalman filtering/smoothing).
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import unittest

import jax.numpy as jnp
from mocat.src.tests.test_ssm import TestSSM
from mocat.src.ssm.linear_gaussian.linear_gaussian import TimeHomogenousLinearGaussian


def create_eye_lgssm(dim: int):
    return TimeHomogenousLinearGaussian(initial_mean=jnp.zeros(dim),
                                        initial_covariance=jnp.eye(dim),
                                        transition_matrix=jnp.eye(dim),
                                        transition_covariance=jnp.eye(dim),
                                        likelihood_matrix=jnp.eye(dim),
                                        likelihood_covariance=jnp.eye(dim),
                                        name=f'{dim}D-eye LGSSM')


class Test1D(TestSSM):

    ssm_scenario = create_eye_lgssm(1)

    def test_simulate(self):
        super()._test_simulate()

    def test_bootstrap(self):
        super()._test_bootstrap()

    def test_backward(self):
        super()._test_backward()


class Test5D(TestSSM):

    ssm_scenario = create_eye_lgssm(5)

    def test_simulate(self):
        super()._test_simulate()

    def test_bootstrap(self):
        super()._test_bootstrap()

    def test_backward(self):
        super()._test_backward()


if __name__ == '__main__':
    unittest.main()
