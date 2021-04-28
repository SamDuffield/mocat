########################################################################################################################
# Module: ssm/tests/test_ssm_linear_gaussian.py
# Description: Tests for linear TemporalGaussian state space models (and Kalman filtering/smoothing).
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import unittest

from jax import numpy as jnp, random, vmap
import numpy.testing as npt
from mocat.src.tests.test_ssm import TestSSM
from mocat.src.ssm.linear_gaussian.linear_gaussian import LinearGaussian, TimeHomogenousLinearGaussian
from mocat.src.ssm.linear_gaussian.kalman import run_kalman_filter_for_marginals, run_kalman_smoother_for_marginals


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

    def test_online_smoothing_pf(self):
        super()._test_online_smoothing_pf_full()
        super()._test_online_smoothing_pf_rejection()

    def test_online_smoothing_bs(self):
        super()._test_online_smoothing_bs_full()
        super()._test_online_smoothing_bs_rejection()


class Test5D(TestSSM):

    ssm_scenario = create_eye_lgssm(5)

    def test_simulate(self):
        super()._test_simulate()

    def test_bootstrap(self):
        super()._test_bootstrap()

    def test_backward(self):
        super()._test_backward()
    #
    # def test_online_smoothing_pf(self):
    #     super()._test_online_smoothing_pf_full()
    #     super()._test_online_smoothing_pf_rejection()
    #
    # def test_online_smoothing_bs(self):
    #     super()._test_online_smoothing_bs_full()
    #     super()._test_online_smoothing_bs_rejection()


class TestKalman(unittest.TestCase):
    ssm_scenario: LinearGaussian
    t = jnp.arange(20)

    def _test_kalman_filter(self, sds=3):
        if not hasattr(self, 'sim_samps'):
            self.sim_samps = self.ssm_scenario.simulate(self.t, random.PRNGKey(0))
        filter_means, filter_covs = run_kalman_filter_for_marginals(self.ssm_scenario,
                                                                    self.sim_samps.y,
                                                                    self.t)

        filter_sds = jnp.sqrt(vmap(jnp.diag)(filter_covs))
        npt.assert_array_less(self.sim_samps.x, filter_means + sds * filter_sds)
        npt.assert_array_less(filter_means - sds * filter_sds, self.sim_samps.x)

    def _test_kalman_smoother(self, sds=3):
        if not hasattr(self, 'sim_samps'):
            self.sim_samps = self.ssm_scenario.simulate(self.t, random.PRNGKey(0))
        smoother_means, smoother_covs = run_kalman_smoother_for_marginals(self.ssm_scenario,
                                                                          self.sim_samps.y,
                                                                          self.t)
        smoother_sds = jnp.sqrt(vmap(jnp.diag)(smoother_covs))
        npt.assert_array_less(self.sim_samps.x, smoother_means + sds * smoother_sds)
        npt.assert_array_less(smoother_means - sds * smoother_sds, self.sim_samps.x)


class TestKalman1D(TestKalman):
    ssm_scenario = create_eye_lgssm(1)

    def test_kalman_filter(self):
        super()._test_kalman_filter(3)

    def test_kalman_smoother(self):
        super()._test_kalman_smoother(3)


class TestKalman5D(TestKalman):
    ssm_scenario = create_eye_lgssm(5)

    def test_kalman_filter(self):
        super()._test_kalman_filter(4)

    def test_kalman_smoother(self):
        super()._test_kalman_smoother(4)


if __name__ == '__main__':
    unittest.main()
