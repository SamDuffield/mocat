########################################################################################################################
# Module: ssm/tests/test_linear_gaussian.py
# Description: Tests for linear TemporalGaussian state space models (and Kalman filtering/smoothing).
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


import unittest

import jax.numpy as np
from jax import random
import numpy.testing as npt
from mocat.src.ssm.linear_gaussian.linear_gaussian import TimeHomogenousLinearGaussian
from mocat.src.ssm.filters import BootstrapFilter, run_particle_filter_for_marginals


def create_eye_lgssm(dim: int):
    return TimeHomogenousLinearGaussian(initial_mean=np.zeros(dim),
                                        initial_covariance=np.ones(dim),
                                        transition_matrix=np.ones(dim),
                                        transition_covariance=np.ones(dim),
                                        likelihood_matrix=np.ones(dim),
                                        likelihood_covariance=np.ones(dim),
                                        name=f'{dim}D-eye LGSSM')


class Test1D(unittest.TestCase):

    def test_potentials(self):
        lgssm = create_eye_lgssm(1)
        x = np.arange(5, dtype='float32').reshape(5, 1)
        t = np.arange(5, dtype='float32')
        y = np.ones((5, 1))

        npt.assert_array_equal(lgssm.smoothing_potential(x, y, t), 9.5)
        lgssm.transition_matrix = np.array([5.])
        npt.assert_array_equal(lgssm.smoothing_potential(x, y, t), 97.5)

    def test_simulate(self):
        lgssm = create_eye_lgssm(1)
        len_t = 20
        t = np.arange(len_t)
        sim_samps = lgssm.simulate(t, random.PRNGKey(0))
        npt.assert_array_equal(sim_samps.t, t)
        npt.assert_array_almost_equal(sim_samps.x, np.zeros((len_t, 1)), decimal=-1)
        npt.assert_array_almost_equal(sim_samps.y, np.zeros((len_t, 1)), decimal=-1)


class Test1DBootstrap(unittest.TestCase):
    lgssm = create_eye_lgssm(1)
    t = np.arange(20)
    sims = lgssm.simulate(t, random.PRNGKey(0))

    bootstrap_filter = BootstrapFilter()
    n = 1000

    def test_run_for_margs(self):
        pf_samps = run_particle_filter_for_marginals(self.lgssm,
                                                     self.bootstrap_filter,
                                                     self.sims.y,
                                                     self.t,
                                                     random.PRNGKey(0),
                                                     n=self.n)

        npt.assert_array_less(self.sims.x, np.max(pf_samps.value, axis=1))
        npt.assert_array_less(np.min(pf_samps.value, axis=1), self.sims.x)


if __name__ == '__main__':
    unittest.main()
