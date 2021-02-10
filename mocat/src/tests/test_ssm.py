########################################################################################################################
# Module: ssm/tests/test_ssm_linear_gaussian.py
# Description: Tests for linear TemporalGaussian state space models (and Kalman filtering/smoothing).
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import unittest

import jax.numpy as np
from jax import random
import numpy.testing as npt
from mocat.src.ssm.ssm import StateSpaceModel
from mocat.src.ssm.filters import BootstrapFilter, run_particle_filter_for_marginals
from mocat.src.ssm.backward import backward_simulation


class TestSSM(unittest.TestCase):
    ssm_scenario: StateSpaceModel
    len_t: int = 20
    t: np.ndarray = np.arange(len_t, dtype='float32')
    max_rejections: int = 2
    n: int = int(2e3)

    def _test_simulate(self):

        self.sim_samps = self.ssm_scenario.simulate(self.t, random.PRNGKey(0))
        npt.assert_array_equal(self.sim_samps.t, self.t)
        self.assertEqual(self.len_t, self.sim_samps.x.shape[0])
        self.assertEqual(self.len_t, self.sim_samps.y.shape[0])

    def _test_bootstrap(self):
        if not hasattr(self, 'sim_samps'):
            self.sim_samps = self.ssm_scenario.simulate(self.t, random.PRNGKey(0))
        self.pf_samps = run_particle_filter_for_marginals(self.ssm_scenario,
                                                          BootstrapFilter(),
                                                          self.sim_samps.y,
                                                          self.t,
                                                          random.PRNGKey(0),
                                                          n=self.n)

        npt.assert_array_less(self.sim_samps.x[:, 0], np.max(self.pf_samps.value, axis=1)[:, 0])
        npt.assert_array_less(np.min(self.pf_samps.value, axis=1)[:, 0], self.sim_samps.x[:, 0])

    def backward_preprocess(self):
        if not hasattr(self, 'sim_samps'):
            self.sim_samps = self.ssm_scenario.simulate(self.t, random.PRNGKey(0))
        if not hasattr(self, 'pf_samps'):
            self.pf_samps = run_particle_filter_for_marginals(self.ssm_scenario,
                                                              BootstrapFilter(),
                                                              self.sim_samps.y,
                                                              self.t,
                                                              random.PRNGKey(0),
                                                              n=int(1e4))

    def backward_postprocess(self):
        npt.assert_array_less(self.sim_samps.x[:, 0], np.max(self.backward_samps.value, axis=1)[:, 0])
        npt.assert_array_less(np.min(self.backward_samps.value, axis=1)[:, 0], self.sim_samps.x[:, 0])
        self.assertFalse(hasattr(self.backward_samps, 'log_weight'))

    def _test_ffbsi_full(self):
        self.backward_preprocess()
        self.backward_samps = backward_simulation(self.ssm_scenario,
                                                  self.pf_samps,
                                                  random.PRNGKey(0),
                                                  self.n,
                                                  maximum_rejections=0)
        self.backward_postprocess()

    def _test_ffbsi_rejection(self):
        self.backward_preprocess()
        self.backward_samps = backward_simulation(self.ssm_scenario,
                                                  self.pf_samps,
                                                  random.PRNGKey(0),
                                                  self.n,
                                                  maximum_rejections=self.max_rejections)
        self.backward_postprocess()

    def _test_backward(self):
        self._test_ffbsi_full()
        self._test_ffbsi_rejection()


if __name__ == '__main__':
    unittest.main()
