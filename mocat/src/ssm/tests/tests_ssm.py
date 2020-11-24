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
from mocat.src.ssm.ssm import StateSpaceModel
from mocat.src.ssm.filters import BootstrapFilter, run_particle_filter_for_marginals
from mocat.src.ssm.backward import backward_simulation


class TestSSM(unittest.TestCase):
    ssm_scenario: StateSpaceModel
    len_t: int = 20
    t: np.ndarray = np.arange(len_t, dtype='float32')
    max_rejections: int = 2

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
                                                          n=int(1e4))

        npt.assert_array_less(self.sim_samps.x, np.max(self.pf_samps.value, axis=1))
        npt.assert_array_less(np.min(self.pf_samps.value, axis=1), self.sim_samps.x)

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
        npt.assert_array_less(self.sim_samps.x, np.max(self.backward_samps.value, axis=1))
        npt.assert_array_less(np.min(self.backward_samps.value, axis=1), self.sim_samps.x)
        self.assertFalse(hasattr(self.backward_samps, 'log_weights'))

    def _test_ffbsi_full_no_bound(self):
        self.backward_preprocess()
        self.backward_samps = backward_simulation(self.ssm_scenario,
                                                  self.pf_samps,
                                                  0,
                                                  0.,
                                                  random.PRNGKey(0))
        self.backward_postprocess()

    def _test_ffbsi_rejection_no_bound(self):
        self.backward_preprocess()
        self.backward_samps = backward_simulation(self.ssm_scenario,
                                                  self.pf_samps,
                                                  self.max_rejections,
                                                  0.,
                                                  random.PRNGKey(0))
        self.backward_postprocess()

    def _test_ffbsi_full_bound(self, bound: float):
        self.backward_preprocess()
        self.backward_samps = backward_simulation(self.ssm_scenario,
                                                  self.pf_samps,
                                                  0,
                                                  bound,
                                                  random.PRNGKey(0))
        self.backward_postprocess()

    def _test_ffbsi_rejection_bound(self, bound: float):
        self.backward_preprocess()
        self.backward_samps = backward_simulation(self.ssm_scenario,
                                                  self.pf_samps,
                                                  self.max_rejections,
                                                  bound,
                                                  random.PRNGKey(0))
        self.backward_postprocess()

    def _test_backward_no_bound(self):
        self._test_ffbsi_full_no_bound()
        self._test_ffbsi_rejection_no_bound()

    def _test_backward_bound(self, bound: float):
        self._test_ffbsi_full_bound(bound)
        self._test_ffbsi_rejection_bound(bound)


if __name__ == '__main__':
    unittest.main()
