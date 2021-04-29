########################################################################################################################
# Module: ssm/tests/test_ssm_linear_gaussian.py
# Description: Tests for linear TemporalGaussian state space models (and Kalman filtering/smoothing).
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

import unittest

import jax.numpy as jnp
from jax import random
import numpy.testing as npt
from mocat.src.ssm.ssm import StateSpaceModel
from mocat.src.ssm.filtering import BootstrapFilter, run_particle_filter_for_marginals, initiate_particles
from mocat.src.ssm.backward import backward_simulation
from mocat.src.ssm.online_smoothing import propagate_particle_smoother


class TestSSM(unittest.TestCase):
    ssm_scenario: StateSpaceModel
    len_t: int = 20
    t: jnp.ndarray = jnp.arange(len_t, dtype='float32')
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

        npt.assert_array_less(self.sim_samps.x[:, 0], jnp.max(self.pf_samps.value, axis=1)[:, 0])
        npt.assert_array_less(jnp.min(self.pf_samps.value, axis=1)[:, 0], self.sim_samps.x[:, 0])

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
        npt.assert_array_less(self.sim_samps.x[:, 0], jnp.max(self.backward_samps.value, axis=1)[:, 0])
        npt.assert_array_less(jnp.min(self.backward_samps.value, axis=1)[:, 0], self.sim_samps.x[:, 0])
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

    def _test_online_smoothing_pf_full(self):
        if not hasattr(self, 'sim_samps'):
            self.sim_samps = self.ssm_scenario.simulate(self.t, random.PRNGKey(0))
        pf = BootstrapFilter()
        len_t = len(self.t)
        rkeys = random.split(random.PRNGKey(0), len_t)

        particles = initiate_particles(self.ssm_scenario, pf, self.n,
                                       rkeys[0], self.sim_samps.y[0], self.t[0])
        for i in range(1, len_t):
            particles = propagate_particle_smoother(self.ssm_scenario, pf, particles,
                                                    self.sim_samps.y[i], self.t[i], rkeys[i], 3,
                                                    False)
        npt.assert_array_less(((self.sim_samps.x[:, 0] - jnp.max(particles.value, axis=1)[:, 0]) > 0).mean(), 0.1)
        npt.assert_array_less(((self.sim_samps.x[:, 0] - jnp.min(particles.value, axis=1)[:, 0]) < 0).mean(), 0.1)

    def _test_online_smoothing_pf_rejection(self):
        if not hasattr(self, 'sim_samps'):
            self.sim_samps = self.ssm_scenario.simulate(self.t, random.PRNGKey(0))
        pf = BootstrapFilter()
        len_t = len(self.t)
        rkeys = random.split(random.PRNGKey(0), len_t)

        particles = initiate_particles(self.ssm_scenario, pf, self.n,
                                       rkeys[0], self.sim_samps.y[0], self.t[0])
        for i in range(1, len_t):
            particles = propagate_particle_smoother(self.ssm_scenario, pf, particles,
                                                    self.sim_samps.y[i], self.t[i], rkeys[i], 3,
                                                    False, maximum_rejections=10)

        npt.assert_array_less(((self.sim_samps.x[:, 0] - jnp.max(particles.value, axis=1)[:, 0]) > 0).mean(), 0.1)
        npt.assert_array_less(((self.sim_samps.x[:, 0] - jnp.min(particles.value, axis=1)[:, 0]) < 0).mean(), 0.1)

    def _test_online_smoothing_bs_full(self):
        if not hasattr(self, 'sim_samps'):
            self.sim_samps = self.ssm_scenario.simulate(self.t, random.PRNGKey(0))
        pf = BootstrapFilter()
        len_t = len(self.t)
        rkeys = random.split(random.PRNGKey(0), len_t)

        particles = initiate_particles(self.ssm_scenario, pf, self.n,
                                       rkeys[0], self.sim_samps.y[0], self.t[0])
        for i in range(1, len_t):
            particles = propagate_particle_smoother(self.ssm_scenario, pf, particles,
                                                    self.sim_samps.y[i], self.t[i], rkeys[i], 3,
                                                    True)
        npt.assert_array_less(((self.sim_samps.x[:, 0] - jnp.max(particles.value, axis=1)[:, 0]) > 0).mean(), 0.1)
        npt.assert_array_less(((self.sim_samps.x[:, 0] - jnp.min(particles.value, axis=1)[:, 0]) < 0).mean(), 0.1)

    def _test_online_smoothing_bs_rejection(self):
        if not hasattr(self, 'sim_samps'):
            self.sim_samps = self.ssm_scenario.simulate(self.t, random.PRNGKey(0))
        pf = BootstrapFilter()
        len_t = len(self.t)
        rkeys = random.split(random.PRNGKey(0), len_t)

        particles = initiate_particles(self.ssm_scenario, pf, self.n,
                                       rkeys[0], self.sim_samps.y[0], self.t[0])
        for i in range(1, len_t):
            particles = propagate_particle_smoother(self.ssm_scenario, pf, particles,
                                                    self.sim_samps.y[i], self.t[i], rkeys[i], 3,
                                                    True, maximum_rejections=10)
        npt.assert_array_less(((self.sim_samps.x[:, 0] - jnp.max(particles.value, axis=1)[:, 0]) > 0).mean(), 0.1)
        npt.assert_array_less(((self.sim_samps.x[:, 0] - jnp.min(particles.value, axis=1)[:, 0]) < 0).mean(), 0.1)


if __name__ == '__main__':
    unittest.main()
