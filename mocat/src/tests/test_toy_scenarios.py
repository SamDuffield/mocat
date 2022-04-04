########################################################################################################################
# Module: tests/test_toy_scenarios.py
# Description: Tests for univariate and multivariate toy scenarios
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


import unittest

import jax.numpy as jnp
import numpy.testing as npt

from mocat.src.scenarios import toy_examples


class Test1DGaussian(unittest.TestCase):

    scenario = toy_examples.Gaussian(1, prior_potential=lambda x, rk: 0.)

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 1)
        npt.assert_equal(self.scenario.mean.shape, (1,))
        npt.assert_equal(self.scenario.covariance.shape, (1,))
        npt.assert_equal(self.scenario.precision_sqrt.shape, (1,))

    def test_scalar_potential(self):
        npt.assert_array_equal(self.scenario.potential(0.), 0.)
        npt.assert_array_equal(self.scenario.potential(5.), 12.5)

    def test_scalar_grad_potential(self):
        npt.assert_array_equal(self.scenario.grad_potential(0.), 0.)
        npt.assert_array_equal(self.scenario.grad_potential(5.), 5.)

    def test_setcov(self):
        self.scenario.covariance = jnp.array([[7.]])
        npt.assert_array_almost_equal(self.scenario.precision_sqrt, 0.37796447)


class TestNDGaussian(unittest.TestCase):

    scenario = toy_examples.Gaussian(10, prior_potential=lambda x, rk: 0.)

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 10)
        npt.assert_equal(self.scenario.mean.shape, (10,))
        npt.assert_equal(self.scenario.covariance.shape, (10,))
        npt.assert_equal(self.scenario.precision_sqrt.shape, (10,))

    def test_array_potential(self):
        npt.assert_array_equal(self.scenario.potential(jnp.zeros(10)), 0.)
        npt.assert_array_equal(self.scenario.potential(jnp.ones(10) * 3), 45.)

    def test_array_grad_potential(self):
        npt.assert_array_equal(self.scenario.grad_potential(jnp.zeros(10)), jnp.zeros(10))
        npt.assert_array_equal(self.scenario.grad_potential(jnp.ones(10) * 3.), jnp.ones(10) * 3.)


class Test1DGaussianMixture(unittest.TestCase):

    scenario = toy_examples.GaussianMixture(means=jnp.array([0, 1]), prior_potential=lambda x, rk: 0.)

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 1)
        npt.assert_equal(self.scenario.means.shape, (2, 1))
        npt.assert_equal(self.scenario.covariances.shape, (2, 1, 1))
        npt.assert_equal(self.scenario.precisions.shape, (2, 1, 1))
        npt.assert_equal(self.scenario.precision_sqrts.shape, (2, 1, 1))
        npt.assert_equal(self.scenario.precision_dets.shape, (2,))

    def test_scalar_potential(self):
        npt.assert_array_equal(self.scenario.potential(0.), self.scenario.potential(1.))
        npt.assert_array_almost_equal(self.scenario.potential(5.), 9.601038)

    def test_scalar_grad_potential(self):
        npt.assert_array_equal(self.scenario.grad_potential(0.), -self.scenario.grad_potential(1.))
        npt.assert_array_almost_equal(self.scenario.grad_potential(5.), 4.010987)

    def test_setcovs(self):
        self.scenario.covariances = jnp.repeat(jnp.array([[7.]])[jnp.newaxis, :, :], 2, axis=0)
        npt.assert_array_equal(self.scenario.precisions, jnp.repeat(jnp.array([[1 / 7.]])[jnp.newaxis, :, :], 2, axis=0))
        npt.assert_array_almost_equal(self.scenario.precision_sqrts,
                                      jnp.repeat(jnp.array([[0.37796447]])[jnp.newaxis, :, :], 2, axis=0))


class TestNDGaussianMixture(unittest.TestCase):

    scenario = toy_examples.GaussianMixture(means=jnp.arange(15).reshape(5, 3), prior_potential=lambda x, rk: 0.)

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 3)
        npt.assert_equal(self.scenario.means.shape, (5, 3))
        npt.assert_equal(self.scenario.covariances.shape, (5, 3, 3))
        npt.assert_equal(self.scenario.precisions.shape, (5, 3, 3))
        npt.assert_equal(self.scenario.precision_sqrts.shape, (5, 3, 3))
        npt.assert_equal(self.scenario.precision_dets.shape, (5,))

    def test_array_potential(self):
        npt.assert_array_almost_equal(self.scenario.potential(jnp.zeros(3)), 6.8662534)
        npt.assert_array_almost_equal(self.scenario.potential(jnp.ones(3) * 2), 6.8552055)

    def test_array_grad_potential(self):
        npt.assert_array_almost_equal(self.scenario.grad_potential(jnp.zeros(3)), jnp.array([0., -1, -2]))
        npt.assert_array_almost_equal(self.scenario.grad_potential(jnp.ones(3) * 2.), jnp.array([1.967039,
                                                                                               0.967039,
                                                                                               -0.032961]))


class Test1DDoubleWell(unittest.TestCase):

    scenario = toy_examples.DoubleWell(1, prior_potential=lambda x, rk: 0.)

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 1)

    def test_scalar_potential(self):
        npt.assert_array_equal(self.scenario.potential(-3.), self.scenario.potential(3.))
        npt.assert_array_equal(self.scenario.potential(0.), 0)
        npt.assert_array_almost_equal(self.scenario.potential(5.), 143.75)

    def test_scalar_grad_potential(self):
        npt.assert_array_equal(self.scenario.grad_potential(-3.), -self.scenario.grad_potential(3.))
        npt.assert_array_equal(self.scenario.grad_potential(0.), 0.)
        npt.assert_array_equal(self.scenario.grad_potential(1.), 0.)
        npt.assert_array_equal(self.scenario.grad_potential(-1.), 0.)
        npt.assert_array_almost_equal(self.scenario.grad_potential(5.), 120.)


class TestNDDoubleWell(unittest.TestCase):

    scenario = toy_examples.DoubleWell(6, prior_potential=lambda x, rk: 0.)

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 6)

    def test_array_potential(self):
        npt.assert_array_equal(self.scenario.potential(jnp.zeros(6)), 0.)
        npt.assert_array_equal(self.scenario.potential(jnp.ones(3) * 2), 6.)
        npt.assert_array_almost_equal(self.scenario.potential(jnp.arange(6)), 217.25)

    def test_array_grad_potential(self):
        npt.assert_array_equal(self.scenario.grad_potential(jnp.zeros(6)), jnp.zeros(6))
        npt.assert_array_equal(self.scenario.grad_potential(jnp.ones(6)), jnp.zeros(6))
        npt.assert_array_equal(self.scenario.grad_potential(-jnp.zeros(6)), jnp.zeros(6))
        npt.assert_array_almost_equal(self.scenario.grad_potential(jnp.arange(6, dtype='float32')),
                                      jnp.array([0., 0., 6., 24., 60., 120.]))


if __name__ == '__main__':
    unittest.main()
