########################################################################################################################
# Module: tests/test_toy_scenarios.py
# Description: Tests for bivariate toy scenarios
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


import unittest

import jax.numpy as jnp
import numpy.testing as npt

from mocat.src.scenarios.twodim import toy_examples


class TestGaussian(unittest.TestCase):
    scenario = toy_examples.Gaussian()

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 2)
        npt.assert_array_equal(self.scenario.mean, jnp.zeros(2))
        npt.assert_array_equal(self.scenario.covariance, jnp.eye(2))

    def test_vec_potential(self):
        npt.assert_array_equal(self.scenario.vec_potential(jnp.array([0, 2, 4]),
                                                           jnp.array([1, 3, 5])),
                               jnp.array([[0.5, 2.5, 8.5],
                                          [4.5, 6.5, 12.5],
                                          [12.5, 14.5, 20.5]]))

    def test_auto_axes_lims(self):
        self.scenario.auto_axes_lims()
        npt.assert_almost_equal(-self.scenario.xlim[0], self.scenario.xlim[1], decimal=4)
        npt.assert_almost_equal(self.scenario.xlim[0], self.scenario.ylim[0], decimal=4)
        npt.assert_almost_equal(self.scenario.xlim[1], self.scenario.ylim[1], decimal=4)

        npt.assert_(self.scenario.xlim[1] > 1)
        npt.assert_(self.scenario.xlim[1] < 5)

        self.scenario.covariance = jnp.array([[10., 9.], [9., 50.]])
        self.scenario.auto_axes_lims()

        npt.assert_almost_equal(-self.scenario.xlim[0], self.scenario.xlim[1], decimal=4)
        npt.assert_almost_equal(-self.scenario.ylim[0], self.scenario.ylim[1], decimal=4)

        npt.assert_(self.scenario.xlim[1] > 5)
        npt.assert_(self.scenario.xlim[1] < 15)

        npt.assert_(self.scenario.ylim[1] > 10)
        npt.assert_(self.scenario.ylim[1] < 20)

        self.scenario.covariance = jnp.eye(2)


class TestBanana(unittest.TestCase):
    scenario = toy_examples.Banana()
    scenario.curviness = 0.03
    scenario.lengthiness = 100

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 2)

    def test_potential(self):
        npt.assert_array_equal(self.scenario.potential(jnp.zeros(2)), 4.5)
        npt.assert_array_equal(self.scenario.potential(jnp.arange(2)), 2.)

    def test_grad_potential(self):
        npt.assert_array_equal(self.scenario.grad_potential(jnp.zeros(2)), [0., -3.])
        npt.assert_array_equal(self.scenario.grad_potential(jnp.arange(2, dtype='float32')), [0., -2.])

    def test_vec_potential(self):
        npt.assert_array_almost_equal(self.scenario.vec_potential(jnp.array([0, 2, 4]),
                                                                  jnp.array([1, 3, 5])),
                                      jnp.array([[2., 1.7872, 1.2352],
                                                 [0., 0.0272, 0.1952],
                                                 [2., 2.2672, 3.1552]]))

    def test_auto_axes_lims(self):
        self.scenario.auto_axes_lims()
        npt.assert_almost_equal(-self.scenario.xlim[0], self.scenario.xlim[1], decimal=4)
        npt.assert_(abs(self.scenario.ylim[0]) > self.scenario.ylim[1])

        npt.assert_(self.scenario.xlim[1] > 15)
        npt.assert_(self.scenario.xlim[1] < 30)

        npt.assert_(self.scenario.ylim[0] < -5)
        npt.assert_(self.scenario.ylim[0] > -20)

        npt.assert_(self.scenario.ylim[1] > 1)
        npt.assert_(self.scenario.ylim[1] < 15)


class TestNealFunnel(unittest.TestCase):
    scenario = toy_examples.NealFunnel()

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 2)

    def test_potential(self):
        npt.assert_array_equal(self.scenario.potential(jnp.zeros(2)), 0.)
        npt.assert_array_almost_equal(self.scenario.potential(jnp.arange(2)), 0.166667)

    def test_grad_potential(self):
        npt.assert_array_almost_equal(self.scenario.grad_potential(jnp.zeros(2)), jnp.zeros(2))
        npt.assert_array_almost_equal(self.scenario.grad_potential(jnp.arange(2, dtype='float32')),
                                      jnp.array([0., 0.333333]))

    def test_vec_potential(self):
        npt.assert_array_almost_equal(self.scenario.vec_potential(jnp.array([0, 2, 4]),
                                                                  jnp.array([1, 3, 5])),
                                      jnp.array([[0.166667, 1.379728, 5.018912],
                                                 [1.5, 1.94626, 3.285041],
                                                 [4.166667, 4.330836, 4.823347]]))

    def test_auto_axes_lims(self):
        self.scenario.auto_axes_lims()
        npt.assert_almost_equal(-self.scenario.xlim[0], self.scenario.xlim[1], decimal=4)
        npt.assert_almost_equal(-self.scenario.ylim[0], self.scenario.ylim[1], decimal=4)

        npt.assert_(self.scenario.xlim[1] > 1)
        npt.assert_(self.scenario.xlim[1] < 10)

        npt.assert_(self.scenario.ylim[1] > 1)
        npt.assert_(self.scenario.ylim[1] < 20)


class TestGaussianMixture(unittest.TestCase):
    scenario = toy_examples.GaussianMixture()

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 2)
        npt.assert_equal(self.scenario.means.ndim, 2)
        npt.assert_equal(self.scenario.means.shape[-1], 2)
        npt.assert_equal(self.scenario.covariances.ndim, 3)
        npt.assert_equal(self.scenario.covariances.shape[-1], 2)
        npt.assert_equal(self.scenario.precisions.ndim, 3)
        npt.assert_equal(self.scenario.precisions.shape[-1], 2)
        npt.assert_equal(self.scenario.precision_sqrts.ndim, 3)
        npt.assert_equal(self.scenario.precision_sqrts.shape[-1], 2)

    def test_vec_potential(self):
        npt.assert_array_almost_equal(self.scenario.vec_potential(jnp.array([0, 2, 4]),
                                                                  jnp.array([1, 3, 5])),
                                      jnp.array([[5.012875, 3.705686, 5.706022],
                                                 [5.031018, 3.72383, 5.724165],
                                                 [9.031024, 7.723836, 9.724172]]))

    def test_auto_axes_lims(self):
        self.scenario.auto_axes_lims()
        npt.assert_almost_equal(-self.scenario.xlim[0], self.scenario.xlim[1], decimal=4)
        npt.assert_almost_equal(self.scenario.xlim[0], self.scenario.ylim[0], decimal=4)
        npt.assert_almost_equal(self.scenario.xlim[1], self.scenario.ylim[1], decimal=4)

        npt.assert_(self.scenario.xlim[1] > 1)
        npt.assert_(self.scenario.xlim[1] < 10)

        self.scenario.covariances = jnp.array([[10., 9.], [9., 30.]])
        self.test_basic()

        self.scenario.auto_axes_lims()

        npt.assert_almost_equal(-self.scenario.xlim[0], self.scenario.xlim[1], decimal=4)
        npt.assert_almost_equal(-self.scenario.ylim[0], self.scenario.ylim[1], decimal=4)

        npt.assert_(self.scenario.xlim[1] > 7)
        npt.assert_(self.scenario.xlim[1] < 15)

        npt.assert_(self.scenario.ylim[1] > 11)
        npt.assert_(self.scenario.ylim[1] < 16)

        self.scenario.covariances = jnp.eye(2)


class TestDoubleWell(unittest.TestCase):
    scenario = toy_examples.DoubleWell()

    def test_basic(self):
        npt.assert_equal(self.scenario.dim, 2)

    def test_vec_potential(self):
        npt.assert_array_almost_equal(self.scenario.vec_potential(jnp.array([0, 2, 4]),
                                                                  jnp.array([1, 3, 5])),
                                      jnp.array([[-0.25, 1.75, 55.75],
                                                 [15.75, 17.75, 71.75],
                                                 [143.75, 145.75, 199.75]]))

    def test_auto_axes_lims(self):
        self.scenario.auto_axes_lims()
        npt.assert_almost_equal(-self.scenario.xlim[0], self.scenario.xlim[1], decimal=4)
        npt.assert_almost_equal(self.scenario.xlim[0], self.scenario.ylim[0], decimal=4)
        npt.assert_almost_equal(self.scenario.xlim[1], self.scenario.ylim[1], decimal=4)

        npt.assert_(self.scenario.xlim[1] > 1)
        npt.assert_(self.scenario.xlim[1] < 5)


if __name__ == '__main__':
    unittest.main()
