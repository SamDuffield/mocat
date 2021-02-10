########################################################################################################################
# Module: tests/test_utils.py
# Description: Tests for mocat utils
#
# Web: https://github.com/SamDuffield/mocat-jax
########################################################################################################################

import unittest
import warnings

import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jax import vmap, random
import numpy.testing as npt

from mocat.src import utils
from mocat.src.core import cdict


class TestLeaveOneOut(unittest.TestCase):
    n = 10

    def test_0(self):
        inds0 = utils.leave_one_out_indices(self.n, 0)
        npt.assert_array_equal(inds0, jnp.arange(1, self.n + 1))

    def test_2(self):
        inds2 = utils.leave_one_out_indices(self.n, 2)
        npt.assert_array_equal(inds2[:2], jnp.arange(2))
        npt.assert_array_equal(inds2[2:], jnp.arange(3, self.n + 1))

    def test_n_plus_1(self):
        inds_n_plus_1 = utils.leave_one_out_indices(self.n, self.n + 1)
        npt.assert_array_equal(inds_n_plus_1, jnp.arange(self.n))


class TestGaussiajnpotential(unittest.TestCase):

    def test_n1_d1(self):
        x = jnp.array([7.])
        npt.assert_array_equal(utils.gaussian_potential(x), -multivariate_normal.logpdf(x, 0., 1.))

        m = jnp.array([1.])
        npt.assert_array_equal(utils.gaussian_potential(x, m), -multivariate_normal.logpdf(x, m, 1.))

        prec = jnp.array([[2.]])
        # test diag
        npt.assert_array_equal(utils.gaussian_potential(x, prec=prec[0]), -multivariate_normal.logpdf(x, 0, 1 / prec))
        npt.assert_array_almost_equal(utils.gaussian_potential(x, m, prec[0]),
                                      -multivariate_normal.logpdf(x, m, 1 / prec), decimal=4)
        # test full (omits norm constant)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            npt.assert_array_equal(utils.gaussian_potential(x, prec=prec), 0.5 * x ** 2 * prec)
            npt.assert_array_equal(utils.gaussian_potential(x, m, prec), 0.5 * (x - m) ** 2 * prec)

    def test_n1_d5(self):
        x = jnp.ones(5)
        npt.assert_array_equal(utils.gaussian_potential(x), -multivariate_normal.logpdf(x, jnp.zeros_like(x), 1.))

        m = 3.
        npt.assert_array_equal(utils.gaussian_potential(x, m), -multivariate_normal.logpdf(x, m * jnp.ones_like(x), 1.))

        m = jnp.ones(5) * 3.
        npt.assert_array_equal(utils.gaussian_potential(x, m), -multivariate_normal.logpdf(x, m, 1.))

        # diagonal precision
        prec = jnp.eye(5) * 2
        # test diag
        npt.assert_array_almost_equal(utils.gaussian_potential(x, prec=jnp.diag(prec)),
                                      -multivariate_normal.logpdf(x, jnp.zeros_like(x), jnp.linalg.inv(prec)), decimal=5)
        npt.assert_array_almost_equal(utils.gaussian_potential(x, m, prec=jnp.diag(prec), det_prec=2 ** 5),
                                      -multivariate_normal.logpdf(x, m, jnp.linalg.inv(prec)), decimal=5)
        # test full (omits norm constant)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            npt.assert_array_equal(utils.gaussian_potential(x, prec=prec), 0.5 * x.T @ prec @ x)
            npt.assert_array_equal(utils.gaussian_potential(x, m, prec), 0.5 * (x - m).T @ prec @ (x - m))
        # test full with det
        npt.assert_array_almost_equal(utils.gaussian_potential(x, prec=prec, det_prec=2 ** 5),
                                      -multivariate_normal.logpdf(x, jnp.zeros_like(x), jnp.linalg.inv(prec)), decimal=5)
        npt.assert_array_almost_equal(utils.gaussian_potential(x, m, prec=prec, det_prec=2 ** 5),
                                      -multivariate_normal.logpdf(x, m, jnp.linalg.inv(prec)), decimal=5)

        # non-diagonal precision
        sqrt_prec = jnp.arange(25).reshape(5, 5) / 100 + jnp.eye(5)
        prec = sqrt_prec @ sqrt_prec.T
        # test full (omits norm constant)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            npt.assert_array_equal(utils.gaussian_potential(x, prec=prec), 0.5 * x.T @ prec @ x)
            npt.assert_array_equal(utils.gaussian_potential(x, m, prec), 0.5 * (x - m).T @ prec @ (x - m))
        # test full with det
        npt.assert_array_almost_equal(utils.gaussian_potential(x, prec=prec, det_prec=jnp.linalg.det(prec)),
                                      -multivariate_normal.logpdf(x, jnp.zeros_like(x), jnp.linalg.inv(prec)), decimal=5)
        npt.assert_array_almost_equal(utils.gaussian_potential(x, m, prec=prec, det_prec=jnp.linalg.det(prec)),
                                      -multivariate_normal.logpdf(x, m, jnp.linalg.inv(prec)), decimal=5)

    def test_n5_d2(self):
        x = jnp.ones((5, 2))
        npt.assert_array_equal(utils.gaussian_potential(x),
                               jnp.repeat(-multivariate_normal.logpdf(x[0], jnp.zeros(x.shape[-1]), 1.), 5))

        m = 3.
        npt.assert_array_equal(utils.gaussian_potential(x, m),
                               jnp.repeat(-multivariate_normal.logpdf(x[0], m * jnp.ones(x.shape[-1]), 1.), 5))

        m = jnp.ones(2) * 3.
        npt.assert_array_equal(utils.gaussian_potential(x, m),
                               jnp.repeat(-multivariate_normal.logpdf(x[0], m, 1.), 5))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sqrt_prec = jnp.array([[5., 0.], [2., 3.]])
            npt.assert_array_equal(utils.gaussian_potential(x, sqrt_prec=sqrt_prec),
                                   jnp.ones(5) * 29)
            npt.assert_array_equal(utils.gaussian_potential(x, sqrt_prec=sqrt_prec),
                                   jnp.repeat(0.5 * x[0].T @ sqrt_prec @ sqrt_prec.T @ x[0], 5))
            npt.assert_array_equal(utils.gaussian_potential(x, m, sqrt_prec=sqrt_prec),
                                   jnp.repeat(0.5 * (x[0] - m).T @ sqrt_prec @ sqrt_prec.T @ (x[0] - m), 5))


class TestLeapfrog(unittest.TestCase):
    stepsize = 0.1
    leapfrog_steps = 3

    start_state = cdict(value=jnp.zeros(2),
                        potential=jnp.array(0.),
                        grad_potential=jnp.array([1., 2.]),
                        momenta=jnp.ones(2),
                        auxiliary_float=5.,
                        auxiliary_0darray=jnp.array(0.),
                        auxiliary_2darray=jnp.zeros(2))

    def test_leapfrog(self):
        full_state = utils.leapfrog(lambda x, _: (0., x),
                                    self.start_state,
                                    self.stepsize,
                                    random.split(random.PRNGKey(0), self.leapfrog_steps))
        out_state = full_state[-1]

        npt.assert_array_equal(out_state.value, jnp.array([0.28120947, 0.266409]))
        npt.assert_array_equal(out_state.grad_potential, jnp.array([0.28120947, 0.266409]))
        npt.assert_array_equal(out_state.momenta, jnp.array([0.9075344, 0.8597696]))

    def test_all_leapfrog(self):
        full_state = utils.leapfrog(lambda x, _: (0., x),
                                    self.start_state,
                                    self.stepsize,
                                    random.split(random.PRNGKey(0), self.leapfrog_steps))

        npt.assert_array_equal(full_state.value.shape, (self.leapfrog_steps + 1, 2))
        npt.assert_array_equal(full_state.momenta.shape, (2 * self.leapfrog_steps + 1, 2))

        npt.assert_array_equal(full_state.value[0], jnp.zeros(2))
        npt.assert_array_equal(full_state.value[-1], jnp.array([0.28120947, 0.266409]))

        npt.assert_array_equal(full_state.momenta[0], jnp.ones(2))
        npt.assert_array_equal(full_state.momenta[-1], jnp.array([0.9075344, 0.8597696]))


class TestStackedWhile(unittest.TestCase):

    def test_trival(self):
        fin_iter = 10
        cond_fun = lambda x, _: x < fin_iter
        body_fun = lambda x, _: (x + 1, None)

        stack = utils.while_loop_stacked(cond_fun, body_fun, (0, None), 100)

        npt.assert_array_equal(stack, jnp.arange(1, fin_iter + 1))


class TestBisect(unittest.TestCase):

    def test_squared_increasing(self):
        bisect_fun = lambda x: x ** 2 - 10
        increasing_init_bounds = jnp.array([0, 1e2])

        increasing_bound, increasing_evals, increasing_iter = utils.bisect(bisect_fun, increasing_init_bounds)

        npt.assert_(jnp.any(jnp.abs(increasing_evals < 1e-3)))

    def test_squared_decreasing(self):
        bisect_fun = lambda x: x ** 2 - 10
        decreasing_init_bounds = jnp.array([-1e1, 0])

        decreasing_bound, decreasing_evals, decreasing_iter = utils.bisect(bisect_fun, decreasing_init_bounds)

        npt.assert_(jnp.any(jnp.abs(decreasing_evals < 1e-3)))


class TestBFGS(unittest.TestCase):
    hess_inv = jnp.array([[1., -0.5], [-0.5, 2.0]])
    # hess_inv = jnp.array([[1., 0.], [0., 5.0]])
    hess = jnp.linalg.inv(hess_inv)
    hess_inv_sqrt = jnp.linalg.cholesky(hess_inv)
    hess_sqrt = jnp.linalg.cholesky(hess)

    grad_gauss_pot = lambda h, x: h @ x

    vals = random.normal(random.PRNGKey(0), shape=(100, 2))
    grads = vmap(grad_gauss_pot, in_axes=(None, 0))(hess, vals)

    def test_correct_init(self):
        a5_sqrt, a5_inv_sqrt = utils.bfgs(self.hess_sqrt, self.hess_inv_sqrt,
                                          self.vals[:5], self.grads[:5])
        npt.assert_array_almost_equal(a5_sqrt @ a5_sqrt.T, self.hess, decimal=3)
        npt.assert_array_almost_equal(a5_inv_sqrt @ a5_inv_sqrt.T, self.hess_inv, decimal=3)

        a10_sqrt, a10_inv_sqrt = utils.bfgs(self.hess_sqrt, self.hess_inv_sqrt,
                                            self.vals[:10], self.grads[:10])
        npt.assert_array_almost_equal(a10_sqrt @ a10_sqrt.T, self.hess, decimal=3)
        npt.assert_array_almost_equal(a10_inv_sqrt @ a10_inv_sqrt.T, self.hess_inv, decimal=3)

        a_sqrt, a_inv_sqrt = utils.bfgs(self.hess_sqrt, self.hess_inv_sqrt,
                                        self.vals, self.grads)
        npt.assert_array_almost_equal(a_sqrt @ a_sqrt.T, self.hess, decimal=3)
        npt.assert_array_almost_equal(a_inv_sqrt @ a_inv_sqrt.T, self.hess_inv, decimal=3)

    def test_diag_init(self):
        gamma = 2.
        init_sqrt = jnp.eye(2) * gamma
        init_inv_sqrt = jnp.eye(2) / gamma

        a5_sqrt, a5_inv_sqrt = utils.bfgs(init_sqrt, init_inv_sqrt,
                                          self.vals[:5], self.grads[:5])
        npt.assert_array_almost_equal(a5_sqrt @ a5_sqrt.T, self.hess, decimal=3)
        npt.assert_array_almost_equal(a5_inv_sqrt @ a5_inv_sqrt.T, self.hess_inv, decimal=1)

        a10_sqrt, a10_inv_sqrt = utils.bfgs(init_sqrt, init_inv_sqrt,
                                            self.vals[:10], self.grads[:10])
        npt.assert_array_almost_equal(a10_sqrt @ a10_sqrt.T, self.hess, decimal=3)
        npt.assert_array_almost_equal(a10_inv_sqrt @ a10_inv_sqrt.T, self.hess_inv, decimal=3)

        a_sqrt, a_inv_sqrt = utils.bfgs(init_sqrt, init_inv_sqrt,
                                        self.vals, self.grads)
        npt.assert_array_almost_equal(a_sqrt @ a_sqrt.T, self.hess, decimal=3)
        npt.assert_array_almost_equal(a_inv_sqrt @ a_inv_sqrt.T, self.hess_inv, decimal=3)


if __name__ == '__main__':
    unittest.main()
