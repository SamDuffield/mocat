########################################################################################################################
# Module: tests/test_utils.py
# Description: Tests for mocat utils
#
# Web: https://github.com/SamDuffield/mocat-jax
########################################################################################################################

import unittest

import jax.numpy as np
import numpy.testing as npt

from mocat.src import utils
from mocat.src.core import CDict


class TestLeaveOneOut(unittest.TestCase):
    n = 10

    def test_0(self):
        inds0 = utils.leave_one_out_indices(self.n, 0)
        npt.assert_array_equal(inds0, np.arange(1, self.n + 1))

    def test_2(self):
        inds2 = utils.leave_one_out_indices(self.n, 2)
        npt.assert_array_equal(inds2[:2], np.arange(2))
        npt.assert_array_equal(inds2[2:], np.arange(3, self.n + 1))

    def test_n_plus_1(self):
        inds_n_plus_1 = utils.leave_one_out_indices(self.n, self.n+1)
        npt.assert_array_equal(inds_n_plus_1, np.arange(self.n))


class TestGaussianPotential(unittest.TestCase):

    def test_n1_d1(self):
        x = np.array([7.])
        npt.assert_array_equal(utils.gaussian_potential(x), np.array(24.5))

        m = np.array([1.])
        npt.assert_array_equal(utils.gaussian_potential(x, m), np.array(18.))

        prec = np.array([[2.]])
        npt.assert_array_equal(utils.gaussian_potential(x, prec=prec), np.array(49.))
        npt.assert_array_equal(utils.gaussian_potential(x, m, prec), np.array(36.))

    def test_n1_d5(self):
        x = np.ones(5)
        npt.assert_array_equal(utils.gaussian_potential(x), np.array(2.5))

        m = 3.
        npt.assert_array_equal(utils.gaussian_potential(x, m), np.array(10.))

        m = np.ones(5) * 3.
        npt.assert_array_equal(utils.gaussian_potential(x, m), np.array(10.))

        prec = np.eye(5) * 2
        npt.assert_array_equal(utils.gaussian_potential(x, prec=prec), np.array(5.))
        npt.assert_array_equal(utils.gaussian_potential(x, m, prec), np.array(20.))

    def test_n5_d2(self):
        x = np.ones((5, 2))
        npt.assert_array_equal(utils.gaussian_potential(x), np.ones(5))

        m = 3.
        npt.assert_array_equal(utils.gaussian_potential(x, m), np.ones(5) * 4)

        m = np.ones(2) * 3.
        npt.assert_array_equal(utils.gaussian_potential(x, m), np.ones(5) * 4)

        sqrt_prec = np.array([[5., 0.], [2., 3.]])
        npt.assert_array_equal(utils.gaussian_potential(x, sqrt_prec=sqrt_prec), np.ones(5) * 29)
        npt.assert_array_equal(utils.gaussian_potential(x, m, sqrt_prec=sqrt_prec), np.ones(5) * 116)


class TestLeapfrog(unittest.TestCase):

    start_state = CDict(value=np.zeros(2),
                        grad_potential=np.array([1., 2.]),
                        momenta=np.ones(2))

    stepsize = 0.1
    n_steps = 3

    def test_leapfrog(self):
        out_state = utils.leapfrog(self.start_state,
                                   lambda x: x,
                                   self.stepsize,
                                   self.n_steps)[-1]

        npt.assert_array_equal(out_state.value, np.array([0.28120947, 0.266409]))
        npt.assert_array_equal(out_state.grad_potential, np.array([0.28120947, 0.266409]))
        npt.assert_array_equal(out_state.momenta, np.array([0.9075344, 0.8597696]))

    def test_all_leapfrog(self):
        self.start_state._all_leapfrog_value = True
        self.start_state._all_leapfrog_momenta = True

        out_state = utils.leapfrog(self.start_state,
                                   lambda x: x,
                                   self.stepsize,
                                   self.n_steps)[-1]

        self.test_leapfrog()

        npt.assert_array_equal(out_state.value, np.array([0.28120947, 0.266409]))
        npt.assert_array_equal(out_state.grad_potential, np.array([0.28120947, 0.266409]))
        npt.assert_array_equal(out_state.momenta, np.array([0.9075344, 0.8597696]))

        all_leapfrog_value = out_state._all_leapfrog_value
        all_leapfrog_momenta = out_state._all_leapfrog_momenta

        npt.assert_array_equal(all_leapfrog_value.shape, (self.n_steps + 1, 2))
        npt.assert_array_equal(all_leapfrog_momenta.shape, (2*self.n_steps + 1, 2))

        npt.assert_array_equal(all_leapfrog_value[0], np.zeros(2))
        npt.assert_array_equal(all_leapfrog_value[-1], np.array([0.28120947, 0.266409]))

        npt.assert_array_equal(all_leapfrog_momenta[0], np.ones(2))
        npt.assert_array_equal(all_leapfrog_momenta[-1], np.array([0.9075344, 0.8597696]))


class TestStackedWhile(unittest.TestCase):

    def test_trival(self):

        fin_iter = 10
        cond_fun = lambda x, _: x < fin_iter
        body_fun = lambda x, _: (x + 1, None)

        stack = utils.while_loop_stacked(cond_fun, body_fun, (0, None), 100)

        npt.assert_array_equal(stack, np.arange(1, fin_iter + 1))


class TestBisect(unittest.TestCase):

    def test_squared_increasing(self):
        bisect_fun = lambda x: x**2 - 10
        increasing_init_bounds = np.array([0, 1e2])

        increasing_bound, increasing_evals, increasing_iter = utils.bisect(bisect_fun, increasing_init_bounds)

        npt.assert_(np.any(np.abs(increasing_evals < 1e-3)))

    def test_squared_decreasing(self):
        bisect_fun = lambda x: x ** 2 - 10
        decreasing_init_bounds = np.array([-1e1, 0])

        decreasing_bound, decreasing_evals, decreasing_iter = utils.bisect(bisect_fun, decreasing_init_bounds)

        npt.assert_(np.any(np.abs(decreasing_evals < 1e-3)))


if __name__ == '__main__':
    unittest.main()
