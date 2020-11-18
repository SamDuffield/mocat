########################################################################################################################
# Module: ssm/scenarios/lorenz63.py
# Description: Latent Lorenz63 dynamics with Gaussian transition and observation noise.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from functools import partial

from jax import numpy as np, random, jit
from jax.experimental.ode import odeint

from mocat.src.ssm.nonlinear_gaussian import NonLinearGaussian


class Lorenz63(NonLinearGaussian):
    name = 'Lorenz 63'
    dim = 3

    def __init__(self,
                 sigma: float = 10.,
                 rho: float = 28.,
                 beta: float = 8 / 3,
                 **kwargs):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        super().__init__(**kwargs)

    @staticmethod
    def lorenz63_dynamics(x: np.ndarray,
                          t: float,
                          sigma: float,
                          rho: float,
                          beta: float) -> np.ndarray:
        return np.array([sigma * (x[1] - x[0]),
                         x[0] * (rho - x[2]) - x[1],
                         x[0] * x[1] - beta * x[2]])

    @staticmethod
    @jit
    def _lorenz63_integrator(x: np.ndarray,
                             delta_t: float,
                             sigma: float,
                             rho: float,
                             beta: float) -> np.ndarray:
        return odeint(Lorenz63.lorenz63_dynamics, x, np.array([0, delta_t]), sigma, rho, beta)[-1]

    def transition_function(self,
                            x_previous: np.ndarray,
                            t_previous: float,
                            t_new: float) -> np.ndarray:
        return self._lorenz63_integrator(x_previous, t_new - t_previous, self.sigma, self.rho, self.beta)

