########################################################################################################################
# Module: ssm/scenarios/lorenz63.py
# Description: Latent Lorenz63 dynamics with Gaussian transition and observation noise.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from jax import numpy as jnp, random, jit
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
    def lorenz63_dynamics(x: jnp.ndarray,
                          t: float,
                          sigma: float,
                          rho: float,
                          beta: float) -> jnp.ndarray:
        return jnp.array([sigma * (x[1] - x[0]),
                         x[0] * (rho - x[2]) - x[1],
                         x[0] * x[1] - beta * x[2]])

    @staticmethod
    @jit
    def _lorenz63_integrator(x: jnp.ndarray,
                             delta_t: float,
                             sigma: float,
                             rho: float,
                             beta: float) -> jnp.ndarray:
        return odeint(Lorenz63.lorenz63_dynamics, x, jnp.array([0, delta_t]), sigma, rho, beta)[-1]

    def transition_function(self,
                            x_previous: jnp.ndarray,
                            t_previous: float,
                            t_new: float) -> jnp.ndarray:
        return self._lorenz63_integrator(x_previous, t_new - t_previous, self.sigma, self.rho, self.beta)

