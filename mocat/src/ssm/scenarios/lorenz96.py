########################################################################################################################
# Module: ssm/scenarios/lorenz96.py
# Description: Latent Lorenz96 dynamics with TemporalGaussian transition and observation noise.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from jax import numpy as np, jit
from jax.experimental.ode import odeint

from mocat.src.ssm.nonlinear_gaussian import NonLinearGaussian


def lorenz96_dynamics(x: np.ndarray,
                      t: float,
                      forcing_constant: float) -> np.ndarray:
    d = len(x)
    return (x[(np.arange(d) + 1) % d] - x[(np.arange(d) - 2) % d]) * x[(np.arange(d) - 1) % d] \
           - x + forcing_constant


@jit
def lorenz96_integrator(x: np.ndarray,
                         delta_t: float,
                         forcing_constant: float) -> np.ndarray:
    return odeint(lorenz96_dynamics, x, np.array([0, delta_t]), forcing_constant)[-1]


class Lorenz96(NonLinearGaussian):

    name = 'Lorenz 96'

    def __init__(self,
                 dim: int = 40,
                 forcing_constant: float = 8.,
                 **kwargs):
        self.dim = dim
        self.forcing_constant = forcing_constant
        super().__init__(**kwargs)

    def transition_function(self,
                            x_previous: np.ndarray,
                            t_previous: float,
                            t_new: float) -> np.ndarray:
        return lorenz96_integrator(x_previous, t_new - t_previous, self.forcing_constant)


