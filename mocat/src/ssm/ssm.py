########################################################################################################################
# Module: ssm/ssm.py
# Description: state-space model class
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from functools import partial
from typing import Union

import jax.numpy as np
from jax import vmap, grad, jit
from jax.lax import scan
from jax import random

from mocat.src.core import cdict


class StateSpaceModel:
    name = None
    dim = None
    dim_obs = None

    def __init__(self,
                 name: str = None,
                 **kwargs):
        if name is not None:
            self.name = name

        for key, value in kwargs.items():
            setattr(self, key, value)

        if not hasattr(self, 'grad_initial_potential'):
            self.grad_initial_potential = jit(grad(self.initial_potential, argnums=0))

        if not hasattr(self, 'grad_transition_potential'):
            self.grad_transition_potential = jit(grad(self.transition_potential, argnums=(0, 2)))

        if not hasattr(self, 'grad_likelihood_potential'):
            self.grad_likelihood_potential = jit(grad(self.likelihood_potential, argnums=0))

    def __repr__(self):
        return f"mocat.StateSpaceModel.{self.__class__.__name__}"

    def initial_potential(self,
                          x: np.ndarray,
                          t: float) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.__class__.__name__} initial_potential not implemented')

    def initial_sample(self,
                       t: float,
                       random_key: np.ndarray) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.__class__.__name__} initial_sample not implemented')

    def transition_potential(self,
                             x_previous: np.ndarray,
                             t_previous: float,
                             x_new: np.ndarray,
                             t_new: float) -> Union[float, np.ndarray]:
        raise AttributeError(f'{self.__class__.__name__} transition_potential not implemented')

    def transition_sample(self,
                          x_previous: np.ndarray,
                          t_previous: float,
                          t_new: float,
                          random_key: np.ndarray) -> np.ndarray:
        raise AttributeError(f'{self.__class__.__name__} transition_sample not implemented')

    def likelihood_potential(self,
                             x: np.ndarray,
                             y: np.ndarray,
                             t: float) -> Union[float, np.ndarray]:
        raise NotImplementedError(f'{self.__class__.__name__} likelihood_potential not implemented')

    def likelihood_sample(self,
                          x: np.ndarray,
                          t: float,
                          random_key: np.ndarray) -> np.ndarray:
        raise AttributeError(f'{self.__class__.__name__} transition_sample not implemented')

    def _smoothing_potential(self,
                             x_all: np.ndarray,
                             y_all: np.ndarray,
                             t_all: np.ndarray) -> Union[float, np.ndarray]:
        # x_all: shape = (T, d)
        # t_all: shape = (T,)
        # y_all: shape = (T, d_y)
        return self.initial_potential(x_all[0], t_all[0]) \
               + vmap(self.transition_potential)(x_all[:-1], t_all[:-1], x_all[1:], t_all[1:]).sum() \
               + vmap(self.likelihood_potential)(x_all, y_all, t_all).sum()

    def smoothing_potential(self,
                            x_all: np.ndarray,
                            y_all: np.ndarray,
                            t_all: np.ndarray) -> Union[float, np.ndarray]:
        if len(t_all) == 1:
            return self.initial_potential(x_all[0], t_all[0]) \
                   + self.likelihood_potential(x_all[0], y_all[0], t_all[0])
        elif len(t_all) == 2:
            trans_pot = self.transition_potential(x_all[0], t_all[0], x_all[1], t_all[1])
            return np.vstack([self.initial_potential(x_all[0], t_all[0]) + trans_pot[0] \
                              + self.likelihood_potential(x_all[0], y_all[0], t_all[0]),
                              trans_pot[1] + self.likelihood_potential(x_all[1], y_all[1], t_all[1])])
        else:
            return self._smoothing_potential(x_all, y_all, t_all)

    @partial(jit, static_argnums=(0,))
    def _grad_smoothing_potential(self,
                                  x_all: np.ndarray,
                                  y_all: np.ndarray,
                                  t_all: np.ndarray) -> Union[float, np.ndarray]:
        # it may well be quicker to do grad(smoothing_potential)
        # but currently there is a jax bug that breaks on combination of scan - grad - vmap - odeint
        initial_grads = self.grad_initial_potential(x_all[0], t_all[0])
        transition_prev_grads, transition_next_grads = vmap(self.grad_transition_potential)(x_all[:-1], t_all[:-1],
                                                                                            x_all[1:], t_all[1:])
        likelihood_grads = vmap(self.grad_likelihood_potential)(x_all, y_all, t_all)
        return np.vstack([initial_grads + transition_prev_grads[0] + likelihood_grads[0],
                          transition_prev_grads[1:] + transition_next_grads[:-1] + likelihood_grads[1:-1],
                          transition_next_grads[-1] + likelihood_grads[-1]])

    def grad_smoothing_potential(self,
                                 x_all: np.ndarray,
                                 y_all: np.ndarray,
                                 t_all: np.ndarray) -> Union[float, np.ndarray]:
        if len(t_all) == 1:
            return (self.grad_initial_potential(x_all[0], t_all[0])
                     + self.grad_likelihood_potential(x_all[0], y_all[0], t_all[0]))[np.newaxis]
        elif len(t_all) == 2:
            grad_trans_pot = self.grad_transition_potential(x_all[0], t_all[0], x_all[1], t_all[1])
            return np.vstack([self.grad_initial_potential(x_all[0], t_all[0]) + grad_trans_pot[0] \
                              + self.grad_likelihood_potential(x_all[0], y_all[0], t_all[0]),
                              grad_trans_pot[1] + self.grad_likelihood_potential(x_all[1], y_all[1], t_all[1])])
        else:
            return self._grad_smoothing_potential(x_all, y_all, t_all)

    def simulate(self,
                 t_all: np.ndarray,
                 random_key: np.ndarray) -> cdict:

        len_t = len(t_all)

        random_keys = random.split(random_key, 2 * len_t)
        latent_keys = random_keys[:len_t]
        obs_keys = random_keys[len_t:]

        x_init = self.initial_sample(t_all[0], latent_keys[0])

        def transition_body(x, i):
            new_x = self.transition_sample(x, t_all[i - 1], t_all[i], latent_keys[i])
            return new_x, new_x

        _, x_all_but_zero = scan(transition_body,
                                 x_init,
                                 np.arange(1, len_t))

        x_all = np.vstack([x_init, x_all_but_zero])

        y = vmap(self.likelihood_sample)(x_all, t_all, obs_keys)

        out_cdict = cdict(x=x_all, y=y, t=t_all, name=f'{self.name} run')
        return out_cdict
