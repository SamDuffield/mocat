########################################################################################################################
# Module: ssm/linear_gaussian.py
# Description: Linear TemporalGaussian state-space models plus Kalman filtering/smoothing
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


from typing import Union, Any

from jax import numpy as jnp, random
from mocat.src.utils import gaussian_potential, extract_dimension, reset_covariance
from mocat.src.ssm.ssm import StateSpaceModel


class LinearGaussian(StateSpaceModel):
    name = 'Linear TemporalGaussian'

    # x_{t_new} ~ N(F_{t_new} @ x_{t_previous}, Q_{t_new})
    # y_t ~ N(H_t @ x_t, R_t)

    def get_initial_mean(self,
                         t: Union[float, None]) -> jnp.ndarray:
        raise NotImplementedError

    def get_initial_covariance_sqrt(self,
                                    t: Union[float, None]) -> jnp.ndarray:
        raise NotImplementedError

    def get_initial_precision_sqrt(self,
                                   t: Union[float, None]) -> jnp.ndarray:
        raise NotImplementedError

    def get_initial_precision_det(self,
                                  t: Union[float, None]) -> float:
        raise NotImplementedError

    def initial_potential(self,
                          x: jnp.ndarray,
                          t: Union[float, None]) -> Union[float, jnp.ndarray]:
        init_mean = self.get_initial_mean(t)
        init_prec_sqrt = self.get_initial_precision_sqrt(t)
        init_prec_det = self.get_initial_precision_det(t)
        return gaussian_potential(x, init_mean, sqrt_prec=init_prec_sqrt, det_prec=init_prec_det)

    def initial_sample(self,
                       t: Union[float, None],
                       random_key: jnp.ndarray) -> Union[float, jnp.ndarray]:
        init_mean = self.get_initial_mean(t)
        init_cov_sqrt = self.get_initial_covariance_sqrt(t)
        return random.normal(random_key, shape=(self.dim,)) @ init_cov_sqrt.T + init_mean

    def get_transition_matrix(self,
                              t_previous: float,
                              t_new: float) -> jnp.ndarray:
        raise NotImplementedError

    def get_transition_covariance_sqrt(self,
                                       t_previous: float,
                                       t_new: float) -> jnp.ndarray:
        raise NotImplementedError

    def get_transition_precision_sqrt(self,
                                      t_previous: float,
                                      t_new: float) -> jnp.ndarray:
        raise NotImplementedError

    def get_transition_precision_det(self,
                                     t_previous: float,
                                     t_new: float) -> float:
        raise NotImplementedError

    def transition_potential(self,
                             x_previous: jnp.ndarray,
                             t_previous: float,
                             x_new: jnp.ndarray,
                             t_new: float) -> Union[float, jnp.ndarray]:
        transition_mat = self.get_transition_matrix(t_previous, t_new)
        transition_prec_sqrt = self.get_transition_precision_sqrt(t_previous, t_new)
        transition_prec_det = self.get_transition_precision_det(t_previous, t_new)
        return gaussian_potential(x_new,
                                  x_previous @ transition_mat.T,
                                  sqrt_prec=transition_prec_sqrt,
                                  det_prec=transition_prec_det)

    def transition_sample(self,
                          x_previous: jnp.ndarray,
                          t_previous: float,
                          t_new: float,
                          random_key: jnp.ndarray) -> jnp.ndarray:
        transition_mat = self.get_transition_matrix(t_previous, t_new)
        transition_cov_sqrt = self.get_transition_covariance_sqrt(t_previous, t_new)
        return (x_previous @ transition_mat.T
                + random.normal(random_key, shape=x_previous.shape) @ transition_cov_sqrt.T).reshape(x_previous.shape)

    def get_likelihood_matrix(self,
                              t: float) -> jnp.ndarray:
        raise NotImplementedError

    def get_likelihood_covariance_sqrt(self,
                                       t: float) -> jnp.ndarray:
        raise NotImplementedError

    def get_likelihood_precision_sqrt(self,
                                      t: float) -> jnp.ndarray:
        raise NotImplementedError

    def get_likelihood_precision_det(self,
                                     t: float) -> float:
        raise NotImplementedError

    def transition_function(self,
                            x_previous: jnp.ndarray,
                            t_previous: float,
                            t_new: float) -> jnp.ndarray:
        return x_previous @ self.get_transition_matrix(t_previous, t_new).T

    def likelihood_potential(self,
                             x: jnp.ndarray,
                             y: jnp.ndarray,
                             t: float) -> Union[float, jnp.ndarray]:
        likelihood_mat = self.get_likelihood_matrix(t)
        likelihood_prec_sqrt = self.get_likelihood_precision_sqrt(t)
        likelihood_prec_det = self.get_likelihood_precision_det(t)
        return gaussian_potential(y,
                                  x @ likelihood_mat.T,
                                  sqrt_prec=likelihood_prec_sqrt,
                                  det_prec=likelihood_prec_det)

    def likelihood_sample(self,
                          x: jnp.ndarray,
                          t: float,
                          random_key: jnp.ndarray) -> jnp.ndarray:
        likelihood_mat = self.get_likelihood_matrix(t)
        likelihood_cov_sqrt = self.get_likelihood_covariance_sqrt(t)

        rand_shape = list(x.shape)
        rand_shape[-1] = likelihood_cov_sqrt.shape[0]
        return (x @ likelihood_mat.T
                + random.normal(random_key, shape=rand_shape) @ likelihood_cov_sqrt.T).reshape(x.shape)


class TimeHomogenousLinearGaussian(LinearGaussian):
    name = 'Time-homogenous Linear TemporalGaussian'

    # x_{t_new} ~ N(F @ x_{t_previous}, Q)
    # y_t ~ N(H @ x_t, R)
    # where F, Q, H, R are all time-homogenous
    # typically this means t is of the form jnp.arange(start, end, step)

    initial_covariance_sqrt: jnp.ndarray = None
    initial_precision_sqrt: jnp.ndarray = None
    initial_precision_det: float = None
    transition_covariance_sqrt: jnp.ndarray = None
    transition_precision_sqrt: jnp.ndarray = None
    transition_precision_det: float = None
    likelihood_covariance_sqrt: jnp.ndarray = None
    likelihood_precision_sqrt: jnp.ndarray = None
    likelihood_precision_det: float = None

    def __init__(self,
                 initial_mean: jnp.ndarray = None,
                 initial_covariance: jnp.ndarray = None,
                 transition_matrix: jnp.ndarray = None,
                 transition_covariance: jnp.ndarray = None,
                 likelihood_matrix: jnp.ndarray = None,
                 likelihood_covariance: jnp.ndarray = None,
                 name: str = None,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        if 'dim' not in kwargs:
            self.dim = extract_dimension(initial_mean, initial_covariance,
                                         transition_covariance,
                                         likelihood_matrix, likelihood_covariance)
        if self.dim is None:
            raise AttributeError(f'Could not find dimension for {self.__class__.__name__}')

        if initial_mean is None:
            initial_mean = jnp.zeros(self.dim)
        self.initial_mean = initial_mean

        if initial_covariance is None:
            initial_covariance = jnp.eye(self.dim)
        self.initial_covariance = initial_covariance

        if transition_matrix is None:
            transition_matrix = jnp.eye(self.dim)
        self.transition_matrix = transition_matrix

        if transition_covariance is None:
            transition_covariance = jnp.eye(self.dim)
        self.transition_covariance = transition_covariance

        if likelihood_matrix is None:
            likelihood_matrix = jnp.eye(self.dim)
        self.likelihood_matrix = likelihood_matrix

        self.dim_obs = self.likelihood_matrix.shape[0]

        if likelihood_covariance is None:
            likelihood_covariance = jnp.eye(self.dim_obs)
        self.likelihood_covariance = likelihood_covariance

    def __setattr__(self,
                    key: str,
                    value: Any):
        self.__dict__[key] = value
        if key[-10:] == 'covariance':
            reset_covariance(self, key, value)

    def get_initial_mean(self,
                         t: Union[float, None]) -> jnp.ndarray:
        return self.initial_mean

    def get_initial_covariance_sqrt(self,
                                    t: Union[float, None]) -> jnp.ndarray:
        return self.initial_covariance_sqrt

    def get_initial_precision_sqrt(self,
                                   t: Union[float, None]) -> jnp.ndarray:
        return self.initial_precision_sqrt

    def get_initial_precision_det(self,
                                  t: Union[float, None]) -> float:
        return self.initial_precision_det

    def get_transition_matrix(self,
                              t_previous: float,
                              t_new: float) -> jnp.ndarray:
        return self.transition_matrix

    def get_transition_covariance_sqrt(self,
                                       t_previous: float,
                                       t_new: float) -> jnp.ndarray:
        return self.transition_covariance_sqrt

    def get_transition_precision_sqrt(self,
                                      t_previous: float,
                                      t_new: float) -> jnp.ndarray:
        return self.transition_precision_sqrt

    def get_transition_precision_det(self,
                                     t_previous: float,
                                     t_new: float) -> float:
        return self.transition_precision_det

    def get_likelihood_matrix(self,
                              t: float) -> jnp.ndarray:
        return self.likelihood_matrix

    def get_likelihood_covariance_sqrt(self,
                                       t: float) -> jnp.ndarray:
        return self.likelihood_covariance_sqrt

    def get_likelihood_precision_sqrt(self,
                                      t: float) -> jnp.ndarray:
        return self.likelihood_precision_sqrt

    def get_likelihood_precision_det(self,
                                     t: float) -> float:
        return self.likelihood_precision_det
