########################################################################################################################
# Module: ssm/nonlinear_gaussian.py
# Description: State-space models with non-linear transition function, linear likelihood
#              and all (time-homogenous) Gaussian noise
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from functools import partial
from typing import Union, Any, Tuple

from jax import numpy as np, random, vmap, jit

from mocat.src.ssm.ssm import StateSpaceModel
from mocat.src.ssm._utils import kalman_gain
from mocat.src.utils import gaussian_potential, extract_dimension, reset_covariance
from mocat.src.ssm.filters import ParticleFilter


class NonLinearGaussian(StateSpaceModel):
    # x_{t_new} ~ N(f(x_{t_previous}, t_previous, t_new), Q)
    # y_t ~ N(H @ x_t, R)
    # where Q, H, R are all time-homogenous
    # typically this means t is of the form np.arange(start, end, step)

    initial_covariance_sqrt: np.ndarray = None
    initial_precision_sqrt: np.ndarray = None
    transition_covariance_sqrt: np.ndarray = None
    transition_precision_sqrt: np.ndarray = None
    likelihood_covariance_sqrt: np.ndarray = None
    likelihood_precision_sqrt: np.ndarray = None

    def __init__(self,
                 initial_mean: np.ndarray = None,
                 initial_covariance: float = None,
                 transition_covariance: np.ndarray = None,
                 likelihood_matrix: np.ndarray = None,
                 likelihood_covariance: np.ndarray = None,
                 name: str = None):
        if self.dim is None:
            self.dim = extract_dimension(initial_mean, initial_covariance,
                                         transition_covariance,
                                         likelihood_matrix, likelihood_covariance)
        if self.dim is None:
            raise AttributeError(f'Could not find dimension for {self.__class__.__name__}')

        if initial_mean is None:
            initial_mean = np.zeros(self.dim)
        self.initial_mean = initial_mean

        if initial_covariance is None:
            initial_covariance = np.eye(self.dim)
        self.initial_covariance = initial_covariance

        if transition_covariance is None:
            transition_covariance = np.eye(self.dim)
        self.transition_covariance = transition_covariance

        if likelihood_matrix is None:
            likelihood_matrix = np.eye(self.dim, dtype='float32')

        self.likelihood_matrix = likelihood_matrix
        self.dim_obs = self.likelihood_matrix.shape[0]

        if likelihood_covariance is None:
            likelihood_covariance = np.eye(self.dim_obs)
        self.likelihood_covariance = likelihood_covariance

        super().__init__(name=name)

    def __setattr__(self,
                    key: str,
                    value: Any):
        self.__dict__[key] = value
        if key[-10:] == 'covariance':
            reset_covariance(self, key, value)

    def initial_potential(self,
                          x: np.ndarray,
                          t: float) -> Union[float, np.ndarray]:
        return gaussian_potential(x, self.initial_mean, sqrt_prec=self.initial_precision_sqrt)

    def initial_sample(self,
                       t: float,
                       random_key: np.ndarray) -> Union[float, np.ndarray]:
        return random.normal(random_key, shape=(self.dim,)) @ self.initial_covariance_sqrt.T \
               + self.initial_mean

    def transition_function(self,
                            x_previous: np.ndarray,
                            t_previous: float,
                            t_new: float) -> np.ndarray:
        raise NotImplementedError(f'{self.__class__.__name__} nonlinear transition_function not implemented')

    def transition_potential(self,
                             x_previous: np.ndarray,
                             t_previous: float,
                             x_new: np.ndarray,
                             t_new: float) -> Union[float, np.ndarray]:
        return gaussian_potential(x_new,
                                  self.transition_function(x_previous, t_previous, t_new),
                                  sqrt_prec=self.transition_precision_sqrt)

    def transition_sample(self,
                          x_previous: np.ndarray,
                          t_previous: float,
                          t_new: float,
                          random_key: np.ndarray) -> np.ndarray:
        return self.transition_function(x_previous, t_previous, t_new) \
               + random.normal(random_key, shape=x_previous.shape) @ self.transition_covariance_sqrt

    def likelihood_potential(self,
                             x: np.ndarray,
                             y: np.ndarray,
                             t: float) -> Union[float, np.ndarray]:
        return gaussian_potential(y,
                                  x @ self.likelihood_matrix.T,
                                  sqrt_prec=self.likelihood_precision_sqrt)

    def likelihood_sample(self,
                          x: np.ndarray,
                          t: float,
                          random_key: np.ndarray) -> np.ndarray:

        rand_shape = list(x.shape)
        rand_shape[-1] = self.likelihood_covariance_sqrt.shape[0]
        return x @ self.likelihood_matrix.T \
               + random.normal(random_key, shape=rand_shape) @ self.likelihood_covariance_sqrt.T


class OptimalNonLinearGaussianParticleFilter(ParticleFilter):
    # NOTE we assume time-homogenous transition noise covariance
    # that is x_t ~ N(f_t(x_s), Q) where Q is independent of s, t or t-s
    # time-inhomogoneous transition noise covariance would require matrix inversions
    # at every time point rather than just once
    # typically this means t is of the form np.arange(start, end, step)

    name = 'Optimal Non-linear Gaussian Particle Filter'

    initial_kalman_gain: np.ndarray = None
    initial_conditioned_precision_sqrt: np.ndarray = None
    initial_conditioned_covariance_sqrt: np.ndarray = None
    proposal_kalman_gain: np.ndarray = None
    proposal_precision_sqrt: np.ndarray = None
    proposal_covariance_sqrt: np.ndarray = None
    weight_precision_sqrt: np.ndarray = None

    def startup(self,
                ssm_scenario: NonLinearGaussian):
        init_cov_sqrt = ssm_scenario.initial_covariance_sqrt
        init_prec_sqrt = ssm_scenario.initial_precision_sqrt
        transition_cov_sqrt = ssm_scenario.transition_covariance_sqrt
        lik_mat = ssm_scenario.likelihood_matrix
        lik_cov_sqrt = ssm_scenario.likelihood_covariance_sqrt
        lik_prec_sqrt = ssm_scenario.likelihood_precision_sqrt

        self.initial_kalman_gain = kalman_gain(init_cov_sqrt, lik_mat, lik_cov_sqrt)
        lik_mat_t_lik_prec_sqrt = lik_mat.T @ lik_prec_sqrt
        init_cond_prec = init_prec_sqrt @ init_prec_sqrt \
                         + lik_mat_t_lik_prec_sqrt @ lik_mat_t_lik_prec_sqrt.T
        self.initial_conditioned_precision_sqrt = np.linalg.cholesky(init_cond_prec)
        self.initial_conditioned_covariance_sqrt = np.linalg.inv(self.initial_conditioned_precision_sqrt)

        lik_mat_transition_cov_sqrt = lik_mat @ transition_cov_sqrt
        weight_precision = np.linalg.inv(lik_mat_transition_cov_sqrt @ lik_mat_transition_cov_sqrt.T
                                         + lik_cov_sqrt @ lik_cov_sqrt.T)
        self.proposal_kalman_gain = kalman_gain(transition_cov_sqrt, lik_mat, lik_cov_sqrt, inv_mat=weight_precision)

        proposal_cov = ssm_scenario.transition_covariance \
                       - self.proposal_kalman_gain @ ssm_scenario.likelihood_matrix \
                       @ ssm_scenario.transition_covariance
        self.proposal_covariance_sqrt = np.linalg.cholesky(proposal_cov)
        self.proposal_precision_sqrt = np.linalg.inv(self.proposal_covariance_sqrt)

        self.weight_precision_sqrt = np.linalg.cholesky(weight_precision)

    def initial_potential(self,
                          ssm_scenario: NonLinearGaussian,
                          x: np.ndarray,
                          y: np.ndarray,
                          t: float) -> Union[float, np.ndarray]:
        initial_conditioned_mean = ssm_scenario.initial_mean \
                                   + self.initial_kalman_gain \
                                   @ (y - ssm_scenario.likelihood_matrix @ ssm_scenario.initial_mean)
        return gaussian_potential(x, initial_conditioned_mean, sqrt_prec=self.initial_conditioned_precision_sqrt)

    def initial_sample(self,
                       ssm_scenario: NonLinearGaussian,
                       y: np.ndarray,
                       t: float,
                       random_key: np.ndarray) -> np.ndarray:
        initial_conditioned_mean = ssm_scenario.initial_mean \
                                   + self.initial_kalman_gain \
                                   @ (y - ssm_scenario.likelihood_matrix @ ssm_scenario.initial_mean)
        rand_shape = (ssm_scenario.dim,) if random_key.ndim == 1 else (len(random_key), ssm_scenario.dim)
        random_key = random_key if random_key.ndim == 1 else random_key[0]
        return initial_conditioned_mean \
               + random.normal(random_key, shape=rand_shape) @ self.initial_conditioned_covariance_sqrt

    def initial_log_weight(self,
                           ssm_scenario: NonLinearGaussian,
                           x: np.ndarray,
                           y: np.ndarray,
                           t: float) -> Union[float, np.ndarray]:
        return np.zeros(len(x))

    def initial_sample_and_weight_vectorised(self,
                                             ssm_scenario: NonLinearGaussian,
                                             y: np.ndarray,
                                             t: float,
                                             random_keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        init_vals = self.initial_sample(ssm_scenario, y, t, random_keys)
        return init_vals, np.zeros(len(init_vals))

    def proposal_potential(self,
                           ssm_scenario: NonLinearGaussian,
                           x_previous: np.ndarray,
                           t_previous: float,
                           x_new: np.ndarray,
                           y_new: np.ndarray,
                           t_new: float) -> Union[float, np.ndarray]:
        mx = ssm_scenario.transition_function(x_previous, t_previous, t_new)
        conditioned_mean = mx + (y_new - mx @ ssm_scenario.likelihood_matrix.T) @ self.proposal_kalman_gain.T
        return gaussian_potential(x_new, conditioned_mean,
                                  sqrt_prec=self.proposal_precision_sqrt)

    def proposal_sample(self,
                        ssm_scenario: NonLinearGaussian,
                        x_previous: np.ndarray,
                        t_previous: float,
                        y_new: np.ndarray,
                        t_new: float,
                        random_key: np.ndarray) -> np.ndarray:
        mx = ssm_scenario.transition_function(x_previous, t_previous, t_new)
        conditioned_mean = mx + (y_new - mx @ ssm_scenario.likelihood_matrix.T) @ self.proposal_kalman_gain.T
        return conditioned_mean \
               + random.normal(random_key, shape=(ssm_scenario.dim,)) @ self.proposal_covariance_sqrt.T

    def intermediate_log_weight(self,
                                ssm_scenario: NonLinearGaussian,
                                x_previous: np.ndarray,
                                t_previous: float,
                                x_new: np.ndarray,
                                y_new: np.ndarray,
                                t_new: float) -> Union[float, np.ndarray]:
        mx = ssm_scenario.transition_function(x_previous, t_previous, t_new)
        return -gaussian_potential(y_new,
                                   mx @ ssm_scenario.likelihood_matrix.T,
                                   sqrt_prec=self.weight_precision_sqrt)

    @partial(jit, static_argnums=(0, 1))
    def propose_and_intermediate_weight_vectorised(self,
                                                   ssm_scenario: NonLinearGaussian,
                                                   x_previous: np.ndarray,
                                                   t_previous: float,
                                                   y_new: np.ndarray,
                                                   t_new: float,
                                                   random_keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mx = vmap(ssm_scenario.transition_function, (0, None, None))(x_previous, t_previous, t_new)
        conditioned_mean = mx + (y_new - mx @ ssm_scenario.likelihood_matrix.T) @ self.proposal_kalman_gain.T
        x_new = conditioned_mean \
                + random.normal(random_keys[0], shape=x_previous.shape) @ self.proposal_covariance_sqrt.T

        log_weights_new = -gaussian_potential(y_new,
                                              mx @ ssm_scenario.likelihood_matrix.T,
                                              sqrt_prec=self.weight_precision_sqrt)
        return x_new, log_weights_new


class EnsembleKalmanFilter(ParticleFilter):
    # NOTE we assume time-homogenous transition noise covariance
    # that is x_t ~ N(f_t(x_s), Q) where Q is independent of s, t or t-s
    # time-inhomogoneous transition noise covariance would require matrix inversions
    # at every time point rather than just once
    # typically this means t is of the form np.arange(start, end, step)

    name = 'Optimal Non-linear Gaussian Particle Filter'

    initial_kalman_gain: np.ndarray = None
    initial_conditioned_precision_sqrt: np.ndarray = None
    initial_conditioned_covariance_sqrt: np.ndarray = None

    def startup(self,
                ssm_scenario: NonLinearGaussian):
        init_cov_sqrt = ssm_scenario.initial_covariance_sqrt
        init_prec_sqrt = ssm_scenario.initial_precision_sqrt
        lik_mat = ssm_scenario.likelihood_matrix
        lik_cov_sqrt = ssm_scenario.likelihood_covariance_sqrt
        lik_prec_sqrt = ssm_scenario.likelihood_precision_sqrt

        self.initial_kalman_gain = kalman_gain(init_cov_sqrt, lik_mat, lik_cov_sqrt)
        lik_mat_t_lik_prec_sqrt = lik_mat.T @ lik_prec_sqrt
        init_cond_prec = init_prec_sqrt @ init_prec_sqrt \
                         + lik_mat_t_lik_prec_sqrt @ lik_mat_t_lik_prec_sqrt.T
        initial_conditioned_precision_sqrt = np.linalg.cholesky(init_cond_prec)
        self.initial_conditioned_covariance_sqrt = np.linalg.inv(initial_conditioned_precision_sqrt)

    def initial_sample_and_weight_vectorised(self,
                                             ssm_scenario: NonLinearGaussian,
                                             y: np.ndarray,
                                             t: float,
                                             random_keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        initial_conditioned_mean = ssm_scenario.initial_mean \
                                   + self.initial_kalman_gain \
                                   @ (y - ssm_scenario.likelihood_matrix @ ssm_scenario.initial_mean)
        rand_shape = (len(random_keys), ssm_scenario.dim)
        init_x = initial_conditioned_mean \
                 + random.normal(random_keys[0], shape=rand_shape) @ self.initial_conditioned_covariance_sqrt
        return init_x, np.zeros(len(random_keys))

    @partial(jit, static_argnums=(0, 1))
    def propose_and_intermediate_weight_vectorised(self,
                                                   ssm_scenario: NonLinearGaussian,
                                                   x_previous: np.ndarray,
                                                   t_previous: float,
                                                   y_new: np.ndarray,
                                                   t_new: float,
                                                   random_keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = x_previous.shape[0]
        split_keys = random.split(random_keys[0], 2)

        mx = vmap(ssm_scenario.transition_function, (0, None, None))(x_previous, t_previous, t_new) \
             + random.normal(split_keys[0], shape=x_previous.shape) @ ssm_scenario.transition_covariance_sqrt.T

        spread_matrix = (mx - mx.mean(0)).T / np.sqrt(n - 1)

        prop_kalman_gain = kalman_gain(spread_matrix,
                                       ssm_scenario.likelihood_matrix,
                                       ssm_scenario.likelihood_covariance_sqrt)

        y_prop = mx @ ssm_scenario.likelihood_matrix.T \
                 + random.normal(split_keys[1], shape=(n, len(y_new))) @ ssm_scenario.likelihood_covariance_sqrt.T

        x_new = mx + (y_new - y_prop) @ prop_kalman_gain.T

        return x_new, np.zeros(n)
