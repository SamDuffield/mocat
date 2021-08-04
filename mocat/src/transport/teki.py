########################################################################################################################
# Module: transport/teki.py
# Description: Tempered ensemble Kalman inversion for intractable likelihoods.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


from typing import Tuple, Union, Callable

from jax import numpy as jnp, random, vmap

from mocat.src.core import Scenario, cdict
from mocat.src.transport.sampler import TransportSampler
from mocat.src.utils import gaussian_potential, bisect
from mocat.src.metrics import log_ess_log_weight


def calculate_covariances(vals: jnp.ndarray,
                          simulated_data: jnp.ndarray,
                          mean_vals: jnp.ndarray = None,
                          mean_sim_data: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n = vals.shape[0]
    if mean_vals is None:
        mean_vals = vals.mean(0)
    if mean_sim_data is None:
        mean_sim_data = simulated_data.mean(0)

    s_x = (vals - mean_vals) / jnp.sqrt(n - 1)
    s_y = (simulated_data - mean_sim_data) / jnp.sqrt(n - 1)

    cov_x = s_x.T @ s_x
    cov_xy = s_x.T @ s_y
    cov_y = s_y.T @ s_y

    return cov_x, cov_xy, cov_y


class TemperedEKI(TransportSampler):
    name = "Tempered EKI"

    def __init__(self,
                 temperature_schedule: Union[None, float, jnp.ndarray] = None,
                 next_temperature: Callable = None,
                 max_temperature: float = 1.,
                 max_iter: int = int(1e4),
                 term_std: float = 0.,
                 nugget: float = 1e-5,
                 **kwargs):
        self.max_iter = max_iter
        self.max_temperature = max_temperature
        self.temperature_schedule = temperature_schedule
        super().__init__(**kwargs)
        self.parameters.nugget = nugget
        self.parameters.term_std = term_std
        if next_temperature is not None:
            self.next_temperature = next_temperature
        self.ensemble_std = None
        self.prior_stds = None

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == 'temperature_schedule':
            if value is not None:
                self.max_temperature = value[-1]
                self.max_iter = len(value)
                self.next_temperature = lambda state, extra: self.temperature_schedule[extra.iter]

    def startup(self,
                scenario: Scenario,
                n: int,
                initial_state: cdict,
                initial_extra: cdict,
                **kwargs) -> Tuple[cdict, cdict]:
        initial_state, initial_extra = super().startup(scenario, n, initial_state, initial_extra, **kwargs)

        if not hasattr(scenario, 'data') or scenario.data is None:
            raise AttributeError(f'{self.name} requires scenario to have data attribute != None')

        if self.ensemble_std is None:
            ensemble_std = lambda vals: jnp.std(vals, axis=0, ddof=1)
            if hasattr(scenario, 'constrain'):
                self.ensemble_std = lambda vals: ensemble_std(scenario.constrain(vals))
            else:
                self.ensemble_std = ensemble_std

        if not hasattr(self, 'next_temperature'):
            # gamm = scenario.dim ** -0.5
            # self.next_temperature = lambda state, extra: (1 + gamm) ** extra.iter - 1

            gamm = 2 ** (1 / 50)
            self.next_temperature = lambda state, extra: jnp.round(gamm ** extra.iter - 1, 4)

        if not hasattr(initial_state, 'simulated_data'):
            random_keys = random.split(initial_extra.random_key, 2 * n)
            initial_extra.random_key = random_keys[-1]
            initial_state.simulated_data = vmap(scenario.likelihood_sample)(initial_state.value,
                                                                            random_keys[:n])

        self.prior_stds = jnp.std(initial_state.value, axis=0, ddof=1)

        initial_state.temperature = 0.
        initial_extra.data = scenario.data
        initial_extra.prec_y_given_x = jnp.eye(initial_state.simulated_data.shape[-1])
        initial_state.perturb_nan = 0
        return initial_state, initial_extra

    def termination_criterion(self,
                              ensemble_state: cdict,
                              extra: cdict) -> bool:
        return jnp.any(jnp.array([ensemble_state.temperature >= self.max_temperature,
                                  extra.iter >= self.max_iter,
                                  jnp.all(self.ensemble_std(ensemble_state.value) <
                                          (extra.parameters.term_std * self.prior_stds)),
                                  jnp.any(jnp.isnan(ensemble_state.value))]))

    def update(self,
               scenario: Scenario,
               ensemble_state: cdict,
               extra: cdict) -> Tuple[cdict, cdict]:
        extra.iter += 1
        n, d_x = ensemble_state.value.shape
        d_y = ensemble_state.simulated_data.shape[-1]
        random_keys = random.split(extra.random_key, n + 2)
        extra.random_key = random_keys[-1]

        cov_x, cov_xy, cov_y = calculate_covariances(ensemble_state.value, ensemble_state.simulated_data)
        cov_y_given_x = cov_y - cov_xy.T @ jnp.linalg.inv(cov_x + self.parameters.nugget * jnp.eye(d_x)) @ cov_xy
        cov_y_given_x_chol = jnp.linalg.cholesky(cov_y_given_x + self.parameters.nugget * jnp.eye(d_y))
        extra.prec_y_given_x = jnp.linalg.inv(cov_y_given_x + self.parameters.nugget * jnp.eye(d_y))

        prev_temp = ensemble_state.temperature
        new_temp = self.next_temperature(ensemble_state, extra)
        alph = 1. / (new_temp - prev_temp)
        ensemble_state.temperature = new_temp

        cov_alph = cov_y + (alph - 1) * cov_y_given_x
        kalman_gain = cov_xy @ jnp.linalg.inv(cov_alph + self.parameters.nugget * jnp.eye(d_y))

        perturbs = jnp.sqrt(alph - 1) * random.normal(random_keys[-2], shape=(n, d_y)) @ cov_y_given_x_chol.T
        ensemble_state.perturb_nan = jnp.isnan(perturbs).sum()
        perturbs = jnp.where(jnp.isnan(perturbs), 0., perturbs)

        ensemble_state.value = ensemble_state.value \
                               + (scenario.data - ensemble_state.simulated_data + perturbs) @ kalman_gain.T

        ensemble_state.simulated_data = vmap(scenario.likelihood_sample)(ensemble_state.value,
                                                                         random_keys[:n])

        return ensemble_state, extra


class AdaptiveTemperedEKI(TemperedEKI):

    def __init__(self,
                 max_temperature: float = 1.,
                 max_iter: int = int(1e4),
                 nugget: float = 1e-5,
                 term_std: float = 0.,
                 ess_threshold: float = 0.9,
                 bisection_tol: float = 1e-5,
                 max_bisection_iter: int = 1000,
                 **kwargs):
        super().__init__(temperature_schedule=None,
                         max_temperature=max_temperature, max_iter=max_iter,
                         nugget=nugget, term_std=term_std, **kwargs)
        self.parameters.ess_threshold = ess_threshold
        self.parameters.bisection_tol = bisection_tol
        self.parameters.max_bisection_iter = max_bisection_iter

    def next_temperature(self,
                         ensemble_state: cdict,
                         extra: cdict) -> float:
        temperature_bounds = jnp.array([ensemble_state.temperature, self.max_temperature])
        data_diff = ensemble_state.simulated_data - extra.data
        pseudo_likelihood_potential = vmap(lambda diffi: 0.5 * diffi.T @ extra.prec_y_given_x @ diffi)(data_diff)
        log_n_samp_threshold = jnp.log(len(pseudo_likelihood_potential) * self.parameters.ess_threshold)
        bisect_out_bounds, bisect_out_evals, bisect_out_iter \
            = bisect(lambda x: log_ess_log_weight(- (x - ensemble_state.temperature) * pseudo_likelihood_potential)
                               - log_n_samp_threshold,
                     temperature_bounds,
                     max_iter=self.parameters.max_bisection_iter,
                     tol=self.parameters.bisection_tol)

        return bisect_out_bounds[jnp.argmin(jnp.abs(bisect_out_evals))]
