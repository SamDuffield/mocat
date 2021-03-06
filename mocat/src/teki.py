########################################################################################################################
# Module: teki.py
# Description: Tempered ensemble Kalman inversion.
#              Iteratively simulate data (or summary statistic) to get joint empirical mean and cov
#              before conditioning on given data.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from time import time
from typing import Tuple, Union, Callable

from jax import numpy as np, random, vmap
from jax.lax import scan
from jax.scipy.special import logsumexp

from mocat.src.core import cdict, Scenario
from mocat.src.abc.abc import ABCScenario
from mocat.src.utils import while_loop_stacked, bisect

nugget = 1e-3


def run_tempered_ensemble_kalman_inversion(scenario: Union[Scenario, Callable],
                                           n_samps: int,
                                           random_key: np.ndarray,
                                           data: Union[float, np.ndarray] = None,
                                           prior_samps: np.ndarray = None,
                                           temperature_schedule: np.ndarray = None,
                                           max_iter: int = 1000,
                                           max_temp: float = 1.,
                                           continuation_criterion: Callable[
                                               [cdict, cdict, np.ndarray], bool] = None,
                                           theta: float = None,
                                           ess_threshold: float = None,
                                           min_temp_increase: float = 0.01,
                                           max_bisection_iter: int = 1000,
                                           bisection_tol: float = 1e-5,
                                           name: str = None) -> cdict:
    if isinstance(scenario, Scenario) and hasattr(scenario, 'simulate'):
        simulator = scenario.simulate
    elif isinstance(scenario, ABCScenario):
        simulator = lambda x, r_key: scenario.summarise_data(scenario.simulate_data(x, r_key))
    else:
        simulator = scenario

    if data is None:
        if hasattr(scenario, 'summary_statistic'):
            data = scenario.summary_statistic
        elif hasattr(scenario, 'data'):
            data = scenario.data
        else:
            raise TypeError(f"TEKI cannot find summary_statistic or data in arguments or {scenario}.")

    data = np.atleast_1d(data)

    if prior_samps is None and hasattr(scenario, 'prior_sample'):
        random_key, prior_key = random.split(random_key)
        prior_samp_keys = random.split(prior_key, n_samps)
        prior_samps = vmap(scenario.prior_sample)(prior_samp_keys)

    start = time()
    if temperature_schedule is None:
        if ess_threshold is not None:
            samps = _run_adaptive_teki_ess(simulator, n_samps, random_key, data, prior_samps,
                                           max_iter, max_temp, ess_threshold, min_temp_increase,
                                           max_bisection_iter, bisection_tol)
        else:
            if theta is None:
                theta = data.size / 2
            if continuation_criterion is None:
                continuation_criterion = lambda state, extra, prev_sim_data: state.temperature_schedule < max_temp
            else:
                max_temp = np.inf

            samps = _run_adaptive_teki(simulator, n_samps, random_key, data, prior_samps,
                                       max_iter, continuation_criterion, theta, max_temp)

    else:
        samps = _run_presheduled_teki(simulator, n_samps, random_key, data, prior_samps,
                                      temperature_schedule)
    samps.value.block_until_ready()
    end = time()

    samps.time = end - start

    if name is None and scenario.name is not None:
        name = scenario.name + ": TEKI"

    samps.name = name
    samps.run_params = cdict(max_iter=max_iter, max_temp=max_temp, theta=theta,
                             ess_threshold=ess_threshold,
                             min_temp_increase=min_temp_increase, max_bisection_iter=max_bisection_iter,
                             bisection_tol=bisection_tol)
    return samps


def _run_presheduled_teki(simulator: Callable,
                          n_samps: int,
                          random_key: np.ndarray,
                          data: np.ndarray,
                          prior_samps: np.ndarray,
                          temperature_schedule: np.ndarray) -> cdict:
    d = prior_samps.shape[-1]
    d_y = len(data)

    n_iter = len(temperature_schedule)

    sim_key, perturb_key = random.split(random_key)
    sim_keys = random.split(sim_key, n_iter)
    perturb_keys = random.split(perturb_key, n_iter)

    simulate_data = vmap(lambda samp, key: simulator(samp, key))

    def eki_body(previous_samps: cdict,
                 i: int) -> Tuple[cdict, cdict]:
        out_samps = previous_samps.copy()
        data_samp_keys = random.split(sim_keys[i], n_samps)

        simulated_data \
            = simulate_data(previous_samps.value, data_samp_keys).reshape(n_samps, data.size)

        stack_samps_summs = np.hstack([previous_samps.value, simulated_data])

        stack_mean = np.mean(stack_samps_summs, axis=0)
        stack_cov_sqrt = (stack_samps_summs - stack_mean) / np.sqrt(n_samps - 1)
        stack_cov = stack_cov_sqrt.T @ stack_cov_sqrt

        samps_cov = stack_cov[:d, :d]
        samps_prec = np.linalg.inv(samps_cov)

        sim_data_cov = stack_cov[d:, d:]

        cross_cov_upper = stack_cov[:d, d:]

        inferred_likelihood_noise = sim_data_cov - cross_cov_upper.T @ samps_prec @ cross_cov_upper

        incremental_temp_recip = 1 / (temperature_schedule[i] - temperature_schedule[i - 1])

        tempered_kalman_gain = cross_cov_upper \
                               @ np.linalg.inv(sim_data_cov
                                               + (incremental_temp_recip - 1)
                                               * inferred_likelihood_noise)

        inferred_likelihood_noise_chol = np.linalg.cholesky(inferred_likelihood_noise)

        perturbations = random.normal(perturb_keys[i], (n_samps, d_y)) \
                        @ (inferred_likelihood_noise_chol * np.sqrt(incremental_temp_recip - 1)).T

        out_samps.value = previous_samps.value \
                          + ((data - simulated_data - perturbations)
                             @ tempered_kalman_gain.T)
        return out_samps, out_samps

    initial_state = cdict(value=prior_samps)
    final_samp, samps = scan(eki_body,
                             initial_state,
                             np.arange(1, n_iter))
    samps.value = np.vstack((initial_state.value[np.newaxis],
                             samps.value))
    samps.temperature_schedule = temperature_schedule
    return samps


def _run_adaptive_teki(simulator: Callable,
                       n_samps: int,
                       random_key: np.ndarray,
                       data: np.ndarray,
                       prior_samps: np.ndarray,
                       max_iter: int,
                       continuation_criterion: Callable[
                                                  [cdict, cdict, np.ndarray], bool],
                       theta: float,
                       max_temp: float) -> cdict:
    d = prior_samps.shape[-1]
    d_y = len(data)

    sim_key, perturb_key = random.split(random_key)
    sim_keys = random.split(sim_key, max_iter)
    perturb_keys = random.split(perturb_key, max_iter)

    simulate_data = vmap(lambda samp, key: simulator(samp, key))

    def eki_body(previous_samps: cdict,
                 extra: cdict) -> Tuple[cdict, cdict]:
        extra.i = extra.i + 1
        out_samps = previous_samps.copy()
        data_samp_keys = random.split(sim_keys[extra.i], n_samps)

        simulated_data \
            = simulate_data(previous_samps.value, data_samp_keys).reshape(n_samps, data.size)

        stack_samps_summs = np.hstack([previous_samps.value, simulated_data])

        stack_mean = np.mean(stack_samps_summs, axis=0)
        stack_cov_sqrt = (stack_samps_summs - stack_mean) / np.sqrt(n_samps - 1)
        stack_cov = stack_cov_sqrt.T @ stack_cov_sqrt

        samps_cov = stack_cov[:d, :d]
        samps_prec = np.linalg.inv(samps_cov)

        sim_data_cov = stack_cov[d:, d:]

        cross_cov_upper = stack_cov[:d, d:]

        inferred_likelihood_noise = sim_data_cov - cross_cov_upper.T @ samps_prec @ cross_cov_upper \
                                    + np.eye(d_y) * nugget
        chol_likelihood_noise = np.linalg.cholesky(inferred_likelihood_noise)
        inv_chol_likelihood_noise = np.linalg.inv(chol_likelihood_noise)

        previous_temp = previous_samps.temperature_schedule

        least_squares_functionals = 0.5 * np.square((data - simulated_data) @ inv_chol_likelihood_noise.T).sum(-1)

        new_alpha_recip = np.maximum(theta / least_squares_functionals.mean(),
                                     np.sqrt(theta / np.cov(least_squares_functionals)))
        new_alpha_recip = np.minimum(new_alpha_recip, 1)
        new_alpha = 1 / new_alpha_recip
        new_temp = np.minimum(previous_temp + new_alpha_recip, max_temp)

        tempered_kalman_gain = cross_cov_upper \
                               @ np.linalg.inv(sim_data_cov
                                               + (new_alpha - 1)
                                               * inferred_likelihood_noise)

        perturbations = random.normal(perturb_keys[extra.i], (n_samps, d_y)) \
                        @ (chol_likelihood_noise * np.sqrt(new_alpha - 1)).T

        out_samps.value = previous_samps.value \
                          + ((data - simulated_data - perturbations)
                             @ tempered_kalman_gain.T)

        out_samps.temperature_schedule = new_temp

        extra.continuation = continuation_criterion(out_samps, extra, simulated_data)
        return out_samps, extra

    initial_state = cdict(value=prior_samps, temperature_schedule=0.)
    initial_extra = cdict(i=0, continuation=True)
    samps = while_loop_stacked(lambda state, extra: extra.continuation,
                               eki_body,
                               (initial_state, initial_extra),
                               max_iter)
    samps.value = np.vstack((initial_state.value[np.newaxis],
                             samps.value))
    samps.temperature_schedule = np.hstack((0., samps.temperature_schedule))
    return samps


def _run_adaptive_teki_ess(simulator: Callable,
                           n_samps: int,
                           random_key: np.ndarray,
                           data: np.ndarray,
                           prior_samps: np.ndarray,
                           max_iter: int,
                           max_temp: float,
                           ess_threshold: float,
                           min_temp_increase: float,
                           max_bisection_iter: int,
                           bisection_tol: float) -> cdict:
    d = prior_samps.shape[-1]
    d_y = len(data)

    sim_key, perturb_key = random.split(random_key)
    sim_keys = random.split(sim_key, max_iter)
    perturb_keys = random.split(perturb_key, max_iter)

    simulate_data = vmap(lambda samp, key: simulator(samp, key))

    log_n_samp_threshold = np.log(n_samps * ess_threshold)

    def log_ess(previous_temp: float,
                new_temp: float,
                data_dispersion: np.ndarray) -> float:
        log_weights = -0.5 * (new_temp - previous_temp) * data_dispersion
        return 2 * logsumexp(log_weights) - logsumexp(2 * log_weights)

    def eki_body(previous_samps: cdict,
                 i: int) -> Tuple[cdict, int]:
        i = i + 1
        out_samps = previous_samps.copy()
        data_samp_keys = random.split(sim_keys[i], n_samps)

        simulated_data \
            = simulate_data(previous_samps.value, data_samp_keys).reshape(n_samps, data.size)

        stack_samps_summs = np.hstack([previous_samps.value, simulated_data])

        stack_mean = np.mean(stack_samps_summs, axis=0)
        stack_cov_sqrt = (stack_samps_summs - stack_mean) / np.sqrt(n_samps - 1)
        stack_cov = stack_cov_sqrt.T @ stack_cov_sqrt

        samps_cov = stack_cov[:d, :d]
        samps_prec = np.linalg.inv(samps_cov)

        sim_data_cov = stack_cov[d:, d:]

        cross_cov_upper = stack_cov[:d, d:]

        inferred_likelihood_noise = sim_data_cov - cross_cov_upper.T @ samps_prec @ cross_cov_upper \
                                    + np.eye(d_y) * nugget
        chol_likelihood_noise = np.linalg.cholesky(inferred_likelihood_noise)
        inv_chol_likelihood_noise = np.linalg.inv(chol_likelihood_noise)

        previous_temp = previous_samps.temperature_schedule

        data_dispersion = np.square((data - simulated_data) @ inv_chol_likelihood_noise.T).sum(axis=-1)

        latest_bounds = np.array([previous_temp + min_temp_increase, previous_temp + 1])
        bisect_out_bounds, bisect_out_evals, bisect_out_iter = bisect(lambda x: log_ess(previous_temp,
                                                                                        x,
                                                                                        data_dispersion)
                                                                                - log_n_samp_threshold,
                                                                      latest_bounds,
                                                                      max_iter=max_bisection_iter,
                                                                      tol=bisection_tol)
        new_temp = bisect_out_bounds[np.argmin(np.abs(bisect_out_evals))]

        new_temp = np.minimum(max_temp, new_temp)

        new_alpha = 1 / (new_temp - previous_temp)

        tempered_kalman_gain = cross_cov_upper \
                               @ np.linalg.inv(sim_data_cov
                                               + (new_alpha - 1)
                                               * inferred_likelihood_noise)

        perturbations = random.normal(perturb_keys[i], (n_samps, d_y)) \
                        @ (chol_likelihood_noise * np.sqrt(new_alpha - 1)).T

        out_samps.value = previous_samps.value \
                          + ((data - simulated_data - perturbations)
                             @ tempered_kalman_gain.T)

        out_samps.temperature_schedule = new_temp
        return out_samps, i

    initial_state = cdict(value=prior_samps, temperature_schedule=0.)
    samps = while_loop_stacked(lambda state, extra: state.temperature_schedule < max_temp,
                               eki_body,
                               (initial_state, 0),
                               max_iter)
    samps.value = np.vstack((initial_state.value[np.newaxis],
                             samps.value))
    samps.temperature_schedule = np.hstack((0., samps.temperature_schedule))
    return samps
