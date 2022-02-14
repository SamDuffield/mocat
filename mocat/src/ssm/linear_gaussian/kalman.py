########################################################################################################################
# Module: ssm/linear_gaussian/linear_gaussian_1D_N500.py
# Description: Kalman filtering/smoothing for linear Gaussian state-space models.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Tuple

from jax import numpy as jnp
from jax.lax import scan

from mocat.src.ssm.linear_gaussian.linear_gaussian import LinearGaussian


def run_kalman_filter_for_marginals(lgssm_scenario: LinearGaussian,
                                    y: jnp.ndarray,
                                    t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mu_0 = lgssm_scenario.get_initial_mean(t[0])
    cov_0 = lgssm_scenario.get_initial_covariance_sqrt(t[0])

    l_mat_0 = lgssm_scenario.get_likelihood_matrix(t[0])
    l_cov_sqrt_0 = lgssm_scenario.get_likelihood_covariance_sqrt(t[0])
    l_cov_0 = l_cov_sqrt_0 @ l_cov_sqrt_0.T

    kal_0 = cov_0 @ l_mat_0.T @ jnp.linalg.inv(l_mat_0 @ cov_0 @ l_mat_0.T + l_cov_0)
    mu_0 = mu_0 + kal_0 @ (y[0] - l_mat_0 @ mu_0)
    cov_0 = cov_0 - kal_0 @ l_mat_0 @ cov_0

    def body_fun(carry: Tuple[jnp.ndarray, jnp.ndarray],
                 i: int) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        mu_tmin1, cov_tmin1 = carry

        t_mat = lgssm_scenario.get_transition_matrix(t[i - 1], t[i])
        t_cov_sqrt = lgssm_scenario.get_transition_covariance_sqrt(t[i - 1], t[i])
        t_cov = t_cov_sqrt @ t_cov_sqrt.T

        # Predict
        mu_t_given_tmin1 = t_mat @ mu_tmin1
        cov_t_given_tmin1 = t_mat @ cov_tmin1 @ t_mat.T + t_cov

        # Update
        l_mat = lgssm_scenario.get_likelihood_matrix(t[i])
        l_cov_sqrt = lgssm_scenario.get_likelihood_covariance_sqrt(t[i])
        l_cov = l_cov_sqrt @ l_cov_sqrt.T

        kal_gain = cov_t_given_tmin1 @ l_mat.T @ jnp.linalg.inv(l_mat @ cov_t_given_tmin1 @ l_mat.T + l_cov)
        mu_t = mu_t_given_tmin1 + kal_gain @ (y[i] - l_mat @ mu_t_given_tmin1)
        cov_t = cov_t_given_tmin1 - kal_gain @ l_mat @ cov_t_given_tmin1

        return (mu_t, cov_t), (mu_t, cov_t)

    _, (mus, covs) = scan(body_fun,
                          (mu_0, cov_0),
                          jnp.arange(1, len(t)))

    return jnp.append(mu_0[jnp.newaxis], mus, 0), jnp.append(cov_0[jnp.newaxis], covs, 0)


def run_kalman_smoother_for_marginals_and_lag_ones(lgssm_scenario: LinearGaussian,
                                                   y: jnp.ndarray,
                                                   t: jnp.ndarray,
                                                   filter_output: Tuple[jnp.ndarray, jnp.ndarray] = None) \
        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if filter_output is None:
        filter_output = run_kalman_filter_for_marginals(lgssm_scenario, y, t)

    f_mus, f_covs = filter_output

    def body_fun(carry: Tuple[jnp.ndarray, jnp.ndarray],
                 i: int) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        mu_tplus1, cov_tplus1 = carry

        t_mat = lgssm_scenario.get_transition_matrix(t[i], t[i + 1])
        t_cov_sqrt = lgssm_scenario.get_transition_covariance_sqrt(t[i], t[i + 1])
        t_cov = t_cov_sqrt @ t_cov_sqrt.T

        f_mu_t = f_mus[i]
        f_cov_t = f_covs[i]
        back_kal_gain = f_cov_t @ t_mat.T @ jnp.linalg.inv(t_mat @ f_cov_t @ t_mat.T + t_cov)
        mu_t = f_mu_t + back_kal_gain @ (mu_tplus1 - t_mat @ f_mu_t)
        cov_t = f_cov_t + back_kal_gain @ (cov_tplus1 - t_mat @ f_cov_t @ t_mat.T - t_cov) @ back_kal_gain.T

        e_xt_xtp1 = f_mu_t @ mu_tplus1.T + back_kal_gain @ (cov_tplus1 + (mu_tplus1 - f_mus[i + 1]) @ mu_tplus1.T)

        return (mu_t, cov_t), (mu_t, cov_t, e_xt_xtp1)

    _, (mus, covs, lag_one_covs) = scan(body_fun,
                                        (f_mus[-1], f_covs[-1]),
                                        jnp.arange(len(t) - 2, -1, -1))

    return jnp.append(mus[::-1], f_mus[-1, jnp.newaxis], 0), \
           jnp.append(covs[::-1], f_covs[-1, jnp.newaxis], 0), \
           lag_one_covs[::-1]


def run_kalman_smoother_for_marginals(lgssm_scenario: LinearGaussian,
                                      y: jnp.ndarray,
                                      t: jnp.ndarray,
                                      filter_output: Tuple[jnp.ndarray, jnp.ndarray] = None) \
        -> Tuple[jnp.ndarray, jnp.ndarray]:
    mus, covs, _ = run_kalman_smoother_for_marginals_and_lag_ones(lgssm_scenario, y, t, filter_output)
    return mus, covs
