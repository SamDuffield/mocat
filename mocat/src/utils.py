########################################################################################################################
# Module: utils.py
# Description: Some useful functions including TemporalGaussian potential evaluations and leapfrog integrator.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from functools import partial
from typing import Any, Union, Callable, Tuple
from warnings import warn

from jax import jit, numpy as jnp, vmap
from jax.lax import scan, while_loop, cond
from mocat.src.core import cdict


@partial(jit, static_argnums=(0,))
def leave_one_out_indices(n: int,
                          remove_index: int) -> jnp.ndarray:
    full_inds = jnp.arange(n + 1)
    leave_one_out_inds = jnp.where(jnp.arange(n) < remove_index, full_inds[:-1], full_inds[1:])
    return leave_one_out_inds


@jit
def _vectorised_gaussian_potential(x: jnp.ndarray,
                                   mean: Union[float, jnp.ndarray],
                                   sqrt_prec: jnp.ndarray) -> Union[float, jnp.ndarray]:
    diff = x - mean
    return 0.5 * jnp.square(diff @ sqrt_prec).sum(axis=-1)


@jit
def _mv_gaussian_potential(x: jnp.ndarray,
                           mean: Union[jnp.ndarray, float],
                           prec: jnp.ndarray) -> float:
    diff = x - mean
    return 0.5 * diff.T @ prec @ diff


@jit
def _mv_gaussian_potential_diag(x: jnp.ndarray,
                                mean: Union[jnp.ndarray, float],
                                prec_diag: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    diff = x - mean
    return 0.5 * jnp.sum(jnp.square(diff) * prec_diag, axis=-1)


def gaussian_potential(x: jnp.ndarray,
                       mean: Union[float, jnp.ndarray] = 0.,
                       prec: Union[float, jnp.ndarray] = None,
                       sqrt_prec: Union[float, jnp.ndarray] = None,
                       det_prec: float = None) -> Union[float, jnp.ndarray]:
    # sqrt_prec such that prec = sqrt_prec @ sqrt_prec.T
    x = jnp.atleast_1d(x)
    d = x.shape[-1]

    if prec is None and sqrt_prec is None:
        prec = 1.

    if isinstance(prec, float) or (prec is not None and prec.ndim == 0):
        prec = jnp.ones(d) * prec

    if isinstance(sqrt_prec, float) or (sqrt_prec is not None and sqrt_prec.ndim == 0):
        sqrt_prec = jnp.ones(d) * sqrt_prec

    if det_prec is None:
        if prec is not None and prec.ndim < 2:
            det_prec = jnp.prod(prec)
        elif sqrt_prec is not None and sqrt_prec.ndim < 2:
            det_prec = jnp.prod(sqrt_prec) ** 2

    if det_prec is None:
        # full precision matrix given but no det - computing without norm constant
        neg_log_z = 0
        warn('gaussian_potential queried with non-diagonal prec (or sqrt-prec) but no det_prec given'
             ' -> executing without normalising constant term')
    else:
        neg_log_z = (d * jnp.log(2 * jnp.pi) - jnp.log(det_prec)) / 2

    if x.ndim == 1 and sqrt_prec is None:
        # Single vals value (not vectorised)
        if prec is None:
            out_val = _mv_gaussian_potential_diag(x, mean, 1.)
        elif prec.ndim < 2:
            out_val = _mv_gaussian_potential_diag(x, mean, prec)
        else:
            out_val = _mv_gaussian_potential(x, mean, prec)
    else:
        # Multiple vals values (vectorised)
        if prec is not None and sqrt_prec is None:
            if prec.ndim < 2:
                sqrt_prec = jnp.sqrt(prec)
            else:
                sqrt_prec = jnp.linalg.cholesky(prec)
                warn('vectorised gaussian_potential queried with prec rather than sqrt_prec'
                     '-> executing using Cholesky decomp')

        if sqrt_prec is None:
            out_val = _mv_gaussian_potential_diag(x, mean, 1.)
        elif sqrt_prec.ndim < 2:
            out_val = _mv_gaussian_potential_diag(x, mean, sqrt_prec ** 2)
        else:
            out_val = _vectorised_gaussian_potential(x, mean, sqrt_prec)
    return out_val + neg_log_z


@partial(jit, static_argnums=(0, 1))
def _leapfrog(prior_potential_and_grad: Callable,
              likelihood_potential_and_grad: Callable,
              state: cdict,
              stepsize: float,
              random_keys: jnp.ndarray,
              temperature: float) -> cdict:
    leapfrog_steps = len(random_keys)

    def leapfrog_step(init_state: cdict,
                      i: int):
        new_state = init_state.copy()

        p_half = init_state.momenta - stepsize / 2. * init_state.grad_potential

        new_state.value = init_state.value + stepsize * p_half

        new_state.prior_potential, new_state.grad_prior_potential \
            = prior_potential_and_grad(new_state.value, random_keys[i])

        new_state.likelihood_potential, new_state.grad_likelihood_potential \
            = likelihood_potential_and_grad(new_state.value, random_keys[i])

        new_state.potential = new_state.prior_potential + temperature * new_state.likelihood_potential
        new_state.grad_potential = new_state.grad_prior_potential + temperature * new_state.grad_likelihood_potential

        new_state.momenta = p_half - stepsize / 2. * new_state.grad_potential

        next_sample_chain = new_state.copy()
        next_sample_chain.momenta = jnp.vstack([p_half, new_state.momenta])
        return new_state, next_sample_chain

    final_leapfrog, all_leapfrog = scan(leapfrog_step, state, jnp.arange(leapfrog_steps))

    all_leapfrog.momenta = jnp.concatenate(all_leapfrog.momenta)

    all_leapfrog = state[jnp.newaxis] + all_leapfrog

    return all_leapfrog


def leapfrog(prior_potential_and_grad: Callable,
             likelihood_potential_and_grad: Callable,
             state: cdict,
             stepsize: float,
             random_keys: jnp.ndarray,
             temperature: float = 1.) -> cdict:
    return _leapfrog(prior_potential_and_grad, likelihood_potential_and_grad,
                     state, stepsize, random_keys, temperature)


@partial(jit, static_argnums=(0, 1, 3))
def _while_loop_stacked(cond_fun: Callable,
                        body_fun: Callable,
                        init_carry: Tuple[Any, Any],
                        max_iter: int) -> Tuple[Any, Tuple[Any, Any], int]:
    # init_carry = (init_state, extra)
    # cond_func: (state, extra) -> bool
    # body_func: (state, extra) -> (state, extra)
    # while_loop_stacked -> stacked(state)

    def update_cond(previous_carry: Any,
                    final_iter: int,
                    live_iter: int) -> bool:
        return cond(live_iter <= (final_iter + 1),
                    lambda args: cond_fun(*args),
                    lambda _: False,
                    previous_carry)

    def update_kernel(previous_val_and_final_iter: Tuple[Any, int],
                      live_iter: int) -> Tuple[Tuple[Any, int], Any]:
        previous_val, final_iter = previous_val_and_final_iter

        update_bool = update_cond(previous_val, final_iter, live_iter)

        final_iter = jnp.where(update_bool, live_iter, final_iter)

        new_val = cond(update_bool,
                       lambda args: body_fun(*args),
                       lambda _: init_carry,
                       previous_val)

        return (new_val, final_iter), new_val[0]

    final_val_and_final_iter, stack = scan(update_kernel, (init_carry, 0), jnp.arange(1, max_iter + 1))
    final_val, final_final_iter = final_val_and_final_iter
    return stack, final_val, final_final_iter


def while_loop_stacked(cond_fun: Callable,
                       body_fun: Callable,
                       init_carry: Tuple[Any, Any],
                       max_iter: int = 1000) -> Any:
    full_stack, final_carry, final_int = _while_loop_stacked(cond_fun, body_fun, init_carry, max_iter)
    return tuple(a[:final_int] for a in full_stack) if isinstance(full_stack, tuple) else full_stack[:final_int]


@partial(jit, static_argnums=(0,))
def bisect(fun: Callable,
           bounds: Union[list, jnp.ndarray],
           max_iter: int = 1000,
           tol: float = 1e-5) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    evals = jnp.array([fun(bounds[0]), fun(bounds[1])])
    increasing_bool = evals[1] > evals[0]

    def conv_check(int_state: Tuple[jnp.ndarray, jnp.ndarray, int]) -> bool:
        int_bounds, int_evals, iter_ind = int_state
        return ~jnp.any(jnp.array([jnp.min(jnp.abs(int_evals)) < tol,
                                   iter_ind >= max_iter,
                                   jnp.all(int_evals < 0),
                                   jnp.all(int_evals > 0)]))

    def body_func(int_state: Tuple[jnp.ndarray, jnp.ndarray, int]) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        int_bounds, int_evals, iter_ind = int_state

        new_pos = int_bounds[0] - int_evals[0] * (int_bounds[1] - int_bounds[0]) / (int_evals[1] - int_evals[0])
        new_eval = fun(new_pos)

        replace_upper = jnp.where(increasing_bool, new_eval > 0, new_eval < 0)

        out_bounds = jnp.where(replace_upper, jnp.array([int_bounds[0], new_pos]), jnp.array([new_pos, int_bounds[1]]))
        out_evals = jnp.where(replace_upper, jnp.array([int_evals[0], new_eval]), jnp.array([new_eval, int_evals[1]]))

        return out_bounds, out_evals, iter_ind + 1

    fin_bounds, fin_evals, fin_iter = while_loop(conv_check,
                                                 body_func,
                                                 (bounds, evals, 0))

    return fin_bounds, fin_evals, fin_iter


def extract_dimension(*args):
    for a in args:
        if isinstance(a, jnp.ndarray) and a.ndim > 0:
            return a.shape[-1]
    return None


def reset_covariance(obj: Any,
                     key: str,
                     value: Any):
    prec_key = key.replace('covariance', 'precision')
    if value.ndim < 2:
        sqrt_val = jnp.sqrt(value)
        setattr(obj, key + '_sqrt', sqrt_val)
        setattr(obj, prec_key + '_sqrt', 1 / sqrt_val)
        setattr(obj, prec_key + '_det', 1 / value)
        setattr(obj, prec_key + '_mul', lambda a, b: a * b)
    else:
        sqrt_mat = jnp.linalg.cholesky(value)
        setattr(obj, key + '_sqrt', sqrt_mat)
        setattr(obj, prec_key + '_sqrt', jnp.linalg.inv(sqrt_mat))
        setattr(obj, prec_key + '_det', 1 / jnp.linalg.det(value))
        setattr(obj, prec_key + '_mul', lambda a, b: a @ b)


# @jit
# def _bfgs_sqrt_pqut(vals: jnp.ndarray,
#                     grads: jnp.ndarray,
#                     init_hessian_sqrt_diag: jnp.ndarray,
#                     r: float,
#                     update_bools: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
#     s = vals[1:] - vals[:-1]
#     y = grads[1:] - grads[:-1]
#
#     m, d = s.shape
#
#     update_bools = jnp.ones(m) * update_bools
#
#     def body_fun(val, i):
#         us, ts = val
#         sit_yi = jnp.dot(s[i], y[i])
#         sit_si = jnp.dot(s[i], s[i])
#         sit_Binit_si = jnp.square(s[i] * init_hessian_sqrt_diag).sum()
#         #
#         # thetai = (1 - r) * sit_si / \
#         #          (sit_Binit_si - jnp.dot(init_hessian_sqrt_diag * s[i], init_hessian_sqrt_diag * y[i]))
#         thetai = (sit_si - r * sit_Binit_si) / (sit_si - sit_yi)
#         ybar_i = jnp.where(sit_yi < (r * sit_Binit_si), thetai * y[i] + (1 - thetai) * s[i], y[i])
#         sit_yi = jnp.dot(s[i], ybar_i)
#
#         Ct_si = _bfgs_sqrt_transpose_prod(us, ts, s[i], init_hessian_sqrt_diag)
#         B_si = _bfgs_sqrt_prod(us, ts, Ct_si, init_hessian_sqrt_diag)
#         sit_B_si = jnp.dot(s[i], B_si)
#
#         min_denoms = jnp.minimum(jnp.abs(sit_B_si), jnp.abs(sit_yi))
#         update_bool = jnp.logical_or(r == -jnp.inf, sit_yi > 0)
#         update_bool = jnp.logical_and(min_denoms > 1e-10, update_bool)
#         update_bool = jnp.logical_and(update_bools[i], update_bool)
#
#         p = jnp.where(update_bool, s[i] / sit_yi, jnp.zeros(d))
#         q = jnp.where(update_bool, jnp.sqrt(sit_yi / sit_B_si) * B_si + ybar_i, jnp.zeros(d))
#         u = jnp.where(update_bool, jnp.sqrt(sit_B_si / sit_yi) * ybar_i + B_si, jnp.zeros(d))
#         t = jnp.where(update_bool, s[i] / sit_B_si, jnp.zeros(d))
#         return (index_update(us, i, u), index_update(ts, i, t)), (p, q, u, t)
#
#     return scan(body_fun, (jnp.zeros((m, d)), jnp.zeros((m, d))), jnp.arange(m))[1]
#
#
# def bfgs_sqrt_pqut(vals: jnp.ndarray,
#                    grads: jnp.ndarray,
#                    init_hessian_sqrt_diag: jnp.ndarray = 1.,
#                    r: float = -jnp.inf,
#                    update_bools: jnp.ndarray = 1) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
#     # Uses ps, qs for Hessian inverse approximation and us, ts for Hessian approximation
#     # 0 < r < 1 enforces positive definiteness
#     return _bfgs_sqrt_pqut(vals, grads, init_hessian_sqrt_diag, r, update_bools)


@jit
def _bfgs_sqrt_pqut(vals: jnp.ndarray,
                    grads: jnp.ndarray,
                    init_hessian_sqrt_diag: jnp.ndarray,
                    r: float,
                    update_bools: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s = vals[1:] - vals[:-1]
    y = grads[1:] - grads[:-1]

    sit_Binit_si_all = jnp.square(s * init_hessian_sqrt_diag).sum(-1)
    sty = (s * y).sum(-1)
    rats = jnp.where(sit_Binit_si_all < 1e-10, -jnp.inf, -sty/sit_Binit_si_all)
    lambd = jnp.maximum(0., rats.max() + r)
    lambd = lambd * init_hessian_sqrt_diag ** 2

    y = y + lambd * s

    m, d = s.shape

    update_bools = jnp.ones(m) * update_bools

    def body_fun(val, i):
        us, ts = val
        sit_yi = jnp.dot(s[i], y[i])
        sit_si = jnp.dot(s[i], s[i])
        sit_Binit_si = jnp.square(s[i] * init_hessian_sqrt_diag).sum()

        sit_yi = jnp.dot(s[i], y[i])

        Ct_si = _bfgs_sqrt_transpose_prod(us, ts, s[i], init_hessian_sqrt_diag)
        B_si = _bfgs_sqrt_prod(us, ts, Ct_si, init_hessian_sqrt_diag)
        sit_B_si = jnp.dot(s[i], B_si)

        min_denoms = jnp.minimum(jnp.abs(sit_B_si), jnp.abs(sit_yi))
        update_bool = jnp.logical_or(r == -jnp.inf, sit_yi > 0)
        update_bool = jnp.logical_and(min_denoms > 1e-10, update_bool)
        update_bool = jnp.logical_and(update_bools[i], update_bool)

        p = jnp.where(update_bool, s[i] / sit_yi, jnp.zeros(d))
        q = jnp.where(update_bool, jnp.sqrt(sit_yi / sit_B_si) * B_si + y[i], jnp.zeros(d))
        u = jnp.where(update_bool, jnp.sqrt(sit_B_si / sit_yi) * y[i] + B_si, jnp.zeros(d))
        t = jnp.where(update_bool, s[i] / sit_B_si, jnp.zeros(d))
        return (us.at[i].set(u), ts.at[i].set(t)), (p, q, u, t)

    return scan(body_fun, (jnp.zeros((m, d)), jnp.zeros((m, d))), jnp.arange(m))[1]


def bfgs_sqrt_pqut(vals: jnp.ndarray,
                   grads: jnp.ndarray,
                   init_hessian_sqrt_diag: jnp.ndarray = 1.,
                   r: float = -jnp.inf,
                   update_bools: jnp.ndarray = 1) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Uses ps, qs for Hessian inverse approximation and us, ts for Hessian approximation
    # 0 < r enforces positive definiteness
    return _bfgs_sqrt_pqut(vals, grads, init_hessian_sqrt_diag, r, update_bools)


@jit
def _bfgs_sqrt_prod(ps: jnp.ndarray,
                    qs: jnp.ndarray,
                    z: jnp.ndarray,
                    init_diag: jnp.ndarray) -> jnp.ndarray:
    def body_fun(ci_z, i):
        return ci_z - ps[i] * jnp.dot(qs[i], ci_z), None

    return scan(body_fun, init_diag * z, jnp.arange(len(ps)))[0]


def bfgs_sqrt_prod(ps: jnp.ndarray,
                   qs: jnp.ndarray,
                   z: jnp.ndarray,
                   init_diag: jnp.ndarray = 1.) -> jnp.ndarray:
    return _bfgs_sqrt_prod(ps, qs, z, init_diag)


@jit
def _bfgs_sqrt_transpose_prod(ps: jnp.ndarray,
                              qs: jnp.ndarray,
                              z: jnp.ndarray,
                              init_diag: jnp.ndarray) -> jnp.ndarray:
    def body_fun(ci_z, i):
        return ci_z - qs[i] * jnp.dot(ps[i], ci_z), None

    return init_diag * scan(body_fun, z, jnp.arange(len(ps) - 1, -1, -1))[0]


def bfgs_sqrt_transpose_prod(ps: jnp.ndarray,
                             qs: jnp.ndarray,
                             z: jnp.ndarray,
                             init_diag: jnp.ndarray = 1.) -> jnp.ndarray:
    return _bfgs_sqrt_transpose_prod(ps, qs, z, init_diag)


@jit
def _bfgs_sqrt_det(ps: jnp.ndarray,
                   qs: jnp.ndarray,
                   init_diag: jnp.ndarray) -> jnp.ndarray:
    return jnp.prod(init_diag) * (1 - vmap(jnp.dot)(ps, qs)).prod()


def bfgs_sqrt_det(ps: jnp.ndarray,
                  qs: jnp.ndarray,
                  init_diag: jnp.ndarray = 1.) -> jnp.ndarray:
    return _bfgs_sqrt_det(ps, qs, init_diag)


@jit
def l2_distance_matrix(vals: jnp.ndarray) -> jnp.ndarray:
    return vmap(lambda x: vmap(lambda y: jnp.sum(jnp.square(x - y)))(vals))(vals) ** 0.5


def _gc_close(z_over_rad):
    return -0.25 * z_over_rad ** 5 \
           + 0.5 * z_over_rad ** 4 \
           + 5 / 8 * z_over_rad ** 3 \
           - 5 / 3 * z_over_rad ** 2 \
           + 1


def _gc_mid(z_over_rad):
    return 1 / 12 * z_over_rad ** 5 \
           - 0.5 * z_over_rad ** 4 \
           + 5 / 8 * z_over_rad ** 3 \
           + 5 / 3 * z_over_rad ** 2 \
           - 5 * z_over_rad \
           + 4 \
           - 2 / 3 / z_over_rad


def _gc(z, radius):
    z_over_rad = jnp.abs(z) / radius
    out = jnp.zeros_like(z)
    out = jnp.where(z < radius, _gc_close(z_over_rad), out)
    out = jnp.where((z >= radius) * (z < 2 * radius), _gc_mid(z_over_rad), out)
    return out


def gc_matrix(dim: int,
              radius: float):
    return vmap(lambda i: _gc(jnp.min(jnp.array([jnp.abs(jnp.arange(dim) - i),
                                                 jnp.abs(jnp.arange(dim) + dim - i),
                                                 jnp.abs(jnp.arange(dim) - dim - i)]), axis=0), radius)
                )(jnp.arange(dim))


@jit
def kalman_gain(cov_sqrt: jnp.ndarray,
                lik_mat: jnp.ndarray,
                lik_cov_sqrt: jnp.ndarray,
                inv_mat: jnp.ndarray = None) -> jnp.ndarray:
    if inv_mat is None:
        lik_mat_cov_sqrt = lik_mat @ cov_sqrt
        inv_mat = jnp.linalg.inv(lik_mat_cov_sqrt @ lik_mat_cov_sqrt.T + lik_cov_sqrt @ lik_cov_sqrt.T)
    return cov_sqrt @ cov_sqrt.T @ lik_mat.T @ inv_mat
