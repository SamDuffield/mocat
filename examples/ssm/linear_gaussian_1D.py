# Visualise a variety of smoothing techniques for a one-dimensional LGSSM

import matplotlib.pyplot as plt
from jax import numpy as jnp, random, jit
from scipy.interpolate import make_interp_spline
import os
from time import time

plt.style.use('thesis')

from mocat import ssm

# fig_dir = 'simulations/images/linear_gaussian_1D'
fig_dir = os.path.dirname(os.getcwd()) + '/images/linear_gaussian_1D'

# Plot params
particle_scatter_size = 0.8
particle_line_width = 0.4
particle_colour = 'orange'
particle_alpha = 0.5
particle_zorder = 0
truth_line_width = 2.
truth_colour = 'red'
truth_zorder = 3
posterior_colour = 'blue'
posterior_alpha = 0.25
ylim = (-8, 1)


# Use splines to smooth trajectories
def smooth(t, x, k=3):
    t_new = jnp.linspace(t.min(), t.max(), 200)
    spl = make_interp_spline(t, x, k=k)
    y_smooth = spl(t_new)
    return t_new, y_smooth


# Initiate random seed
random_keys = random.split(random.PRNGKey(3), 100)

# Number of particles to generate
n = 50

# Transition variance
sigma2_x = 1.

# Observation variance
sigma2_y = 1.

# Lags
lags = [2, 10]

# Rejections
max_rej = 0

# Initiate scenario
dim = 1
lgssm = ssm.TimeHomogenousLinearGaussian(initial_mean=jnp.zeros(dim),
                                         initial_covariance=jnp.eye(dim) * sigma2_x,
                                         transition_matrix=jnp.eye(dim),
                                         transition_covariance=jnp.eye(dim) * sigma2_x,
                                         likelihood_matrix=jnp.eye(dim),
                                         likelihood_covariance=jnp.eye(dim) * sigma2_y,
                                         name='1D LGSSM')

# Simulate underlying true hidden process and observations
len_t = 40
true_process = lgssm.simulate(jnp.arange(len_t, dtype='float32'), random_keys[0])

# Initiate particle filter
pf = ssm.OptimalNonLinearGaussianParticleFilter()

# Plot truth and observations
fig, ax = plt.subplots()
ax.plot(*smooth(true_process.t, true_process.x[:, 0]),
        c=truth_colour, zorder=truth_zorder, linewidth=truth_line_width)
ax.scatter(true_process.t, true_process.y[:, 0], c=truth_colour, marker='x', zorder=truth_zorder)
ax.set_xlabel(r'$t$')
ax.set_ylim(ylim)
fig.savefig(fig_dir + '/truth', dpi=300)

# Analytical posterior
mean_x = jnp.zeros(len_t)
mean_y = jnp.zeros(len_t)
cov_x = jnp.array([[min(i, j) + 1 for i in range(len_t)] for j in range(len_t)], dtype='float32')
prec_x = jnp.linalg.inv(cov_x)
cov_y = cov_x + jnp.eye(len_t)
prec_y = jnp.linalg.inv(cov_y)
mean_post = mean_x + cov_x @ prec_y @ (true_process.y[:, 0] - mean_y)
cov_post = cov_x - cov_x @ prec_y @ cov_x
prec_post = jnp.linalg.inv(cov_post)
fig, ax = plt.subplots()
smoothed_mean_post = smooth(true_process.t, mean_post)
ax.plot(*smoothed_mean_post, c=posterior_colour, zorder=particle_zorder, linewidth=truth_line_width)
margsds = jnp.sqrt(jnp.diag(cov_post))
telong, smooth_lower = smooth(true_process.t, mean_post - margsds)
smooth_upper = smooth(true_process.t, mean_post + margsds)[1]
ax.fill_between(telong, smooth_lower, smooth_upper,
                color=posterior_colour, alpha=posterior_alpha, zorder=particle_zorder)
ax.plot(*smooth(true_process.t, true_process.x[:, 0]),
        c=truth_colour, zorder=truth_zorder, linewidth=truth_line_width)
ax.scatter(true_process.t, true_process.y[:, 0], c=truth_colour, marker='x', zorder=truth_zorder)
ax.set_xlabel(r'$t$')
ax.set_ylim(ylim)
fig.savefig(fig_dir + '/posterior', dpi=300)

# Run particle filter for marginals
pf_marginals = ssm.run_particle_filter_for_marginals(lgssm, pf, true_process.y, true_process.t, random_keys[1], n,
                                                     ess_threshold=1.)
fig, ax = plt.subplots()
for i in range(n):
    ax.scatter(true_process.t, pf_marginals.value[:, i, 0], marker='o', c=particle_colour,
               alpha=particle_alpha, s=particle_scatter_size, zorder=particle_zorder)
ax.plot(*smoothed_mean_post, c=posterior_colour, zorder=truth_zorder, linewidth=truth_line_width)
ax.scatter(true_process.t, true_process.y[:, 0], c=truth_colour, marker='x', zorder=truth_zorder)
ax.set_xlabel(r'$t$')
ax.set_ylim(ylim)
fig.savefig(fig_dir + '/pf_marginals', dpi=300)

# Run particle filter for trajectories
pft_rks = random.split(random_keys[2], len_t)
pf_trajectories = ssm.initiate_particles(lgssm, pf, n, pft_rks[0], true_process.y[0], true_process.t[0])
for i in range(1, len_t):
    pf_trajectories = ssm.propagate_particle_filter(lgssm, pf, pf_trajectories, true_process.y[i], true_process.t[i],
                                                    pft_rks[i], ess_threshold=1.)
fig, ax = plt.subplots()
for i in range(n):
    ax.plot(*smooth(true_process.t, pf_trajectories.value[:, i, 0]), c=particle_colour,
            alpha=particle_alpha, linewidth=particle_line_width, zorder=particle_zorder)
ax.plot(*smoothed_mean_post, c=posterior_colour, zorder=truth_zorder, linewidth=truth_line_width)
ax.scatter(true_process.t, true_process.y[:, 0], c=truth_colour, marker='x', zorder=truth_zorder)
ax.set_xlabel(r'$t$')
ax.set_ylim(ylim)
fig.savefig(fig_dir + '/pf_joint', dpi=300)

# Backward simulation
start = time()
backsim = ssm.backward_simulation(lgssm, pf_marginals, random_keys[3], n)
backsim.value.block_until_ready()
backsim.time = time() - start
fig, ax = plt.subplots()
for i in range(n):
    ax.plot(*smooth(true_process.t, backsim.value[:, i, 0]), c=particle_colour,
            alpha=particle_alpha, linewidth=particle_line_width, zorder=particle_zorder)
ax.plot(*smoothed_mean_post, c=posterior_colour, zorder=truth_zorder, linewidth=truth_line_width)
ax.scatter(true_process.t, true_process.y[:, 0], c=truth_colour, marker='x', zorder=truth_zorder)
ax.set_xlabel(r'$t$')
ax.set_ylim(ylim)
fig.savefig(fig_dir + '/ffbsi', dpi=300)

# Marginal fixed lag
mfl_rk = random_keys[4]
for lag in lags:
    mfl_rk, _ = random.split(mfl_rk)
    mfl_rks = random.split(mfl_rk, len_t)
    vals = []
    mfl_trajectories = ssm.initiate_particles(lgssm, pf, n, mfl_rks[0], true_process.y[0], true_process.t[0])
    for i in range(1, len_t):
        mfl_trajectories = ssm.propagate_particle_filter(lgssm, pf, mfl_trajectories, true_process.y[i],
                                                         true_process.t[i], mfl_rks[i], ess_threshold=1.)
        if i - lag >= 0:
            vals.append(mfl_trajectories.value[i - lag].copy())

    marg_vals = jnp.array(vals)
    joint_vals = mfl_trajectories.value[(len_t - lag):]
    fig, ax = plt.subplots()
    for i in range(n):
        ax.scatter(true_process.t[:(len_t - lag)], marg_vals[:, i, 0], marker='o', c=particle_colour,
                   alpha=particle_alpha, s=particle_scatter_size, zorder=particle_zorder)
    for i in range(n):
        ax.plot(*smooth(true_process.t[(len_t - lag):], joint_vals[:, i, 0], k=min(3, lag - 1)), c=particle_colour,
                alpha=particle_alpha, linewidth=particle_line_width, zorder=particle_zorder)
    ax.plot(*smoothed_mean_post, c=posterior_colour, zorder=truth_zorder, linewidth=truth_line_width)
    ax.scatter(true_process.t, true_process.y[:, 0], c=truth_colour, marker='x', zorder=truth_zorder)
    ax.set_xlabel(r'$t$')
    ax.set_ylim(ylim)
    fig.savefig(fig_dir + f'/marginal_fixed_lag={lag}', dpi=300)

# Online smoother pf fixed lag
pf_fl_rk = random_keys[5]
for lag in lags:
    pf_fl_rk, _ = random.split(pf_fl_rk)
    pf_fl_rks = random.split(pf_fl_rk, len_t)
    print(f'OPS - wPF - lag = {lag}')
    propagate_fl_pf = lambda i, parts: ssm.propagate_particle_smoother(lgssm, pf, parts, true_process.y[i],
                                                                       true_process.t[i], pf_fl_rks[i], lag, False,
                                                                       maximum_rejections=max_rej)
    propagate_fl_pf = jit(propagate_fl_pf)
    start = time()
    pf_fl_trajectories = ssm.initiate_particles(lgssm, pf, n, pf_fl_rks[0], true_process.y[0], true_process.t[0])
    for i in range(1, len_t):
        pf_fl_trajectories = propagate_fl_pf(i, pf_fl_trajectories)
    pf_fl_trajectories.value.block_until_ready()
    pf_fl_trajectories.time = time() - start
    print(pf_fl_trajectories.time)

    fig, ax = plt.subplots()
    for i in range(n):
        ax.plot(*smooth(true_process.t, pf_fl_trajectories.value[:, i, 0]), c=particle_colour,
                alpha=particle_alpha, linewidth=particle_line_width, zorder=particle_zorder)
    ax.plot(*smoothed_mean_post, c=posterior_colour, zorder=truth_zorder, linewidth=truth_line_width)
    ax.scatter(true_process.t, true_process.y[:, 0], c=truth_colour, marker='x', zorder=truth_zorder)
    ax.set_xlabel(r'$t$')
    ax.set_ylim(ylim)
    fig.savefig(fig_dir + f'/online_smooth_pf_lag={lag}', dpi=300)

# Online smoother bs fixed lag
bs_fl_rk = random_keys[6]
for lag in lags:
    bs_fl_rk, _ = random.split(bs_fl_rk)
    bs_fl_rks = random.split(bs_fl_rk, len_t)
    print(f'OPS - wBS - lag = {lag}')
    propagate_fl_bs = lambda i, parts: ssm.propagate_particle_smoother(lgssm, pf, parts, true_process.y[i],
                                                                       true_process.t[i], bs_fl_rks[i], lag, True,
                                                                       maximum_rejections=max_rej, ess_threshold=1.)
    propagate_fl_bs = jit(propagate_fl_bs)
    start = time()
    bs_fl_trajectories = ssm.initiate_particles(lgssm, pf, n, bs_fl_rks[0], true_process.y[0], true_process.t[0])
    for i in range(1, len_t):
        bs_fl_trajectories = propagate_fl_bs(i, bs_fl_trajectories)
    bs_fl_trajectories.value.block_until_ready()
    bs_fl_trajectories.time = time() - start
    print(bs_fl_trajectories.time)

    fig, ax = plt.subplots()
    for i in range(n):
        ax.plot(*smooth(true_process.t, bs_fl_trajectories.value[:, i, 0]), c=particle_colour,
                alpha=particle_alpha, linewidth=particle_line_width, zorder=particle_zorder)
    ax.plot(*smoothed_mean_post, c=posterior_colour, zorder=truth_zorder, linewidth=truth_line_width)
    ax.scatter(true_process.t, true_process.y[:, 0], c=truth_colour, marker='x', zorder=truth_zorder)
    ax.set_xlabel(r'$t$')
    ax.set_ylim(ylim)
    fig.savefig(fig_dir + f'/online_smooth_bs_lag={lag}', dpi=300)
