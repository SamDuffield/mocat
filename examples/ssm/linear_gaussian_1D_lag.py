# Analyse sensitivity of lag parameter for online smoothing

import matplotlib.pyplot as plt
from jax import numpy as jnp, random, jit
from scipy.interpolate import make_interp_spline
import os
from time import time
import pickle

plt.style.use('thesis')

import mocat
from mocat import ssm

# fig_dir = 'examples/images/linear_gaussian_1D'
fig_dir = os.path.dirname(os.getcwd()) + '/images/linear_gaussian_1D'


# Use splines to smooth trajectories
def smooth(t, x, k=3):
    t_new = jnp.linspace(min(t), max(t), 200)
    spl = make_interp_spline(t, x, k=k)
    y_smooth = spl(t_new)
    return t_new, y_smooth


# Initiate random seed
random_keys = random.split(random.PRNGKey(3), 100)

# Number of particles to generate
n = 10000

# Transition variance
sigma2_x = 1.

# Observation variance
sigma2_y = 1.

# Lags
lags = jnp.arange(2, 11)

# Rejections
max_rejs = 0

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


def metric(samp_arr, log_weight=None):
    if isinstance(samp_arr, mocat.cdict):
        samp_arr = samp_arr.value
    if samp_arr.ndim == 3:
        samp_arr = samp_arr[..., 0].T
    if log_weight is not None:
        samp_arr = samp_arr[random.categorical(random.PRNGKey(0), log_weight, shape=(len(samp_arr),))]
    mean = samp_arr.mean(0)
    cov = jnp.cov(samp_arr.T, ddof=1)
    return 0.5 * (jnp.trace(prec_post @ cov)
                  + (mean_post - mean).T @ prec_post @ (mean_post - mean)
                  - len_t + jnp.log(jnp.linalg.det(cov_post) / jnp.linalg.det(cov)))


# Run particle filter for marginals
pf_marginals = ssm.run_particle_filter_for_marginals(lgssm, pf, true_process.y, true_process.t, random_keys[1], n,
                                                     ess_threshold=1.)

# Run particle filter for trajectories
pft_rks = random.split(random_keys[2], len_t)
pf_trajectories = ssm.initiate_particles(lgssm, pf, n, pft_rks[0], true_process.y[0], true_process.t[0])
for i in range(1, len_t):
    pf_trajectories = ssm.propagate_particle_filter(lgssm, pf, pf_trajectories, true_process.y[i], true_process.t[i],
                                                    pft_rks[i], ess_threshold=1.)

# Backward simulation
start = time()
backsim = ssm.backward_simulation(lgssm, pf_marginals, random_keys[3], n)
backsim.value.block_until_ready()
backsim.time = time() - start

# Online smoother pf fixed lag
pf_fl_rk = random_keys[5]
pf_fl_samps = []
for lag in lags:
    pf_fl_rk, _ = random.split(pf_fl_rk)
    pf_fl_rks = random.split(pf_fl_rk, len_t)
    print(f'OPS - wPF - lag = {lag}')
    propagate_fl_pf = lambda i, parts: ssm.propagate_particle_smoother(lgssm, pf, parts, true_process.y[i],
                                                                       true_process.t[i], pf_fl_rks[i], lag, False,
                                                                       maximum_rejections=max_rejs)
    propagate_fl_pf = jit(propagate_fl_pf)
    start = time()
    pf_fl_trajectories = ssm.initiate_particles(lgssm, pf, n, pf_fl_rks[0], true_process.y[0], true_process.t[0])
    for i in range(1, len_t):
        pf_fl_trajectories = propagate_fl_pf(i, pf_fl_trajectories)
    pf_fl_trajectories.value.block_until_ready()
    pf_fl_trajectories.time = time() - start
    print(pf_fl_trajectories.time)
    pf_fl_trajectories.lag = lag
    pf_fl_trajectories.maximum_rejections = max_rejs
    pf_fl_samps.append(pf_fl_trajectories.copy())

with open(fig_dir + '/fl_pf_varylag.pkl', 'wb') as f:
    pickle.dump(pf_fl_samps, f)

#
# with open(fig_dir + '/fl_pf_varylag.pkl', 'rb') as f:
#     pf_fl_samps = pickle.load(f)


# Online smoother bs fixed lag
bs_fl_rk = random_keys[6]
bs_fl_samps = []
for lag in lags:
    bs_fl_rk, _ = random.split(bs_fl_rk)
    bs_fl_rks = random.split(bs_fl_rk, len_t)
    print(f'OPS - wBS - lag = {lag}')
    propagate_fl_bs = lambda i, parts: ssm.propagate_particle_smoother(lgssm, pf, parts, true_process.y[i],
                                                                       true_process.t[i], bs_fl_rks[i], lag, True,
                                                                       maximum_rejections=max_rejs, ess_threshold=1.)
    propagate_fl_bs = jit(propagate_fl_bs)
    start = time()
    bs_fl_trajectories = ssm.initiate_particles(lgssm, pf, n, bs_fl_rks[0], true_process.y[0], true_process.t[0])
    for i in range(1, len_t):
        bs_fl_trajectories = propagate_fl_bs(i, bs_fl_trajectories)
    bs_fl_trajectories.value.block_until_ready()
    bs_fl_trajectories.time = time() - start
    print(bs_fl_trajectories.time)
    bs_fl_trajectories.lag = lag
    bs_fl_trajectories.maximum_rejections = max_rejs
    bs_fl_samps.append(bs_fl_trajectories.copy())

with open(fig_dir + '/fl_bs_varylag.pkl', 'wb') as f:
    pickle.dump(bs_fl_samps, f)

# with open(fig_dir + '/fl_bs_varylag.pkl', 'rb') as f:
#     bs_fl_samps = pickle.load(f)

# Plot metrics
pf_joint_metric = metric(pf_trajectories, log_weight=pf_trajectories.log_weight[-1])
backsim_metric = metric(backsim)
pf_fl_metrics = jnp.array([metric(pf_fl_samps[j]) for j in range(len(lags))])
bs_fl_metrics = jnp.array([metric(bs_fl_samps[j]) for j in range(len(lags))])
fig, ax = plt.subplots()
ax.axhline(pf_joint_metric, color='purple', label='Particle Filter for smoothing')
ax.axhline(backsim_metric, color='forestgreen', label='FFBSi')
ax.plot(*smooth(lags, pf_fl_metrics, 2), color='red', label='Online Particle Smoother')
ax.plot(*smooth(lags, bs_fl_metrics, 2), color='blue', label='...with Backward Simulation')
ax.set_xlabel(r'$L$')
fig.legend(loc='upper left', bbox_to_anchor=(0.2, 0.9))
fig.savefig(fig_dir + f'/varylag', dpi=300)
