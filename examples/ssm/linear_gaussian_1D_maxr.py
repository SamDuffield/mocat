# Analyse sensitivity of maximum_rejections parameter (and N) for online smoothing

import matplotlib.pyplot as plt
from jax import numpy as jnp, random, jit
from scipy.interpolate import make_interp_spline
import os
from time import time
import pickle

import mocat
from mocat import ssm

fig_dir = os.path.dirname(os.getcwd()) + '/images/linear_gaussian_1D'

# Use splines to smooth trajectories
def smooth(t, x, k=3):
    t_new = jnp.linspace(t.min(), t.max(), 200)
    spl = make_interp_spline(t, x, k=k)
    y_smooth = spl(t_new)
    return t_new, y_smooth


# Initiate random seed
random_keys = random.split(random.PRNGKey(3), 100)

# Number of particles to generate
ns = [200, 400, 600, 800, 1000]

# Transition variance
sigma2_x = 1.

# Observation variance
sigma2_y = 1.

# Lag
lag = 5

# Rejections
max_rejs = [0, 1, 2, 4, 8, 16, 32]

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

# Online smoother pf fixed lag
pf_fl_rk = random_keys[5]
pf_fl_samps = []
for n_int in ns:
    pf_fl_samps_int = []
    for r in max_rejs:
        pf_fl_rk, _ = random.split(pf_fl_rk)
        pf_fl_rks = random.split(pf_fl_rk, len_t)
        print(f'OPS - wPF - N = {n_int}')
        propagate_fl_pf = lambda i, parts: ssm.propagate_particle_smoother(lgssm, pf, parts, true_process.y[i],
                                                                           true_process.t[i], pf_fl_rks[i], lag, False,
                                                                           maximum_rejections=r)
        propagate_fl_pf = jit(propagate_fl_pf)
        start = time()
        pf_fl_trajectories = ssm.initiate_particles(lgssm, pf, n_int, pf_fl_rks[0], true_process.y[0], true_process.t[0])
        for i in range(1, len_t):
            pf_fl_trajectories = propagate_fl_pf(i, pf_fl_trajectories)
        pf_fl_trajectories.value.block_until_ready()
        pf_fl_trajectories.time = time() - start
        print(pf_fl_trajectories.time)
        pf_fl_trajectories.lag = lag
        pf_fl_trajectories.maximum_rejections = max_rejs
        pf_fl_samps_int.append(pf_fl_trajectories.copy())
    pf_fl_samps.append(pf_fl_samps_int)

with open(fig_dir + '/fl_pf_varynr.pkl', 'wb') as f:
    pickle.dump(pf_fl_samps, f)

# Online smoother bs fixed lag
bs_fl_rk = random_keys[6]
bs_fl_samps = []
for n_int in ns:
    bs_fl_samps_int = []
    for r in max_rejs:
        bs_fl_rk, _ = random.split(bs_fl_rk)
        bs_fl_rks = random.split(bs_fl_rk, len_t)
        print(f'OPS - wBS - N = {n_int} - R = {r}')
        propagate_fl_bs = lambda i, parts: ssm.propagate_particle_smoother(lgssm, pf, parts, true_process.y[i],
                                                                           true_process.t[i], bs_fl_rks[i], lag, True,
                                                                           maximum_rejections=r)
        propagate_fl_bs = jit(propagate_fl_bs)
        start = time()
        bs_fl_trajectories = ssm.initiate_particles(lgssm, pf, n_int, bs_fl_rks[0], true_process.y[0], true_process.t[0])
        for i in range(1, len_t):
            bs_fl_trajectories = propagate_fl_bs(i, bs_fl_trajectories)
        bs_fl_trajectories.value.block_until_ready()
        bs_fl_trajectories.time = time() - start
        print(bs_fl_trajectories.time)
        bs_fl_trajectories.lag = lag
        bs_fl_trajectories.maximum_rejections = max_rejs
        bs_fl_samps_int.append(bs_fl_trajectories.copy())
    bs_fl_samps.append(bs_fl_samps_int)

with open(fig_dir + '/fl_bs_varynr.pkl', 'wb') as f:
    pickle.dump(bs_fl_samps, f)

# Plot num_transition_evals
pf_fl_nte = jnp.array(
    [[pf_fl_samps[j][i].num_transition_evals.mean() for i in range(len(max_rejs))] for j in range(len(ns))]) / 1e6
bs_fl_nte = jnp.array(
    [[bs_fl_samps[j][i].num_transition_evals.mean() for i in range(len(max_rejs))] for j in range(len(ns))]) / 1e6

line_styles = ['-', '--', ':', '-.', (0, (10, 10))]
ymax_pf = 0.4
fig_pf, ax_pf = plt.subplots()
for i in range(len(ns)):
    ax_pf.plot(max_rejs, pf_fl_nte[i], color='red', label=f'{ns[i]}', linestyle=line_styles[i])
ax_pf.set_xlabel(r'$R$')
ax_pf.set_ylabel(r'Number of transition density evaluations (millions)')
ax_pf.set_ylim(0, ymax_pf)
handles, labels = ax_pf.get_legend_handles_labels()
ax_pf.legend(handles[::-1], labels[::-1], title=r'$N$')
fig_pf.savefig(fig_dir + f'/pf_fl_num_transition_evals_nr', dpi=300)

fig_bs, ax_bs = plt.subplots()
ymax_bs = 2.0
for i in range(len(ns)):
    ax_bs.plot(max_rejs, bs_fl_nte[i], color='blue', label=f'{ns[i]}', linestyle=line_styles[i])
ax_bs.set_xlabel(r'$R$')
ax_bs.set_ylabel(r'Number of transition density evaluations (millions)')
ax_bs.set_ylim(0, ymax_bs)
handles, labels = ax_bs.get_legend_handles_labels()
ax_bs.legend(handles[::-1], labels[::-1], title=r'$N$')
fig_bs.savefig(fig_dir + f'/bs_fl_num_transition_evals_nr', dpi=300)


