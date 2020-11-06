########################################################################################################################
# Module: visualise.py
# Description: Visualisations of MCMC on two dimensional scenarios.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Union, Tuple, Type, Any
from inspect import isclass
from time import time

import jax.numpy as np
from jax import jit
from jax.lax import scan
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.contour import QuadContourSet

from mocat.src.core import CDict, Sampler
from mocat.src.scenarios.twodim.vectorise import _generate_plot_grid, TwoDimScenario
from mocat.src.mcmc.sampler import MCMCSampler
from mocat.src.mcmc.corrections import Correction, Uncorrected, Metropolis
from mocat.src.mcmc.run import startup_mcmc, mcmc_run_params, check_correction


class RunVisUtils:
    def __init__(self):
        self.plot_scen_potential = False
        self.background_samples_size = 4
        self.background_samples_alpha = 0.9
        self.proposal_size = 20
        self.max_weighted_size = 300
        self.scenario_alpha = 0.4
        self.gradient_alpha = 0.3

    def scenario_contours(self, ax, scenario, potential=False):
        return scenario.plot(ax=ax, alpha=self.scenario_alpha, zorder=0, potential=potential)

    def full_state_points(self, ax, sample):
        sample = np.atleast_2d(sample)
        if sample.ndim == 3:
            sample = np.concatenate(sample)
        return ax.scatter(sample[:, 0], sample[:, 1],
                          s=self.background_samples_size, alpha=self.background_samples_alpha, color='grey', zorder=1)

    def live_state_points(self, ax, state, weights=None):
        two_dim_state = np.atleast_2d(state)
        if weights is None:
            s = self.proposal_size
        else:
            s = weights / np.sum(weights) * self.max_weighted_size
        return ax.scatter(two_dim_state[:, 0], two_dim_state[:, 1], s=s, color='blue', zorder=4)

    @staticmethod
    def proposal_contours(ax, x, y, z):
        z = np.where(z < 1e-10, 1e-10, z)
        return ax.contour(x, y, z, colors='blue', levels=3, zorder=2)

    @staticmethod
    def remove_collection(collection):
        if isinstance(collection, QuadContourSet):
            iter_obj = collection.collections
        else:
            iter_obj = collection

        for artist in iter_obj:
            artist.remove()

    def proposed_points(self, ax, points):
        two_dim_points = np.atleast_2d(points)
        return ax.scatter(two_dim_points[:, 0], two_dim_points[:, 1], s=self.proposal_size, color='orange', zorder=3)

    @staticmethod
    def arrows(ax, start_points, end_points, color='black', alpha=1.):
        out_arrows = []
        two_dim_start_points = np.atleast_2d(start_points)
        two_dim_end_points = np.atleast_2d(end_points)

        if isinstance(alpha, float):
            alpha = np.ones(len(two_dim_start_points)) * alpha

        for i in range(len(two_dim_start_points)):
            if not np.array_equal(two_dim_start_points[i], two_dim_end_points[i]):
                out_arrows.append(ax.annotate("", xy=two_dim_end_points[i], xytext=two_dim_start_points[i],
                                              arrowprops=dict(arrowstyle='-|>', color=color, alpha=float(alpha[i])),
                                              zorder=2))
        return out_arrows

    @staticmethod
    def leapfrog_points(ax, x_points, alpha=1):
        return ax.plot(x_points[:, 0], x_points[:, 1], marker='.', color='black', zorder=2.5, alpha=alpha)

    @staticmethod
    def arrayify(cdict: CDict) -> CDict:
        out_cdict = cdict.copy()
        for key, value in out_cdict.__dict__.items():
            if isinstance(value, (float, int)):
                out_cdict.__dict__[key] = np.asarray(value)

        return out_cdict

    @staticmethod
    def vis_run_mcmc(scenario: TwoDimScenario,
                     sampler: MCMCSampler,
                     n: int,
                     correction: Correction,
                     initial_state: CDict,
                     initial_extra: CDict) -> Tuple[CDict, CDict, CDict]:

        initial_state = RunVisUtils.arrayify(initial_state)
        initial_extra = RunVisUtils.arrayify(initial_extra)

        @jit
        def markov_kernel(previous_carry: Tuple[CDict, CDict],
                          _: Any) -> Tuple[Tuple[CDict, CDict], Tuple[CDict, CDict, CDict]]:
            previous_state, previous_extra = previous_carry
            reject_state = previous_state.copy()
            reject_extra = previous_extra.copy()
            reject_extra.iter += 1

            reject_state, reject_extra = sampler.always(scenario,
                                                        reject_state,
                                                        reject_extra)

            proposed_state, proposed_extra = sampler.proposal(scenario,
                                                              reject_state,
                                                              reject_extra)

            corrected_state, corrected_extra = correction(scenario, sampler,
                                                          reject_state, reject_extra,
                                                          proposed_state, proposed_extra)

            return (corrected_state, corrected_extra), (corrected_state, corrected_extra, proposed_state)

        # Run Sampler
        start = time()
        final_carry, chain = scan(markov_kernel,
                                  (initial_state, initial_extra),
                                  None,
                                  length=n)
        end = time()

        corrected_samples, corrected_extras, proposed_samples = chain

        corrected_samples = initial_state[np.newaxis] + corrected_samples
        corrected_extras = initial_extra[np.newaxis] + corrected_extras

        if proposed_samples.value.shape[1:] == initial_state.value.shape:
            proposed_samples = initial_state[np.newaxis] + proposed_samples
        elif proposed_samples.value.shape[1:] == initial_state.value.shape[1:]:
            proposed_samples = initial_state[np.newaxis, 0] + proposed_samples
        else:
            raise ValueError("Couldn't combine initial_state and proposed_samples")

        corrected_samples.time = end - start

        return corrected_samples, corrected_extras, proposed_samples


class RunVis:

    def __init__(self,
                 ax: plt.Axes,
                 scenario: TwoDimScenario,
                 sampler: Union[Sampler, MCMCSampler],
                 correction: Correction,
                 n: int,
                 initial_state: CDict,
                 initial_extra: CDict,
                 utils: RunVisUtils = RunVisUtils()):
        self.utils = utils
        self.ax = ax
        self.scenario = scenario
        self.sampler = sampler
        self.correction = correction
        self.n = n

        self.leapfrog = hasattr(initial_state, 'momenta') and hasattr(sampler.parameters, 'leapfrog_steps')
        self.requires_gradient = hasattr(initial_state, 'grad_potential')
        self.ensemble = initial_state.value.ndim == 2

        if self.leapfrog:
            initial_state._all_leapfrog_value = np.zeros((self.sampler.parameters.leapfrog_steps + 1,
                                                          initial_state.value.shape[-1]))
            initial_state._all_leapfrog_momenta = np.zeros((2 * self.sampler.parameters.leapfrog_steps + 1,
                                                            *initial_state.momenta.shape))

        self.run_params = mcmc_run_params(sampler, correction, initial_state, initial_extra)
        self.corrected_samples, self.corrected_extras, self.proposed_samples = self.utils.vis_run_mcmc(self.scenario,
                                                                                                       self.sampler,
                                                                                                       n,
                                                                                                       self.correction,
                                                                                                       initial_state,
                                                                                                       initial_extra)

        self.samp_xlim = [np.min(self.corrected_samples.value[..., 0]), np.max(self.corrected_samples.value[..., 0])]
        self.samp_ylim = [np.min(self.corrected_samples.value[..., 1]), np.max(self.corrected_samples.value[..., 1])]

        self.reject_state = self.corrected_samples[0]
        self.reject_extra = self.corrected_extras[1]
        self.proposed_state = self.proposed_samples[1]
        self.corrected_state = self.corrected_samples[1]

        self.plot_space = None
        self.frames_per_sample = None
        self.live_frame_index = 0
        self.sample_index = 0
        self.scenario_dens = None
        self.full_state_points = None
        self.live_state_points = None

        self.scenario.auto_axes_lims()
        self.scenario.xlim = (min(self.scenario.xlim[0], self.samp_xlim[0]),
                              max(self.scenario.xlim[1], self.samp_xlim[1]))
        self.scenario.ylim = (min(self.scenario.ylim[0], self.samp_ylim[0]),
                              max(self.scenario.ylim[1], self.samp_ylim[1]))

        self.ax.set_xlim(*self.scenario.xlim)
        self.ax.set_ylim(*self.scenario.ylim)
        self.scenario_dens = self.utils.scenario_contours(self.ax, self.scenario, self.utils.plot_scen_potential)
        self.plot_space = _generate_plot_grid(self.scenario.xlim, self.scenario.ylim,
                                              self.scenario.plot_resolution, True)

    def anim_init(self):
        self.sample_index = 1
        self.live_frame_index = 0

    def __call__(self, i):

        if i == 0:
            self.full_state_points = self.utils.full_state_points(self.ax, self.reject_state.value[np.newaxis])
            self.live_state_points = self.utils.live_state_points(self.ax, self.reject_state.value)

        self.run_vis()

        if self.live_frame_index == (self.frames_per_sample - 1):

            self.live_state_points.set_offsets(self.corrected_state.value)

            self.sample_index += 1
            self.reject_state = self.corrected_samples[self.sample_index - 1]
            self.reject_extra = self.corrected_extras[self.sample_index]
            self.proposed_state = self.proposed_samples[self.sample_index]
            self.corrected_state = self.corrected_samples[self.sample_index]

            if self.ensemble:
                self.full_state_points.set_offsets(np.concatenate(self.corrected_samples.value[:self.sample_index]))
            else:
                self.full_state_points.set_offsets(self.corrected_samples.value[:self.sample_index])

        self.live_frame_index = (self.live_frame_index + 1) % self.frames_per_sample

    def run_vis(self):
        raise NotImplementedError


class UncorrectedRunVis(RunVis):

    def __init__(self,
                 ax: plt.Axes,
                 scenario: TwoDimScenario,
                 sampler: MCMCSampler,
                 correction: Correction,
                 n: int,
                 initial_state: CDict,
                 initial_extra: CDict,
                 utils: RunVisUtils = RunVisUtils()):
        super().__init__(ax, scenario, sampler, correction, n, initial_state, initial_extra, utils)
        self.frames_per_sample = 2 + self.sampler.parameters.leapfrog_steps if self.leapfrog else 2

        if self.ensemble:
            # Add new empty axis - will be called on this axis with ensemble_index > 0
            # Utilises JAX convention on array overindexing
            self.prop_pot_plot_space = CDict(value=self.plot_space[2][np.newaxis])
        else:
            self.prop_pot_plot_space = CDict(value=self.plot_space[2])

        try:
            prop_pot = self.sampler.proposal_potential(self.scenario,
                                                       self.corrected_samples[0], self.corrected_extras[0],
                                                       self.prop_pot_plot_space, self.corrected_extras[1])

            if prop_pot.shape == self.plot_space[2].shape[:2]:
                self.proposal_potential_available = True
            else:
                self.proposal_potential_available = False

        except AttributeError:
            self.proposal_potential_available = False

    def update_proposed_points(self):
        if self.proposal_potential_available:
            z_proposal_potential = self.sampler.proposal_potential(self.scenario,
                                                                   self.reject_state, self.reject_extra,
                                                                   self.prop_pot_plot_space, self.reject_extra)

            z_proposal_dens = np.exp(-z_proposal_potential)
            self.proposal_dens_contours = self.utils.proposal_contours(self.ax,
                                                                       self.plot_space[0],
                                                                       self.plot_space[1],
                                                                       z_proposal_dens)

        self.proposed_points = self.utils.proposed_points(self.ax, self.proposed_state.value)

        if not self.leapfrog and not hasattr(self.proposed_state, 'momenta'):
            self.arrows = self.utils.arrows(self.ax, self.reject_state.value, self.proposed_state.value)
            if self.requires_gradient:
                if self.proposed_state.value.ndim == 2:
                    alphas = np.where(np.arange(self.sampler.parameters.n_ensemble)
                                      == self.reject_extra.iter % self.sampler.parameters.n_ensemble,
                                      self.utils.gradient_alpha, 0.)
                else:
                    alphas = self.utils.gradient_alpha

                self.arrows = self.arrows + self.utils.arrows(self.ax, self.reject_state.value,
                                                              self.reject_state.value
                                                              - self.reject_state.grad_potential,
                                                              alpha=alphas)

    def clear_live_points(self):
        if self.leapfrog:
            self.utils.remove_collection(self.leapfrog_points)
        else:
            self.utils.remove_collection(self.arrows)
        self.proposed_points.remove()

        if self.proposal_potential_available:
            self.utils.remove_collection(self.proposal_dens_contours)

    def update_leapfrog_points(self):
        leapfrog_x = self.proposed_state._all_leapfrog_value
        leapfrog_p = self.proposed_state._all_leapfrog_momenta

        if self.live_frame_index > 0:
            self.utils.remove_collection(self.leapfrog_points)
        self.leapfrog_points = self.utils.leapfrog_points(self.ax, leapfrog_x[:self.live_frame_index + 2])

        if hasattr(self.proposed_state, 'momenta') and self.proposed_state.value.ndim == 2:
            if hasattr(self, 'arrows'):
                self.utils.remove_collection(self.arrows)

            alphas = np.where(np.arange(self.sampler.parameters.n_ensemble)
                              == self.reject_extra.iter % self.sampler.parameters.n_ensemble,
                              0., self.utils.gradient_alpha)

            self.arrows = self.utils.arrows(self.ax,
                                            self.proposed_state.value,
                                            self.proposed_state.value
                                            + self.reject_extra.parameters.stepsize * leapfrog_p[
                                                2 * self.live_frame_index + 2],
                                            alpha=alphas)

    def run_vis(self):

        if self.live_frame_index == self.frames_per_sample - 2:
            self.update_proposed_points()

        elif self.live_frame_index == self.frames_per_sample - 1:
            self.clear_live_points()

        elif self.live_frame_index >= self.frames_per_sample:
            IndexError("Visualisation live_frame_index exceeded frames_per_sample")

        else:
            self.update_leapfrog_points()


class MHRunVis(UncorrectedRunVis):

    def __init__(self,
                 ax: plt.Axes,
                 scenario: TwoDimScenario,
                 sampler: MCMCSampler,
                 correction: Correction,
                 n: int,
                 initial_state: CDict,
                 initial_extra: CDict,
                 utils: RunVisUtils = RunVisUtils()):
        super().__init__(ax, scenario, sampler, correction, n, initial_state, initial_extra, utils)
        self.frames_per_sample = 3 + self.sampler.parameters.leapfrog_steps if self.leapfrog else 3

    def run_vis(self):

        if self.live_frame_index == self.frames_per_sample - 3:
            super().update_proposed_points()

        elif self.live_frame_index == self.frames_per_sample - 2:
            if np.array_equal(self.proposed_state.value, self.corrected_state.value):
                self.proposed_points.set_color('green')
            else:
                self.proposed_points.set_color('red')

        elif self.live_frame_index == self.frames_per_sample - 1:
            super().clear_live_points()

        elif self.live_frame_index >= self.frames_per_sample:
            raise IndexError("Visualisation live_frame_index exceeded frames_per_sample")

        else:
            super().update_leapfrog_points()


def visualise(scenario: TwoDimScenario,
              sampler: MCMCSampler,
              random_key: np.ndarray,
              correction: Union[None, str, Correction, Type[Correction]] = 'sampler_default',
              initial_state: CDict = None,
              initial_extra: CDict = None,
              run_vis: Union[RunVis, Type[RunVis]] = None,
              n: int = 100,
              ms_per_sample: float = 1500,
              potential: bool = False,
              return_sample: bool = False,
              utils: RunVisUtils = RunVisUtils(),
              **kwargs):

    sampler, correction = check_correction(sampler, correction, **kwargs)
    initial_state, initial_extra = startup_mcmc(scenario, sampler, random_key, correction, initial_state, initial_extra)

    utils.plot_scen_potential = potential

    fig, ax = plt.subplots()
    plt.tight_layout()

    if run_vis is None:
        if isinstance(correction, Uncorrected):
            run_vis = UncorrectedRunVis(ax, scenario, sampler, correction, n, initial_state, initial_extra, utils)
        elif isinstance(correction, Metropolis)\
                or (hasattr(correction, 'super_correction') and isinstance(correction.super_correction, Metropolis)):
            run_vis = MHRunVis(ax, scenario, sampler, correction, n, initial_state, initial_extra, utils)
        else:
            raise ValueError(f'RunVis not found for correction: {correction}')
    elif isclass(run_vis) and issubclass(run_vis, RunVis):
        run_vis = run_vis(ax, scenario, sampler, correction, n, initial_state, initial_extra, utils)
    elif not isinstance(run_vis, RunVis):
        raise ValueError(f'run_vis {run_vis} not understood')

    anim = animation.FuncAnimation(fig,
                                   run_vis,
                                   frames=(n - 1) * run_vis.frames_per_sample,
                                   init_func=run_vis.anim_init,
                                   interval=ms_per_sample / run_vis.frames_per_sample,
                                   repeat=False,
                                   **kwargs)

    if return_sample:
        out_sample = run_vis.corrected_samples
        out_sample.run_params = run_vis.run_params
        return anim, out_sample
    else:
        return anim
