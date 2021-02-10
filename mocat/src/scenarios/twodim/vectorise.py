########################################################################################################################
# Module: twodim/vectorise.py
# Description: Two dimensional scenario class - vectorises potentials (that aren't already)
#              and automatically finds limits for plotting.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Callable

import jax.numpy as jnp
import matplotlib.pyplot as plt

from mocat.src.core import Scenario


def _flex_vectorise(func: Callable):
    try:
        # Check if already vectorised
        test_eval = func(jnp.zeros((3, 2)), None)
        if test_eval.shape != (3, ):
            raise TypeError
    except:
        # Attempt to vectorise
        def vec_function(arr, rk=None):
            if arr.ndim == 1 and arr.shape[-1] == 2:
                return func(arr, rk)
            elif arr.ndim == 2 and arr.shape[-1] == 2:
                return func(arr.transpose(), rk)
            elif arr.ndim == 3 and arr.shape[-1] == 2:
                return func(arr.transpose((2, 0, 1)), rk)
            else:
                raise TypeError('Array of incorrect dimension for vectorisation')

        test_eval = vec_function(jnp.zeros((3, 2)), None)
        if test_eval.shape != (3, ):
            raise TypeError('Scenario potential vectorised successfully but failed sanity check')

        return vec_function
    else:
        return func


def _generate_plot_grid(x_lim: list,
                        y_lim: list = None,
                        resolution: int = 100,
                        linspace: bool = False):
    if y_lim is None:
        y_lim = x_lim
    x_linspace = jnp.linspace(x_lim[0], x_lim[1], resolution)
    y_linspace = jnp.linspace(y_lim[0], y_lim[1], resolution)
    x_grid, y_grid = jnp.meshgrid(x_linspace, y_linspace)
    joint_grid = jnp.stack([x_grid, y_grid], axis=2)

    return [x_linspace, y_linspace, joint_grid] if linspace else joint_grid


def _find_first_non_zero_row(matrix: jnp.ndim,
                             direction: int = 1):
    if direction == 1:
        i = 0
        while not any(matrix[i]):
            i += 1
        return i
    elif direction == -1:
        i = len(matrix) - 1
        while not any(matrix[i]):
            i += -1
        return i


def auto_axes_lims(vec_dens: Callable,
                   xlim: float = 10.,
                   ylim: float = 10.):
    # Assumes ijnput is a vectorised function that goes to 0 in the tails

    # Initial evaluation grid
    ix, iy, grid = _generate_plot_grid([-xlim, xlim], [-ylim, ylim], resolution=100, linspace=True)
    z = vec_dens(grid)

    # Find mode
    max_z = jnp.max(z)
    if jnp.isnan(max_z):
        raise TypeError('nan found attempting auto_axes_lims, try giving manual xlim and ylim as kwargs')

    if max_z == 0.:
        return auto_axes_lims(vec_dens, xlim=xlim/1.5, ylim=ylim/1.5)

    # Area with probability mass
    z_keep = z > max_z / 10

    # Find bounds of area with probability mass
    xlim_new = jnp.array([ix[_find_first_non_zero_row(z_keep.T, direction)] for direction in [1, -1]])
    ylim_new = jnp.array([iy[_find_first_non_zero_row(z_keep, direction)] for direction in [1, -1]])

    if xlim in jnp.abs(xlim_new) or ylim in jnp.abs(ylim_new):
        return auto_axes_lims(vec_dens,
                              xlim=2*xlim if xlim in jnp.abs(xlim_new) else xlim,
                              ylim=2*ylim if ylim in jnp.abs(ylim_new) else ylim)

    # Expand
    expansion = 0.05
    xlim += (xlim_new[1] - xlim_new[0]) * expansion * jnp.array([-1, 1])
    ylim += (ylim_new[1] - ylim_new[0]) * expansion * jnp.array([-1, 1])

    return tuple(xlim_new), tuple(ylim_new)


def _plot_densf(ax, x, y, z, **kwargs):
    return ax.contourf(x, y, z, **kwargs)


class TwoDimToyScenario(Scenario):

    dim = 2
    xlim = None
    ylim = None
    vec_dens = None
    vec_potential = None
    plot_resolution = 1000

    def prior_potential(self,
                        x: jnp.ndarray,
                        random_key: jnp.ndarray = None) -> float:
        return 0.

    def _vectorise(self):
        self.vec_potential = _flex_vectorise(self.potential)
        self.vec_dens = lambda x, rk = None: jnp.exp(-self.vec_potential(x, rk))

    def auto_axes_lims(self, vectorise=True):
        if vectorise:
            self._vectorise()
        self.xlim , self.ylim = auto_axes_lims(self.vec_dens)

    def plot(self, ax=None, xlim=None, ylim=None, recalc_axes=False, potential=False,
             cmap='Purples', **kwargs):
        self._vectorise()

        # Check plot bounds
        if ax is not None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
        elif xlim is None or ylim is None:
            # Generate plot dimensions if not already present
            if self.xlim is None or recalc_axes:
                self.auto_axes_lims(vectorise=False)
            xlim = self.xlim if xlim is None else xlim
            ylim = self.ylim if ylim is None else ylim

        # Generate plot grid
        x_linsp, y_linsp, grid = _generate_plot_grid(xlim, ylim, self.plot_resolution, True)

        # Determine plot function
        plot_func = self.vec_potential if potential else self.vec_dens

        if ax is None:
            # Plot contours
            fig, ax = plt.subplots()
            _plot_densf(ax, x_linsp, y_linsp, plot_func(grid), cmap=cmap, **kwargs)
            return fig, ax
        else:
            return _plot_densf(ax, x_linsp, y_linsp, plot_func(grid), cmap=cmap, **kwargs)
