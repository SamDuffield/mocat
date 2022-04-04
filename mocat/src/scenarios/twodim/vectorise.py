########################################################################################################################
# Module: twodim/vectorise.py
# Description: Two dimensional scenario class - vectorises potentials (that aren't already)
#              and automatically finds limits for plotting.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from typing import Callable, Union, Tuple

import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt

from mocat.src.core import Scenario


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


def _plot_densf(ax, x, y, z, **kwargs):
    return ax.contourf(x, y, z, **kwargs)


class TwoDimToyScenario(Scenario):
    dim = 2
    xlim = None
    ylim = None
    plot_resolution = 1000

    def prior_potential(self,
                        x: jnp.ndarray,
                        random_key: jnp.ndarray = None) -> float:
        # Default to uniform prior unless overwritten (surpresses warning)
        return jnp.zeros_like(jnp.array(x)[..., 0])

    def vec_potential(self,
                      xl: jnp.ndarray,
                      yl: jnp.ndarray,
                      random_key: jnp.ndarray = None):
        return vmap(lambda y: vmap(lambda x: self.potential(jnp.array([x, y]), random_key))(xl))(yl)

    def auto_axes_lims(self,
                       xlim: float = 10,
                       ylim: float = 10.):
        # Assumes self.vec_potential is a vectorised function that goes to 0 in the tails

        # Initial evaluation grid
        ix = jnp.linspace(-xlim, xlim, 100)
        iy = jnp.linspace(-ylim, ylim, 100)

        z = jnp.exp(-self.vec_potential(ix, iy))

        # Find mode
        max_z = jnp.max(z)
        if jnp.isnan(max_z):
            raise TypeError('nan found attempting auto_axes_lims, try giving manual xlim and ylim as kwargs')

        if max_z == 0.:
            return self.auto_axes_lims(xlim=xlim / 1.5, ylim=ylim / 1.5)

        # Area with probability mass
        z_keep = z > max_z / 10

        # Find bounds of area with probability mass
        xlim_new = jnp.array([ix[_find_first_non_zero_row(z_keep.T, direction)] for direction in [1, -1]])
        ylim_new = jnp.array([iy[_find_first_non_zero_row(z_keep, direction)] for direction in [1, -1]])

        if xlim in jnp.abs(xlim_new) or ylim in jnp.abs(ylim_new):
            return self.auto_axes_lims(xlim=2 * xlim if xlim in jnp.abs(xlim_new) else xlim,
                                       ylim=2 * ylim if ylim in jnp.abs(ylim_new) else ylim)

        # Expand
        expansion = 0.05
        xlim += (xlim_new[1] - xlim_new[0]) * expansion * jnp.array([-1, 1])
        ylim += (ylim_new[1] - ylim_new[0]) * expansion * jnp.array([-1, 1])
        self.xlim = tuple(xlim_new)
        self.ylim = tuple(ylim_new)

    def plot(self,
             ax=None,
             xlim=None,
             ylim=None,
             random_key: jnp.ndarray = None,
             recalc_axes=False,
             potential=False,
             cmap='Purples', **kwargs) -> Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]:

        # Check plot bounds
        if ax is not None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
        elif xlim is None or ylim is None:
            # Generate plot dimensions if not already present
            if self.xlim is None or recalc_axes:
                self.auto_axes_lims()
            xlim = self.xlim if xlim is None else xlim
            ylim = self.ylim if ylim is None else ylim

        x_linsp = jnp.linspace(*xlim, self.plot_resolution)
        y_linsp = jnp.linspace(*ylim, self.plot_resolution)
        vals_mat = self.vec_potential(x_linsp, y_linsp, random_key)
        if not potential:
            vals_mat = jnp.exp(-vals_mat)

        if ax is None:
            # Plot contours
            fig, ax = plt.subplots()
            _plot_densf(ax, x_linsp, y_linsp, vals_mat, cmap=cmap, **kwargs)
            return fig, ax
        else:
            return _plot_densf(ax, x_linsp, y_linsp, vals_mat, cmap=cmap, **kwargs)
