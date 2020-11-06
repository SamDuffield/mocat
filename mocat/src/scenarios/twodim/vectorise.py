########################################################################################################################
# Module: twodim/vectorise.py
# Description: Two dimensional ssm_scenario class - vectorises potentials (that aren't already)
#              and automatically finds limits for plotting.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################


import jax.numpy as np
import matplotlib.pyplot as plt

from mocat.src.core import Scenario


def _flex_vectorise(func):
    try:
        # Check if already vectorised
        test_eval = func(np.zeros((3, 2)))
        if test_eval.shape != (3, ):
            raise TypeError
    except:
        # Attempt to vectorise
        def vec_function(arr):
            if arr.ndim == 1 and arr.shape[-1] == 2:
                return func(arr)
            elif arr.ndim == 2 and arr.shape[-1] == 2:
                return func(arr.transpose())
            elif arr.ndim == 3 and arr.shape[-1] == 2:
                return func(arr.transpose((2, 0, 1)))
            else:
                raise TypeError('Array of incorrect dimension for vectorisation')

        test_eval = vec_function(np.zeros((3, 2)))
        if test_eval.shape != (3, ):
            raise TypeError('Scenario potential vectorised successfully but failed sanity check')

        return vec_function
    else:
        return func


def _generate_plot_grid(x_lim, y_lim=None, resolution=100, linspace=False):
    if y_lim is None:
        y_lim = x_lim
    x_linspace = np.linspace(x_lim[0], x_lim[1], resolution)
    y_linspace = np.linspace(y_lim[0], y_lim[1], resolution)
    x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)
    joint_grid = np.stack([x_grid, y_grid], axis=2)

    return [x_linspace, y_linspace, joint_grid] if linspace else joint_grid


def _find_first_non_zero_row(matrix, direction=1):
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


def auto_axes_lims(vec_dens, lim=10):
    # Assumes input is a vectorised function that goes to 0 in the tails

    # Initial evaluation grid
    ix, _, grid = _generate_plot_grid([-lim, lim], resolution=100, linspace=True)
    # with np.errstate(divide='ignore'):
    #     z = vec_dens(grid)
    z = vec_dens(grid)

    # Find mode and find area with probability mass
    max_z = np.max(z)
    z_keep = z > max_z / 10

    # Find bounds of area with probability mass
    xlim = np.array([ix[_find_first_non_zero_row(z_keep.T, direction)] for direction in [1, -1]])
    ylim = np.array([ix[_find_first_non_zero_row(z_keep, direction)] for direction in [1, -1]])

    if lim in np.abs(xlim) or lim in np.abs(ylim):
        return auto_axes_lims(vec_dens, lim=2*lim)

    # Expand
    expansion = 0.05
    xlim += (xlim[1] - xlim[0]) * expansion * np.array([-1, 1])
    ylim += (ylim[1] - ylim[0]) * expansion * np.array([-1, 1])

    return tuple(xlim), tuple(ylim)


def _plot_densf(ax, x, y, z, **kwargs):
    return ax.contourf(x, y, z, **kwargs)


class TwoDimScenario(Scenario):

    dim = 2
    xlim = None
    ylim = None
    vec_dens = None
    vec_potential = None
    plot_resolution = 1000

    def _vectorise(self):
        self.vec_potential = _flex_vectorise(self.potential)
        self.vec_dens = lambda x: np.exp(-self.vec_potential(x))

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
