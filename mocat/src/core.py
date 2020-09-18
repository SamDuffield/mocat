########################################################################################################################
# Module: core.py
# Description: mocat primitives Scenario, CDict and Sampler.
#
# Web: https://github.com/SamDuffield/mocat
########################################################################################################################

from functools import partial
from typing import Union
from pathlib import Path
import pickle
import copy

import jax.numpy as np
from jax import grad, jit
from jax.tree_util import register_pytree_node_class
from jax.util import unzip2


class Scenario:
    name = None
    dim = None

    def __init__(self, name=None, **kwargs):
        if name is not None:
            self.name = name

        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                self.__dict__[key] = value

        if self.dim == 1:
            self._potential = self.potential

            def potential(x):
                x = np.atleast_1d(x)
                x = np.reshape(x, (x.size, 1))
                return np.squeeze(self._potential(x))

            self.potential = potential

        self.potential = jit(self.potential, static_argnums=(0,))
        self.dens = jit(self.dens, static_argnums=(0,))
        self.grad_potential = jit(grad(self.potential), static_argnums=(0,))

    def __repr__(self):
        return f"mocat.Scenario.{self.__class__.__name__}({self.__dict__.__repr__()})"

    def potential(self, x: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError(f'{self.name} potential not initiated')

    def dens(self, x: np.ndarray) -> Union[float, np.ndarray]:
        return np.exp(-self.potential(x))


@register_pytree_node_class
class CDict:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def copy(self) -> 'CDict':
        return CDict(**self.__dict__)

    def __repr__(self):
        return f"mocat.CDict({self.__dict__.__repr__()})"

    def save(self,
             path: Union[str, Path],
             overwrite: bool = False):
        save_CDict(self, path, overwrite)

    def tree_flatten(self):
        return unzip2(self.__dict__.items())[::-1]

    @classmethod
    def tree_unflatten(cls, keys, xs):
        return cls(**dict(zip(keys, xs)))

    def __getitem__(self,
                    item: Union[str, int, slice, np.ndarray]) -> 'CDict':
        if isinstance(item, str):
            return self.__dict__[item]

        out_cdict = self.copy()
        for key, attr in out_cdict.__dict__.items():
            if isinstance(attr, np.ndarray):
                out_cdict.__setattr__(key, self.__dict__[key][item])
        return out_cdict

    def __add__(self,
                other: 'CDict') -> 'CDict':
        out_cdict = self.copy()
        if other is None:
            return out_cdict
        for key, attr in out_cdict.__dict__.items():
            if isinstance(attr, np.ndarray) and hasattr(other, key):
                attr_atl = attr
                other_attr_atl = other.__dict__[key]
                out_cdict.__setattr__(key, np.append(attr_atl,
                                                     other_attr_atl, axis=0))
        if hasattr(self, 'time') and hasattr(other, 'time'):
            out_cdict.time = self.time + other.time
        return out_cdict

    def keys(self):
        return self.__dict__.keys()

    def __iter__(self):
        return self.__dict__.__iter__()


def save_CDict(CDict: CDict,
               path: Union[str, Path],
               overwrite: bool = False):
    path = Path(path)
    if path.suffix != '.CDict':
        path = path.with_suffix('.CDict')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(CDict, file)


def load_CDict(path: Union[str, Path]) -> CDict:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != '.CDict':
        raise ValueError(f'Not a .CDict file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)

    # Convert to DeviceArray (default loads as np.ndarray, although I expect JAX to update this at some point)
    for attr_key, attr in data.__dict__.items():
        if isinstance(attr, np.ndarray):
            data.__dict__[attr_key] = np.asarray(attr)

    return data


class Sampler:

    def __init__(self,
                 name: str = None,
                 **kwargs):

        if not hasattr(self, 'parameters'):
            self.parameters = CDict()

        if name is not None:
            self.name = name
        elif not hasattr(self, 'name'):
            self.name = "Sampler"

        if self.parameters is not None:
            self.parameters.__dict__.update(kwargs)

    def __repr__(self):
        return f"mocat.Sampler.{self.__class__.__name__}({self.__dict__.__repr__()})"

    def copy(self) -> 'Sampler':
        return copy.deepcopy(self)

    def startup(self,
                scenario: Scenario,
                random_key: np.ndarray):
        pass
