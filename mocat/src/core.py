########################################################################################################################
# Module: core.py
# Description: mocat primitives Scenario, cdict and Sampler.
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
    name: str = None
    dim: int = None

    def __init__(self,
                 name: str = None,
                 **kwargs):
        if name is not None:
            self.name = name

        for key, value in kwargs.items():
            if hasattr(self, key):
                self.__dict__[key] = value

        if self.dim == 1:
            self._potential = self.potential

            def potential(x):
                x = np.atleast_1d(x)
                x = np.reshape(x, (x.size, 1))
                return np.squeeze(self._potential(x))

            self.potential = potential

        if not hasattr(self, 'grad_potential'):
            self.grad_potential = grad(self.potential)

    def __repr__(self):
        return f"mocat.Scenario.{self.__class__.__name__}({self.__dict__.__repr__()})"

    def potential(self, x: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError(f'{self.name} potential not initiated')

    def dens(self, x: np.ndarray) -> Union[float, np.ndarray]:
        return np.exp(-self.potential(x))


@register_pytree_node_class
class cdict:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def copy(self) -> 'cdict':
        return cdict(**self.__dict__)

    def deepcopy(self) -> 'cdict':
        return copy.deepcopy(self)

    def __repr__(self):
        return f"mocat.cdict({self.__dict__.__repr__()})"

    def save(self,
             path: Union[str, Path],
             overwrite: bool = False):
        save_cdict(self, path, overwrite)

    def tree_flatten(self):
        return unzip2(self.__dict__.items())[::-1]

    @classmethod
    def tree_unflatten(cls, keys, xs):
        return cls(**dict(zip(keys, xs)))

    def __getitem__(self,
                    item: Union[str, int, slice, np.ndarray]) -> 'cdict':
        if isinstance(item, str):
            return self.__dict__[item]

        out_cdict = self.copy()
        for key, attr in out_cdict.__dict__.items():
            if isinstance(attr, np.ndarray):
                out_cdict.__setattr__(key, self.__dict__[key][item])
        return out_cdict

    def __add__(self,
                other: 'cdict') -> 'cdict':
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

    @property
    def is_empty(self):
        return self.__dict__ == {}

    def keys(self):
        return self.__dict__.keys()

    def __iter__(self):
        return self.__dict__.__iter__()


def save_cdict(in_cdict: cdict,
               path: Union[str, Path],
               overwrite: bool = False):
    path = Path(path)
    if path.suffix != '.cdict':
        path = path.with_suffix('.cdict')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(in_cdict, file)


def load_cdict(path: Union[str, Path]) -> cdict:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != '.cdict':
        raise ValueError(f'Not a .cdict file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)

    # Convert to DeviceArray (default loads as np.ndarray, although I expect JAX to update this at some point)
    for attr_key, attr in data.__dict__.items():
        if isinstance(attr, np.ndarray):
            data.__dict__[attr_key] = np.asarray(attr)

    return data


class Sampler:
    parameters: cdict

    def __init__(self,
                 name: str = None,
                 **kwargs):

        if not hasattr(self, 'parameters'):
            self.parameters = cdict()

        if name is not None:
            self.name = name
        elif not hasattr(self, 'name'):
            self.name = "Sampler"

        if self.parameters is not None:
            self.parameters.__dict__.update(kwargs)

    def __repr__(self):
        return f"mocat.Sampler.{self.__class__.__name__}({self.__dict__.__repr__()})"

    def deepcopy(self) -> 'Sampler':
        return copy.deepcopy(self)

    def startup(self,
                scenario: Scenario,
                random_key: np.ndarray):
        pass
