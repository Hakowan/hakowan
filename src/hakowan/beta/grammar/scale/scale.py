from dataclasses import dataclass
from typing import Optional, Self, Union, Callable
from numpy import typing as npt


@dataclass(kw_only=True)
class Scale:
    child: Optional[Self] = None


@dataclass(kw_only=True)
class Normalize(Scale):
    """Normalize the data so that it fits in a bounding box defined by bbox_min and bbox_max."""

    bbox_min: Union[npt.NDArray, float]
    bbox_max: Union[npt.NDArray, float]
    value_min: Union[npt.NDArray, float]
    value_max: Union[npt.NDArray, float]


@dataclass(kw_only=True)
class Log(Scale):
    """Logarithmic scale."""

    base: float


@dataclass(kw_only=True)
class Uniform(Scale):
    """Scale the data uniformly using factor."""

    factor: float


@dataclass(kw_only=True)
class Custom(Scale):
    """Scale the data using a custom function."""

    function: Callable


@dataclass(kw_only=True)
class Affine(Scale):
    """Scale the data using an affine transformation."""

    matrix: npt.NDArray
