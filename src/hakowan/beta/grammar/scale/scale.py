from dataclasses import dataclass
from typing import Optional, Self, Callable
from numpy import typing as npt


@dataclass(kw_only=True)
class Scale:
    child: Optional[Self] = None


@dataclass(kw_only=True)
class Normalize(Scale):
    """Normalize the data so that it fits in a bounding box defined by bbox_min and bbox_max."""

    bbox_min: npt.NDArray | float
    bbox_max: npt.NDArray | float
    _value_min: npt.NDArray | float | None = None
    _value_max: npt.NDArray | float | None = None


@dataclass(kw_only=True)
class Log(Scale):
    """Logarithmic scale."""

    base: float = 10.0


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
