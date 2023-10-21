from dataclasses import dataclass
from typing import Optional, Callable
from numpy import typing as npt


@dataclass(kw_only=True, slots=True)
class Scale:
    _child: Optional["Scale"] = None


@dataclass(kw_only=True, slots=True)
class Normalize(Scale):
    """Normalize the data so that it fits in a bounding box defined by bbox_min and bbox_max."""

    range_min: npt.NDArray | float
    range_max: npt.NDArray | float
    domain_min: npt.NDArray | float | None = None
    domain_max: npt.NDArray | float | None = None


@dataclass(kw_only=True, slots=True)
class Log(Scale):
    """Logarithmic scale."""

    base: float = 10.0


@dataclass(kw_only=True, slots=True)
class Uniform(Scale):
    """Scale the data uniformly using factor."""

    factor: float


@dataclass(kw_only=True, slots=True)
class Custom(Scale):
    """Scale the data using a custom function."""

    function: Callable


@dataclass(kw_only=True, slots=True)
class Affine(Scale):
    """Scale the data using an affine transformation."""

    matrix: npt.NDArray


@dataclass(kw_only=True, slots=True)
class Clip(Scale):
    """Clip the data to the range [min, max]."""

    domain: tuple[float, float]
