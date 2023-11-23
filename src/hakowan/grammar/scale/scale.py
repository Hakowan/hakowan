import copy
from dataclasses import dataclass
from typing import Optional, Callable, TypeAlias
from numpy import typing as npt


@dataclass(kw_only=True, slots=True)
class Scale:
    """Base class for all scales."""

    _child: Optional["Scale"] = None

    def __imul__(self, other: "Scale") -> "Scale":
        """Combine the current scale with the `other` scale in place. The current scale will be applied
        before the `other` scale.
        """
        s = self
        while s._child is not None:
            s = s._child
        s._child = other
        return self

    def __mul__(self, other: "Scale") -> "Scale":
        """Combine the current scale with the `other` scale in a new scale. Both the current and
        the `other` scale is not modified. In the new scale, the current scale will be applied
        before the `other` scale.
        """
        r = copy.deepcopy(self)
        r *= other
        return r


@dataclass(slots=True)
class Normalize(Scale):
    """Normalize the data so that it fits in a bounding box defined by bbox_min and bbox_max."""

    range_min: npt.NDArray | float
    range_max: npt.NDArray | float
    domain_min: npt.NDArray | float | None = None
    domain_max: npt.NDArray | float | None = None


@dataclass(slots=True)
class Log(Scale):
    """Logarithmic scale."""

    base: float = 10.0


@dataclass(slots=True)
class Uniform(Scale):
    """Scale the data uniformly using factor."""

    factor: float


@dataclass(slots=True)
class Custom(Scale):
    """Scale the data using a custom function."""

    function: Callable


@dataclass(slots=True)
class Affine(Scale):
    """Scale the data using an affine transformation."""

    matrix: npt.NDArray


@dataclass(slots=True)
class Clip(Scale):
    """Clip the data to the range [min, max]."""

    domain: tuple[float, float]


ScaleLike: TypeAlias = float | Scale
"""Type alias for scale-like objects.

* A scalar value will be converted to `Uniform` scale with the scalar value as the factor.
* A Scale object will be unchanged.
"""
