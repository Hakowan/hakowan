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
    """Normalize the data so that the box defined by `domain_min` and `domain_max` is scaled to the box
    defined by `range_min` and `range_max`.

    Attributes:
        range_min: The minimum value of the output range.
        range_max: The maximum value of the output range.
        domain_min: The minimum value of the input range. If not specified, the minimum value of the
            input data will be used.
        domain_max: The maximum value of the input range. If not specified, the maximum value of the
            input data will be used.
    """

    range_min: npt.ArrayLike
    range_max: npt.ArrayLike
    domain_min: npt.ArrayLike | None = None
    domain_max: npt.ArrayLike | None = None


@dataclass(slots=True)
class Log(Scale):
    """Logarithmic scale.

    Attributes:
        base: The base of the logarithm.
    """

    base: float = 10.0


@dataclass(slots=True)
class Uniform(Scale):
    """Scale the data uniformly by multiplying it with a factor.

    Attributes:
        factor: The scaling factor.
    """

    factor: float


@dataclass(slots=True)
class Custom(Scale):
    """Scale the data using a custom function.

    Attributes:
        function: The scaling function. E.g. `lambda x: x ** 2` for squaring the data.
    """

    function: Callable


@dataclass(slots=True)
class Affine(Scale):
    """Scale the data using an affine transformation.

    Attributes:
        matrix: The affine transformation matrix.
    """

    matrix: npt.NDArray


@dataclass(slots=True)
class Clip(Scale):
    """Clip the data to the range [min, max].

    Attributes:
        domain: The clip minimum and maximum values.
    """

    domain: tuple[float, float]


@dataclass(slots=True)
class Norm(Scale):
    """Reduce a vector attribute to its per-element magnitude (a scalar field).

    Unlike the other scales, `Norm` changes the dimensionality of the attribute:
    an `N x d` vector field becomes an `N` scalar field holding the row-wise
    `order`-norm. It is therefore only meaningful as the *leading* scale of an
    attribute (any chained child scales operate on the resulting scalar field).
    The `hakowan.norm()` helper is the convenient shorthand for constructing an
    attribute carrying this scale.

    Attributes:
        order: The order of the norm (e.g. ``2`` for Euclidean length,
            ``1`` for Manhattan, ``numpy.inf`` for max-abs). Default ``2``.
    """

    order: float = 2.0


ScaleLike: TypeAlias = float | Scale
"""Type alias for scale-like objects.

* A scalar value will be converted to `Uniform` scale with the scalar value as the factor.
* A Scale object will be unchanged.
"""


def to_scale(value: ScaleLike) -> Scale:
    """Coerce a scale-like value into a `Scale`.

    This is the single coercion point for [`ScaleLike`][hakowan.grammar.scale.scale.ScaleLike]
    values: a scalar is wrapped as `Uniform(factor=value)`, while an existing
    `Scale` is returned unchanged.

    Args:
        value: A float/int (uniform factor) or a `Scale`.

    Returns:
        The corresponding `Scale` object.
    """
    if isinstance(value, (int, float)):
        return Uniform(factor=float(value))
    assert isinstance(value, Scale), f"Cannot convert {type(value)} to Scale"
    return value
