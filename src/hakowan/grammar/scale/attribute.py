from dataclasses import dataclass
from typing import TypeAlias

from .scale import ScaleLike, Scale, Norm, to_scale


@dataclass(slots=True)
class Attribute:
    """An attribute represents a scalar or vector field that is defined on the 3D geometry.

    An attribute is the 3D equivalent of a column in a table. Each attribute is uniquely identified
    by the attribute name, which must exists in the data frame, and optionally associated with a scale.

    Attributes:
        name: The name of the attribute as it is defined in the data frame.
        scale: The scale to be applied to the attribute. `None` means no scale is used.

    Note:
        The attribute object can be constructed with `hakowan.attribute()` function, which is an
        alias of the constructor of this class.
    """

    name: str
    scale: ScaleLike | None = None

    # (internal) The name of the attribute with scale applied.
    _internal_name: str | None = None

    # (internal) The name of the attribute representing the color field mapped from the scaled attribute.
    _internal_color_field: str | None = None


AttributeLike: TypeAlias = str | Attribute
"""Type alias for a attribute-like objects.

* A string object will be converted to an attribute with the name set to the string.
* An attribute object will be unchanged.
"""


def to_attribute(value: AttributeLike) -> Attribute:
    """Coerce an attribute-like value into an `Attribute`.

    This is the single coercion point for [`AttributeLike`][hakowan.grammar.scale.attribute.AttributeLike]
    values: a string is wrapped as `Attribute(name=value)`, while an existing
    `Attribute` is returned unchanged.

    Args:
        value: A string (attribute name) or an `Attribute`.

    Returns:
        The corresponding `Attribute` object.
    """
    if isinstance(value, str):
        return Attribute(name=value)
    assert isinstance(value, Attribute), f"Cannot convert {type(value)} to Attribute"
    return value


def norm(name: str, scale: ScaleLike | None = None, order: float = 2.0) -> Attribute:
    """Construct an attribute representing the per-element magnitude of a vector field.

    This is a **shorthand** for ``Attribute(name, scale=Norm(order))`` (optionally
    chained with an extra ``scale``). It turns a vector field (e.g. a velocity or
    displacement field) into a derived scalar field, usable anywhere a scalar
    attribute is expected, such as the ``size`` channel (arrow/tube width
    proportional to magnitude) or a ``ScalarField`` texture (color by magnitude).

    The two forms below are equivalent::

        hkw.norm("velocity")
        hkw.attribute("velocity", scale=hkw.scale.Norm())

    Args:
        name: The name of the vector attribute in the data frame.
        scale: An optional scale applied *after* the norm is computed
            (e.g. ``Normalize`` to map magnitudes into a radius range, or a
            float as a uniform multiplier). ``None`` means no extra scale.
        order: The order of the norm (e.g. ``2`` for Euclidean length,
            ``1`` for Manhattan, ``numpy.inf`` for max-abs). Default ``2``.

    Returns:
        An :class:`Attribute` whose value is the row-wise norm of ``name``.

    Example:
        >>> # Color a vector field by its magnitude.
        >>> hkw.texture.ScalarField(data=hkw.norm("velocity"))
        >>> # Make tube radius proportional to magnitude.
        >>> hkw.channel.Size(data=hkw.norm("velocity",
        ...                                 scale=hkw.scale.Normalize(
        ...                                     range_min=0.005, range_max=0.02)))
    """
    norm_scale: Scale = Norm(order=order)
    if scale is not None:
        norm_scale = norm_scale * to_scale(scale)
    return Attribute(name=name, scale=norm_scale)
