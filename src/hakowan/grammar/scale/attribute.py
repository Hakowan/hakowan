from dataclasses import dataclass
from typing import TypeAlias

from .scale import ScaleLike


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
