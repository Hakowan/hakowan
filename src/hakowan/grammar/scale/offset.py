from dataclasses import dataclass

from .scale import Scale
from .attribute import Attribute

@dataclass(slots=True)
class Offset(Scale):
    """Offset the data by a constant.

    Attributes:
        offset (Attribute): The offset to apply to the data.
    """

    offset: Attribute
