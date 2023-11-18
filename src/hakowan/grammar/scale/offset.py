from dataclasses import dataclass

from .scale import Scale
from .attribute import Attribute

@dataclass(slots=True)
class Offset(Scale):
    """Offset the data by a constant."""

    offset: Attribute
