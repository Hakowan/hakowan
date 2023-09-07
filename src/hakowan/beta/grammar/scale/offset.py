from dataclasses import dataclass

from .scale import Scale
from ..dataframe import Attribute

@dataclass(kw_only=True)
class Offset(Scale):
    """Offset the data by a constant."""

    offset: Attribute
