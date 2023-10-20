from dataclasses import dataclass
from typing import Optional, Union

from ..scale import Attribute


@dataclass(kw_only=True, slots=True)
class Channel:
    pass


@dataclass(kw_only=True, slots=True)
class Position(Channel):
    data: Attribute


@dataclass(kw_only=True, slots=True)
class Normal(Channel):
    data: Attribute


@dataclass(kw_only=True, slots=True)
class Size(Channel):
    data: Union[Attribute, float]


@dataclass(kw_only=True, slots=True)
class VectorField(Channel):
    data: Attribute
