from dataclasses import dataclass
from typing import Optional, Union

from ..scale import Attribute


@dataclass(kw_only=True)
class Channel:
    pass


@dataclass(kw_only=True)
class Position(Channel):
    data: Attribute


@dataclass(kw_only=True)
class Normal(Channel):
    data: Attribute


@dataclass(kw_only=True)
class Size(Channel):
    data: Union[Attribute, float]
