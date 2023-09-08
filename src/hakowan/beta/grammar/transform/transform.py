from dataclasses import dataclass
from typing import Self, Callable

from ..scale import Attribute


@dataclass(kw_only=True)
class Transform:
    child: Self | None = None


@dataclass(kw_only=True)
class Filter(Transform):
    data: Attribute
    condition: Callable
