from dataclasses import dataclass
from typing import Optional, Self, Callable

from ..scale import Attribute


@dataclass(kw_only=True)
class Transform:
    child: Optional[Self] = None


@dataclass(kw_only=True)
class Filter(Transform):
    data: Attribute
    condition: Callable
