from dataclasses import dataclass, field
from typing import Optional, Self

from ..dataframe import DataFrame
from ..mark import Mark
from ..channel import Channel
from ..transform import Transform


@dataclass(kw_only=True)
class Layer:
    data: Optional[DataFrame] = None
    mark: Optional[Mark] = None
    channels: list[Channel] = field(default_factory=list)
    transform: Optional[Transform] = None
    children: list[Self] = field(default_factory=list)

    def __add__(self, other: Self) -> "Layer":
        parent = Layer()
        parent.children = [self, other]
        return parent
