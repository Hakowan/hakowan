from dataclasses import dataclass, field
from typing import Optional, Self
import copy

import lagrange

from .layer_spec import LayerSpec
from ..dataframe import DataFrame
from ..mark import Mark
from ..channel import Channel
from ..transform import Transform


@dataclass(kw_only=True)
class Layer:
    _spec: LayerSpec = field(default_factory=LayerSpec)
    _children: list[Self] = field(default_factory=list)

    def __add__(self, other: Self) -> "Layer":
        parent = Layer()
        parent._children = [self, other]
        return parent

    def data(self, data: lagrange.SurfaceMesh | DataFrame) -> "Layer":
        match (data):
            case lagrange.SurfaceMesh():
                self._spec.data = DataFrame(mesh=data)
            case DataFrame():
                self._spec.data = data
            case _:
                raise TypeError(f"Unsupported data type: {type(data)}!")
        return self

    def mark(self, mark: Mark) -> "Layer":
        self._spec.mark = mark
        return self

    def channel(self, channel: Channel) -> "Layer":
        self._spec.channels.append(channel)
        return self

    def transform(self, transform: Transform) -> "Layer":
        if self._spec.transform is None:
            self._spec.transform = copy.deepcopy(transform)
        else:
            self._spec.transform *= transform
        return self