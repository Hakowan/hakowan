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
        parent = Layer()
        match (data):
            case lagrange.SurfaceMesh():
                parent._spec.data = DataFrame(mesh=data)
            case DataFrame():
                parent._spec.data = data
            case _:
                raise TypeError(f"Unsupported data type: {type(data)}!")
        parent._children = [self]
        return parent

    def mark(self, mark: Mark) -> "Layer":
        parent = Layer()
        parent._spec.mark = mark
        parent._children = [self]
        return parent

    def channel(self, channel: Channel) -> "Layer":
        parent = Layer()
        parent._spec.channels.append(channel)
        parent._children = [self]
        return parent

    def transform(self, transform: Transform) -> "Layer":
        parent = Layer()
        parent._spec.transform = transform
        parent._children = [self]
        return parent
