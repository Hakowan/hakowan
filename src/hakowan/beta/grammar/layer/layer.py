from .layer_spec import LayerSpec
from ..dataframe import DataFrame
from ..mark import Mark
from ..channel import Channel, Position, Normal, Size, VectorField, Material
from ..transform import Transform
from ..scale import Attribute

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Self
import lagrange


@dataclass(kw_only=True, slots=True)
class Layer:
    _spec: LayerSpec = field(default_factory=LayerSpec)
    _children: list[Self] = field(default_factory=list)

    def __init__(
        self,
        *,
        data: lagrange.SurfaceMesh | DataFrame | None = None,
        mark: Mark | None = None,
        channels: list[Channel] | None = None,
        transform: Transform | None = None,
    ):
        data = DataFrame(mesh=data) if isinstance(data, lagrange.SurfaceMesh) else data
        self._children = []
        self._spec = LayerSpec(
            data=data,
            mark=mark,
            channels=channels if channels is not None else [],
            transform=transform,
        )

    def __add__(self, other: Self) -> "Layer":
        parent = Layer()
        parent._children = [self, other]
        return parent

    def data(self, data: str | Path | lagrange.SurfaceMesh | DataFrame) -> "Layer":
        parent = Layer()
        match (data):
            case str() | Path():
                mesh = lagrange.io.load_mesh(data)  # type: ignore
                parent._spec.data = DataFrame(mesh=mesh)
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

    def channel(
        self,
        *,
        position: Position | None = None,
        normal: Normal | None = None,
        size: float | str | Size | None = None,
        vector_field: VectorField | None = None,
        material: Material | None = None,
    ) -> "Layer":
        parent = Layer()
        if position is not None:
            parent._spec.channels.append(position)
        if normal is not None:
            parent._spec.channels.append(normal)
        if size is not None:
            if isinstance(size, float):
                parent._spec.channels.append(Size(data=size))
            elif isinstance(size, str):
                parent._spec.channels.append(Size(data=Attribute(name=size)))
            elif isinstance(size, Size):
                parent._spec.channels.append(size)
            else:
                raise TypeError(f"Unsupported size type: {type(size)}!")
        if vector_field is not None:
            if isinstance(vector_field, VectorField):
                parent._spec.channels.append(vector_field)
            elif isinstance(vector_field, str):
                parent._spec.channels.append(VectorField(data=Attribute(name=vector_field)))
            else:
                raise TypeError(f"Unsupported vector field type: {type(vector_field)}!")
        if material is not None:
            parent._spec.channels.append(material)
        parent._children = [self]
        return parent

    def transform(self, transform: Transform) -> "Layer":
        parent = Layer()
        parent._spec.transform = transform
        parent._children = [self]
        return parent
