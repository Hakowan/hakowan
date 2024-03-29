from .layer_spec import LayerSpec
from ..dataframe import DataFrame, DataFrameLike
from ..mark import Mark
from ..channel import Channel, Position, Normal, Size, VectorField, BumpMap
from ..channel.material import Material
from ..transform import Transform, Affine
from ..scale import Attribute

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Self, Sequence
import lagrange
import numpy as np
import numpy.typing as npt


@dataclass(kw_only=True, slots=True)
class Layer:
    """Layer contains the specification of data, mark, channels and transform.

    Note:
        `hakowan.layer()` method is an alias of the constructor of this class.
    """

    _spec: LayerSpec = field(default_factory=LayerSpec)
    _children: list["Layer"] = field(default_factory=list)

    def __init__(
        self,
        data: DataFrameLike | None = None,
        *,
        mark: Mark | None = None,
        channels: list[Channel] | None = None,
        transform: Transform | None = None,
    ):
        """Constructor of Layer.

        Args:
            data (DataFrameLike | None, optional): The data component of
                the layer.
            mark (Mark|None, optional): The mark component of the layer.
            channels (list[Channel], optional): The channels of the layer.
            transform (Transform, optional): The transform component of the layer.

        Returns:
            (Layer): The constructed layer object.
        """
        self._spec = LayerSpec()
        self._children = []

        if data is not None:
            self.data(data, in_place=True)
        if mark is not None:
            self.mark(mark, in_place=True)
        if transform is not None:
            self.transform(transform, in_place=True)
        if channels is not None:
            self._spec.channels = channels

    def __add__(self, other: Self) -> "Layer":
        """Combine two layers into a composite layer.

        Args:
            other (Layer): The other layer to be combined with.

        Returns:
            (Layer): The composite layer.
        """
        parent = Layer()
        parent._children = [self, other]
        return parent

    def __get_working_layer(self, in_place: bool = False) -> "Layer":
        if in_place:
            return self
        else:
            l = Layer()
            l._children = [self]
            return l

    def data(
        self,
        data: DataFrameLike,
        *,
        roi_box: npt.ArrayLike | None = None,
        in_place: bool = False,
    ) -> "Layer":
        """Overwrite the data component of this layer.

        Args:
            data (DataFrameLike): The new data component.
            roi_box (npt.ArrayLike, optional): The region of interest box of the data.
            in_place (bool, optional): Whether to modify the current layer in place or create new
                layer. Defaults to False (i.e. create a new layer).

        Returns:
            result (Layer): The layer object with data component overwritten.
        """
        l = self.__get_working_layer(in_place)
        match (data):
            case str() | Path():
                mesh = lagrange.io.load_mesh(data)  # type: ignore
                l._spec.data = DataFrame(mesh=mesh, roi_box=roi_box)
            case lagrange.SurfaceMesh():
                l._spec.data = DataFrame(mesh=data, roi_box=roi_box)
            case DataFrame():
                l._spec.data = data
                if roi_box is not None:
                    l._spec.data.roi_box = roi_box
            case _:
                raise TypeError(f"Unsupported data type: {type(data)}!")
        return l

    def mark(self, mark: Mark, *, in_place: bool = False) -> "Layer":
        """Overwrite the mark component of this layer.

        Args:
            mark (Mark): The new mark component.
            in_place (bool, optional): Whether to modify the current layer in place or create new
                layer. Defaults to False (i.e. create a new layer).

        Returns:
            result (Layer): The layer object with mark component overwritten.
        """
        l = self.__get_working_layer(in_place)
        l._spec.mark = mark
        return l

    def channel(
        self,
        *,
        position: Position | str | None = None,
        normal: Normal | str | None = None,
        size: float | str | Size | None = None,
        vector_field: VectorField | str | None = None,
        material: Material | None = None,
        bump_map: BumpMap | None = None,
        in_place: bool = False,
    ) -> "Layer":
        """Overwrite a channel component of this layer.

        Args:
            position (Position | str, optional): The new position channel.
            normal (Normal | str, optional): The new normal channel.
            size (float | str | Size, optional): The new size channel.
            vector_field (VectorField | str, optional): The new vector field channel.
            material (Material, optional): The new material channel.
            in_place (bool, optional): Whether to modify the current layer in place or create new
                layer. Defaults to False (i.e. create a new layer).

        Returns:
            result (Layer): The layer object with the channel component overwritten.
        """
        l = self.__get_working_layer(in_place)

        convert = (
            lambda value, cls: cls(data=Attribute(name=value))
            if isinstance(value, str)
            else value
        )
        if position is not None:
            assert isinstance(
                position, (Position, str)
            ), f"Unsupported position type: {type(position)}!"
            l._spec.channels.append(convert(position, Position))
        if normal is not None:
            assert isinstance(
                normal, (Normal, str)
            ), f"Unsupported normal type: {type(normal)}!"
            l._spec.channels.append(convert(normal, Normal))
        if size is not None:
            if isinstance(size, float):
                l._spec.channels.append(Size(data=size))
            else:
                assert isinstance(
                    size, (Size, str)
                ), f"Unsupported size type: {type(size)}!"
                l._spec.channels.append(convert(size, Size))
        if vector_field is not None:
            assert isinstance(
                vector_field, (VectorField, str)
            ), f"Unsupported vector_field type: {type(vector_field)}!"
            l._spec.channels.append(convert(vector_field, VectorField))
        if material is not None:
            l._spec.channels.append(material)
        if bump_map is not None:
            l._spec.channels.append(bump_map)
        return l

    def transform(self, transform: Transform, *, in_place: bool = False) -> "Layer":
        """Overwrite the transform component of this layer.

        Args:
            transform (Transform): The new transform component.
            in_place (bool, optional): Whether to modify the current layer in place or create new
                layer. Defaults to False (i.e. create a new layer).

        Returns:
            result (Layer): The layer object with transform component overwritten.
        """
        l = self.__get_working_layer(in_place)
        l._spec.transform = transform
        return l

    def rotate(
        self, axis: npt.ArrayLike, angle: float, in_place: bool = False
    ) -> "Layer":
        """Update the transform component of the current layer by applying a rotation.

        Args:
            axis (npt.ArrayLike): The unit rotation axis.
            angle (float): The rotation angle (in radians).
            in_place (bool, optional): Whether to modify the current layer in place or create new
                layer. Defaults to False (i.e. create a new layer).

        Returns:
            result (Layer): The layer object with transform component updated.
        """
        l = self.__get_working_layer(in_place)
        v = np.array(axis, dtype=np.float64)
        I = np.eye(3)
        H = np.outer(v, v)
        S = np.cross(I, v)
        M = I * np.cos(angle) + S * np.sin(angle) + H * (1 - np.cos(angle))
        if l._spec.transform is None:
            l._spec.transform = Affine(M)
        else:
            l._spec.transform *= Affine(M)
        return l

    def translate(self, offset: npt.ArrayLike, in_place: bool = False) -> "Layer":
        """Update the transform component of the current layer by applying a translation.

        Args:
            offset (npt.ArrayLike): The translation offset.
            in_place (bool, optional): Whether to modify the current layer in place or create new
                layer. Defaults to False (i.e. create a new layer).

        Returns:
            result (Layer): The layer object with transform component updated.
        """
        l = self.__get_working_layer(in_place)
        M = np.eye(4)
        M[:3, 3] = np.array(offset, dtype=np.float64)

        if l._spec.transform is None:
            l._spec.transform = Affine(M)
        else:
            l._spec.transform *= Affine(M)
        return l

    @property
    def children(self) -> list["Layer"]:
        """Get the child layers of this layer."""
        return self._children

    @children.setter
    def children(self, value: Sequence["Layer"]) -> None:
        """Set the child layers of this layer."""
        self._children = list(value)
