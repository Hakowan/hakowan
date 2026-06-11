from .layer_spec import LayerSpec
from ..dataframe import DataFrame, DataFrameLike
from ..mark import Mark
from ..channel import (
    BumpMap,
    Channel,
    Covariance,
    Normal,
    NormalMap,
    Position,
    Shape,
    Size,
    VectorField,
)
from ..channel.material import (
    Conductor,
    Dielectric,
    Diffuse,
    Hair,
    Material,
    Plastic,
    Principled,
    RoughConductor,
    RoughDielectric,
    RoughPlastic,
    ThinDielectric,
    ThinPrincipled,
)
from ..transform import Transform, Affine
from ..scale import Attribute, AttributeLike, to_attribute
from ..texture import TextureLike

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence
import lagrange
import numpy as np
import numpy.typing as npt

_MarkStr = Literal[
    "point",
    "Point",
    "POINT",
    "curve",
    "Curve",
    "CURVE",
    "surface",
    "Surface",
    "SURFACE",
]

_MaterialTypeStr = Literal[
    "diffuse",
    "Diffuse",
    "DIFFUSE",
    "conductor",
    "Conductor",
    "CONDUCTOR",
    "rough_conductor",
    "RoughConductor",
    "ROUGH_CONDUCTOR",
    "plastic",
    "Plastic",
    "PLASTIC",
    "rough_plastic",
    "RoughPlastic",
    "ROUGH_PLASTIC",
    "principled",
    "Principled",
    "PRINCIPLED",
    "thin_principled",
    "ThinPrincipled",
    "THIN_PRINCIPLED",
    "dielectric",
    "Dielectric",
    "DIELECTRIC",
    "thin_dielectric",
    "ThinDielectric",
    "THIN_DIELECTRIC",
    "rough_dielectric",
    "RoughDielectric",
    "ROUGH_DIELECTRIC",
    "hair",
    "Hair",
    "HAIR",
]

_BaseShapeStr = Literal["sphere", "disk", "cube"]


@dataclass
class LayoutOptions:
    """Parameters for a juxtaposition (``|``) layout.

    This is the single source of truth for the layout defaults; everywhere else
    just constructs or reads a :class:`LayoutOptions`.
    """

    axis: int = 0  # layout axis: 0 = x, 1 = y, 2 = z
    gap: float = 0.05  # spacing between cells, as a fraction of mean cell diameter
    normalize: bool = False  # scale each cell to equal size before placing


@dataclass(kw_only=True, slots=True)
class Layer:
    """Layer contains the specification of data, mark, channels and transform.

    Note:
        `hakowan.layer()` method is an alias of the constructor of this class.
    """

    _spec: LayerSpec = field(default_factory=LayerSpec)
    _children: list["Layer"] = field(default_factory=list)

    # Juxtaposition layout. ``None`` for a plain layer or an overlay node (``+``);
    # a :class:`LayoutOptions` for a juxtaposition node (``|`` / ``juxtapose``).
    _layout: LayoutOptions | None = None

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
        self._layout = None

        if data is not None:
            self.data(data, in_place=True)
        if mark is not None:
            self.mark(mark, in_place=True)
        if transform is not None:
            self.transform(transform, in_place=True)
        if channels is not None:
            self._spec.channels = channels

    def __add__(self, other: "Layer") -> "Layer":
        """Combine two layers into a composite layer.

        Args:
            other (Layer): The other layer to be combined with.

        Returns:
            (Layer): The composite layer.
        """
        parent = Layer()
        parent._children = [self, other]
        return parent

    def juxtapose(
        self,
        *others: "Layer",
        axis: int | Literal["x", "y", "z"] | None = None,
        gap: float | None = None,
        normalize: bool | None = None,
    ) -> "Layer":
        """Lay out this layer and ``others`` side by side for comparison.

        Unlike ``+`` (which overlays layers in the same coordinate space), this
        creates a *juxtaposition* node whose operands are translated apart along
        ``axis`` at compile time so they sit next to each other.

        Any argument left as ``None`` uses the corresponding default from
        :class:`LayoutOptions` (horizontal row, small gap, true relative scale).

        Args:
            *others (Layer): The other layer(s) to place beside this one.
            axis (int | str, optional): Layout axis, ``"x"`` / ``"y"`` / ``"z"``
                (or ``0`` / ``1`` / ``2``).
            gap (float, optional): Spacing between cells, as a fraction of the
                mean cell diameter.
            normalize (bool, optional): If ``True``, scale each cell to equal
                size before placing them; otherwise preserve true relative scale.

        Returns:
            (Layer): The composite juxtaposition layer.
        """
        if len(others) == 0:
            raise ValueError("juxtapose() requires at least one other layer.")

        options = LayoutOptions()
        if axis is not None:
            match axis:
                case "x" | 0:
                    options.axis = 0
                case "y" | 1:
                    options.axis = 1
                case "z" | 2:
                    options.axis = 2
                case _:
                    raise ValueError(f"Unsupported layout axis: {axis!r}!")
        if gap is not None:
            options.gap = float(gap)
        if normalize is not None:
            options.normalize = bool(normalize)

        parent = Layer()
        parent._children = [self, *others]
        parent._layout = options
        return parent

    def __or__(self, other: "Layer") -> "Layer":
        """Lay out two layers side by side for comparison.

        ``l1 | l2`` is shorthand for ``l1.juxtapose(l2)`` using default layout
        parameters (horizontal row, true scale).

        Args:
            other (Layer): The layer to place beside this one.

        Returns:
            (Layer): The composite juxtaposition layer.
        """
        return self.juxtapose(other)

    def __and__(self, other: "Layer") -> "Layer":
        """Lay out two layers stacked vertically for comparison.

        ``l1 & l2`` is shorthand for ``l1.juxtapose(l2, axis="y")`` — a vertical
        column, in contrast to the horizontal row produced by ``|``.

        Note:
            Python binds ``&`` tighter than ``|``, so ``a | b & c`` parses as
            ``a | (b & c)``. Parenthesise when mixing the two operators.

        Args:
            other (Layer): The layer to place below this one.

        Returns:
            (Layer): The composite juxtaposition layer.
        """
        return self.juxtapose(other, axis="y")

    def __get_working_layer(self, in_place: bool = False) -> "Layer":
        if in_place:
            return self
        else:
            layer = Layer()
            layer._children = [self]
            return layer

    def __compose_affine(self, layer: "Layer", matrix: npt.ArrayLike) -> None:
        """Pre-compose ``Affine(matrix)`` onto ``layer``'s transform in place.

        The new affine becomes the *head* of the transform chain. Because
        ``apply_transform`` evaluates the chain tail-first, this makes successive
        in-place ``translate`` / ``rotate`` / ``scale`` calls apply in call order
        — matching the non-in-place path, where each call wraps the previous
        layer and the compiler accumulates transforms root-first.
        """
        affine = Affine(matrix)
        if layer._spec.transform is None:
            layer._spec.transform = affine
        else:
            layer._spec.transform = affine * layer._spec.transform

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
        layer = self.__get_working_layer(in_place)
        match data:
            case str() | Path():
                mesh = lagrange.io.load_mesh(data, quiet=True, stitch_vertices=True)  # type: ignore
                layer._spec.data = DataFrame(mesh=mesh, roi_box=roi_box)
            case lagrange.SurfaceMesh():
                layer._spec.data = DataFrame(mesh=data, roi_box=roi_box)
            case DataFrame():
                layer._spec.data = data
                if roi_box is not None:
                    layer._spec.data.roi_box = roi_box
            case _:
                raise TypeError(f"Unsupported data type: {type(data)}!")
        return layer

    def mark(self, mark: Mark | _MarkStr, *, in_place: bool = False) -> "Layer":
        """Overwrite the mark component of this layer.

        Args:
            mark (Mark | str): The new mark component. When a string is given, accepted
                values are ``"point"`` / ``"Point"`` / ``"POINT"``,
                ``"curve"`` / ``"Curve"`` / ``"CURVE"``, and
                ``"surface"`` / ``"Surface"`` / ``"SURFACE"``.
            in_place (bool, optional): Whether to modify the current layer in place or create new
                layer. Defaults to False (i.e. create a new layer).

        Returns:
            result (Layer): The layer object with mark component overwritten.
        """
        layer = self.__get_working_layer(in_place)
        match mark:
            case Mark():
                layer._spec.mark = mark
            case "point" | "Point" | "POINT":
                layer._spec.mark = Mark.Point
            case "curve" | "Curve" | "CURVE":
                layer._spec.mark = Mark.Curve
            case "surface" | "Surface" | "SURFACE":
                layer._spec.mark = Mark.Surface
            case _:
                raise ValueError(f"Unsupported mark type: {mark}!")
        return layer

    def channel(
        self,
        *,
        position: Position | AttributeLike | None = None,
        normal: Normal | AttributeLike | None = None,
        size: float | Size | AttributeLike | None = None,
        shape: _BaseShapeStr | Shape | None = None,
        vector_field: VectorField | str | None = None,
        covariance: Covariance | str | None = None,
        material: Material | None = None,
        bump_map: BumpMap | TextureLike | None = None,
        normal_map: NormalMap | TextureLike | None = None,
        in_place: bool = False,
    ) -> "Layer":
        """Overwrite a channel component of this layer.

        Args:
            position (Position | AttributeLike, optional): The new position channel.
            normal (Normal | AttributeLike, optional): The new normal channel.
            size (float | Size | AttributeLike, optional): The new size channel.
                An ``Attribute`` (e.g. from ``hakowan.norm()``) maps a data
                field to size.
            shape (Literal["sphere", "disk", "cube"] | Shape, optional): The new shape channel.
                When a string is given, it sets ``Shape.base_shape`` directly.
            vector_field (VectorField | str, optional): The new vector field channel.
            material (Material, optional): The new material channel.
            bump_map (BumpMap | TextureLike, optional): The new bump map channel.
            normal_map (NormalMap | TextureLike, optional): The new normal map channel.
            in_place (bool, optional): Whether to modify the current layer in place or create new
                layer. Defaults to False (i.e. create a new layer).

        Returns:
            result (Layer): The layer object with the channel component overwritten.
        """
        layer = self.__get_working_layer(in_place)

        def convert(value, cls):
            if isinstance(value, (str, Attribute)):
                return cls(data=to_attribute(value))
            return value

        if position is not None:
            assert isinstance(position, (Position, str, Attribute)), (
                f"Unsupported position type: {type(position)}!"
            )
            layer._spec.channels.append(convert(position, Position))
        if normal is not None:
            assert isinstance(normal, (Normal, str, Attribute)), (
                f"Unsupported normal type: {type(normal)}!"
            )
            layer._spec.channels.append(convert(normal, Normal))
        if size is not None:
            if isinstance(size, (int, float)):
                layer._spec.channels.append(Size(data=float(size)))
            else:
                assert isinstance(size, (Size, str, Attribute)), (
                    f"Unsupported size type: {type(size)}!"
                )
                layer._spec.channels.append(convert(size, Size))
        if shape is not None:
            if isinstance(shape, str):
                layer._spec.channels.append(Shape(base_shape=shape))
            else:
                assert isinstance(shape, Shape), (
                    f"Unsupported shape type: {type(shape)}!"
                )
                layer._spec.channels.append(shape)
        if vector_field is not None:
            assert isinstance(vector_field, (VectorField, str)), (
                f"Unsupported vector_field type: {type(vector_field)}!"
            )
            layer._spec.channels.append(convert(vector_field, VectorField))
        if covariance is not None:
            assert isinstance(covariance, (Covariance, str)), (
                f"Unsupported covariance type: {type(covariance)}!"
            )
            layer._spec.channels.append(convert(covariance, Covariance))
        if material is not None:
            layer._spec.channels.append(material)
        if bump_map is not None:
            if isinstance(bump_map, BumpMap):
                layer._spec.channels.append(bump_map)
            else:
                layer._spec.channels.append(BumpMap(bump_map))
        if normal_map is not None:
            if isinstance(normal_map, NormalMap):
                layer._spec.channels.append(normal_map)
            else:
                layer._spec.channels.append(NormalMap(normal_map))
        return layer

    def material(
        self, type: _MaterialTypeStr, *args: Any, in_place: bool = False, **kwargs: Any
    ) -> "Layer":
        """Overwrite material for this layer.

        Args:
            type (str): The material type. Accepted values (case-insensitive canonical forms):
                ``"diffuse"``, ``"conductor"``, ``"rough_conductor"``, ``"plastic"``,
                ``"rough_plastic"``, ``"principled"``, ``"thin_principled"``,
                ``"dielectric"``, ``"thin_dielectric"``, ``"rough_dielectric"``,
                ``"hair"``. PascalCase (e.g. ``"RoughConductor"``) and UPPER_CASE
                (e.g. ``"ROUGH_CONDUCTOR"``) variants are also accepted.
            in_place (bool, optional): Whether to modify the current layer in place or create new
                layer. Defaults to False (i.e. create a new layer).
            *args: Variable length argument list that will be forwarded to material constructor.
            **kwargs: Arbitrary keyword arguments that will be forwarded to material constructor.

        Returns:
            result (Layer): The layer object with the channel component overwritten.
        """
        layer = self.__get_working_layer(in_place)
        match type:
            case "diffuse" | "Diffuse" | "DIFFUSE":
                layer._spec.channels.append(Diffuse(*args, **kwargs))
            case "conductor" | "Conductor" | "CONDUCTOR":
                layer._spec.channels.append(Conductor(*args, **kwargs))
            case "rough_conductor" | "RoughConductor" | "ROUGH_CONDUCTOR":
                layer._spec.channels.append(RoughConductor(*args, **kwargs))
            case "plastic" | "Plastic" | "PLASTIC":
                layer._spec.channels.append(Plastic(*args, **kwargs))
            case "rough_plastic" | "RoughPlastic" | "ROUGH_PLASTIC":
                layer._spec.channels.append(RoughPlastic(*args, **kwargs))
            case "principled" | "Principled" | "PRINCIPLED":
                layer._spec.channels.append(Principled(*args, **kwargs))
            case "thin_principled" | "ThinPrincipled" | "THIN_PRINCIPLED":
                layer._spec.channels.append(ThinPrincipled(*args, **kwargs))
            case "dielectric" | "Dielectric" | "DIELECTRIC":
                layer._spec.channels.append(Dielectric(*args, **kwargs))
            case "thin_dielectric" | "ThinDielectric" | "THIN_DIELECTRIC":
                layer._spec.channels.append(ThinDielectric(*args, **kwargs))
            case "rough_dielectric" | "RoughDielectric" | "ROUGH_DIELECTRIC":
                layer._spec.channels.append(RoughDielectric(*args, **kwargs))
            case "hair" | "Hair" | "HAIR":
                layer._spec.channels.append(Hair(*args, **kwargs))
            case _:
                raise ValueError(f"Unsupported material type: {type}!")
        return layer

    def transform(self, transform: Transform, *, in_place: bool = False) -> "Layer":
        """Overwrite the transform component of this layer.

        Args:
            transform (Transform): The new transform component.
            in_place (bool, optional): Whether to modify the current layer in place or create new
                layer. Defaults to False (i.e. create a new layer).

        Returns:
            result (Layer): The layer object with transform component overwritten.
        """
        layer = self.__get_working_layer(in_place)
        layer._spec.transform = transform
        return layer

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
        layer = self.__get_working_layer(in_place)
        v = np.array(axis, dtype=np.float64)
        eye3 = np.eye(3)
        H = np.outer(v, v)
        S = np.cross(eye3, v)
        M = eye3 * np.cos(angle) + S * np.sin(angle) + H * (1 - np.cos(angle))
        self.__compose_affine(layer, M)
        return layer

    def translate(self, offset: npt.ArrayLike, in_place: bool = False) -> "Layer":
        """Update the transform component of the current layer by applying a translation.

        Args:
            offset (npt.ArrayLike): The translation offset.
            in_place (bool, optional): Whether to modify the current layer in place or create new
                layer. Defaults to False (i.e. create a new layer).

        Returns:
            result (Layer): The layer object with transform component updated.
        """
        layer = self.__get_working_layer(in_place)
        M = np.eye(4)
        M[:3, 3] = np.array(offset, dtype=np.float64)
        self.__compose_affine(layer, M)
        return layer

    def scale(self, factor: float, in_place: bool = False) -> "Layer":
        """Update the transform component of the current layer by applying uniform scaling.

        Args:
            factor (float): The scaling factor.
            in_place (bool, optional): Whether to modify the current layer in place or create new
                layer. Defaults to False (i.e. create a new layer).

        Returns:
            result (Layer): The layer object with transform component updated.
        """
        layer = self.__get_working_layer(in_place)
        M = np.eye(4)
        M[0, 0] = M[1, 1] = M[2, 2] = factor
        self.__compose_affine(layer, M)
        return layer

    @property
    def children(self) -> list["Layer"]:
        """Get the child layers of this layer."""
        return self._children

    @children.setter
    def children(self, value: Sequence["Layer"]) -> None:
        """Set the child layers of this layer."""
        self._children = list(value)

    def _repr_html_(self) -> str:
        """Return an interactive Three.js viewer for Jupyter display.

        Requires the ``pygltflib`` package (WebGL backend).  If it is not
        installed the method falls back to a plain-text representation.
        """
        try:
            from ...backends.webgl import WebGLBackend
        except ImportError:
            return (
                "<pre>Install pygltflib for inline preview: pip install pygltflib</pre>"
            )
        try:
            from ...compiler.compile import compile as _compile
            from ...setup.config import Config

            scene = _compile(self)
            html_str = WebGLBackend().html_string(scene, Config())
        except Exception as exc:
            return f"<pre>hakowan preview error: {exc}</pre>"

        # Embed the full HTML page in an srcdoc iframe.
        # Double-quotes inside srcdoc must be entity-encoded.
        escaped = html_str.replace("&", "&amp;").replace('"', "&quot;")
        return (
            f'<iframe srcdoc="{escaped}" width="100%" height="500"'
            f' style="border:none;"></iframe>'
        )
