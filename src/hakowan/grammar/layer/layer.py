"""Immutable-by-default layer grammar and composition helpers."""

from .layer_spec import LayerSpec
from ..dataframe import DataFrameLike, PositionColumns, to_dataframe
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
from ..channel.curvestyle import CurveStyle
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
from ..transform import Transform, Affine, Clip, Compute, Filter
from ..scale import (
    Attribute,
    AttributeLike,
    Uniform as UniformScale,
    to_attribute,
    to_scale,
)
from ..texture import ScalarField, TextureLike
from ...common.color import ColorLike
from ..overlay import Annotation, Legend
import copy

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence
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
    """Specify data, marks, channels, transforms, labels, and annotations.

    ``hakowan.layer`` is an alias for this class. Fluent methods return wrapper
    layers by default, preserving the original layer for reuse.
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
        positions: PositionColumns = None,
        mark: Mark | None = None,
        channels: list[Channel] | None = None,
        transform: Transform | None = None,
        name: str | None = None,
        annotations: list[Annotation] | None = None,
    ):
        """Initialize a layer specification.

        Args:
            data: Any source supported by :func:`hakowan.dataframe.to_dataframe`.
            positions: Position columns for pandas or xarray inputs; inferred
                from ``x``, ``y``, and optional ``z`` when omitted.
            mark: Optional point, curve, or surface mark.
            channels: Initial channel specifications.
            transform: Initial transform chain.
            name: Human-readable layer label used by interactive viewers.
            annotations: Initial screen-space annotations.

        """
        self._spec = LayerSpec()
        self._children = []
        self._layout = None

        if data is not None:
            self.data(data, positions=positions, in_place=True)
        if mark is not None:
            self.mark(mark, in_place=True)
        if transform is not None:
            self.transform(transform, in_place=True)
        if channels is not None:
            self._spec.channels = channels
        if name is not None:
            self._spec.name = name
        if annotations is not None:
            self._spec.annotations = list(annotations)

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
        positions: PositionColumns = None,
        roi_box: npt.ArrayLike | None = None,
        in_place: bool = False,
    ) -> "Layer":
        """Overwrite this layer's data component.

        Args:
            data: Mesh path, SurfaceMesh, point array, pandas DataFrame,
                xarray Dataset, PyVista dataset, Trimesh object, or existing
                Hakowan DataFrame.
            positions: Position column names for pandas and xarray inputs.
                Hakowan infers ``x, y, z`` or ``x, y`` when omitted.
            roi_box: Optional region-of-interest bounds.
            in_place: Modify this layer rather than returning a copy.

        """
        layer = self.__get_working_layer(in_place)
        layer._spec.data = to_dataframe(data, positions=positions, roi_box=roi_box)
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

    def name(self, name: str, *, in_place: bool = False) -> "Layer":
        """Set a human-readable label for this layer.

        The name is surfaced as the layer's checkbox label in the interactive
        WebGL viewer (falling back to ``"Layer N"`` when unset).

        Args:
            name (str): The layer label.
            in_place (bool, optional): Whether to modify the current layer in place or create new
                layer. Defaults to False (i.e. create a new layer).

        Returns:
            result (Layer): The layer object with its name set.

        """
        layer = self.__get_working_layer(in_place)
        layer._spec.name = name
        return layer

    def annotate(
        self,
        annotation: Annotation | str,
        *,
        in_place: bool = False,
        **kwargs: Any,
    ) -> "Layer":
        """Add a screen-space text annotation.

        A string is shorthand for ``Annotation(text=annotation, **kwargs)``.
        The returned wrapper follows the same immutable-by-default behavior as
        the other layer methods.
        """
        layer = self.__get_working_layer(in_place)
        if isinstance(annotation, str):
            annotation = Annotation(text=annotation, **kwargs)
        elif kwargs:
            raise TypeError("Keyword options require a string annotation.")
        elif not isinstance(annotation, Annotation):
            raise TypeError(f"Unsupported annotation type: {type(annotation)!r}")
        layer._spec.annotations.append(annotation)
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
        """Add visual channels to this layer node.

        Each keyword names a semantic slot, so one call may set several
        independent channels. Calls are immutable by default: a new wrapper
        node is created around the current layer. During compilation, nodes are
        visited from root to leaf and the first channel for each slot wins.
        Therefore a later fluent call overrides the same slot on the wrapped
        child, while unrelated slots compose. With ``in_place=True``, channels
        append directly to the current node in keyword order and the earlier
        channel of the same kind remains effective.

        Args:
            position: Position channel or attribute reference.
            normal: Surface normal channel or attribute reference.
            size: Constant size, Size channel, or scalar attribute reference.
            shape: Point-glyph primitive or Shape channel.
            vector_field: VectorField channel or attribute name.
            covariance: Covariance channel or attribute name.
            material: Material channel.
            bump_map: BumpMap channel or texture shorthand.
            normal_map: NormalMap channel or texture shorthand.
            in_place: Append to this node instead of creating a wrapper layer.

        Returns:
            The modified node or an immutable-style wrapper layer.

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
        """Add a material channel constructed from a registered material kind.

        Args:
            type: Case-insensitive material kind: diffuse, conductor,
                rough_conductor, plastic, rough_plastic, principled,
                thin_principled, dielectric, thin_dielectric,
                rough_dielectric, or hair.
            *args: Positional arguments forwarded to the material constructor.
            in_place: Append to this node instead of returning a wrapper layer.
            **kwargs: Keyword arguments forwarded to the material constructor.

        Returns:
            The modified node or an immutable-style wrapper layer.

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

    def color_by(
        self,
        attribute: AttributeLike,
        *,
        colormap: str | list[ColorLike] = "viridis",
        domain: tuple[float, float] | None = None,
        range: tuple[float, float] | None = None,
        categories: bool = False,
        reverse: bool = False,
        legend: bool | Legend = True,
        two_sided: bool = False,
    ) -> "Layer":
        """Map a scalar attribute to diffuse color and an automatic legend.

        ``hkw.layer(data).color_by("temperature")`` is the shortest supported
        scalar-field workflow. Domain inference, the ``viridis`` colormap, and
        a semantic legend are enabled by default. The shorthand expands to a
        Diffuse material containing a ScalarField, so canonical serialization
        uses only the ordinary grammar models.
        """
        texture = ScalarField(
            data=attribute,
            colormap=colormap,
            domain=domain,
            range=range,
            categories=categories,
            reverse=reverse,
            legend=legend,
        )
        return self.channel(material=Diffuse(reflectance=texture, two_sided=two_sided))

    def show_edges(
        self,
        *,
        color: ColorLike = "black",
        width: float = 0.01,
        name: str | None = "Edges",
    ) -> "Layer":
        """Overlay mesh edges as a named curve layer of constant color and width."""
        if width <= 0.0:
            raise ValueError("Edge width must be positive")
        edges = (
            self.mark(Mark.Curve)
            .channel(size=width)
            .channel(material=Diffuse(reflectance=color))
        )
        if name is not None:
            edges = edges.name(name)
        return self + edges

    def glyph_vectors(
        self,
        attribute: AttributeLike,
        *,
        scale: float = 1.0,
        size: float = 0.01,
        color: ColorLike = "black",
        normalize: bool = False,
        end_type: Literal["point", "arrow", "flat"] = "arrow",
        refinement_level: int = 0,
        style: CurveStyle | None = None,
        overlay: bool = True,
        name: str | None = "Vectors",
    ) -> "Layer":
        """Create vector glyphs and optionally overlay them on this layer.

        ``scale`` controls glyph length, ``size`` controls thickness, and
        ``overlay=False`` returns only the generated curve-mark layer.
        """
        if scale <= 0.0 or size <= 0.0:
            raise ValueError("Vector scale and size must be positive")
        vector_attribute = copy.deepcopy(to_attribute(attribute))
        length_scale = UniformScale(factor=scale)
        vector_attribute.scale = (
            length_scale
            if vector_attribute.scale is None
            else to_scale(vector_attribute.scale) * length_scale
        )
        glyphs = (
            self.mark(Mark.Curve)
            .channel(
                vector_field=VectorField(
                    data=vector_attribute,
                    refinement_level=refinement_level,
                    style=style,
                    end_type=end_type,
                    normalize=normalize,
                ),
                size=size,
            )
            .channel(material=Diffuse(reflectance=color))
        )
        if name is not None:
            glyphs = glyphs.name(name)
        return self + glyphs if overlay else glyphs

    def slice(
        self,
        normal: npt.ArrayLike,
        *,
        offset: float = 0.0,
        point: npt.ArrayLike | None = None,
    ) -> "Layer":
        """Clip geometry to the positive side of a normalized plane.

        ``offset`` is signed distance along ``normal``. Use ``point`` instead to
        define a plane through an explicit point; the two forms are exclusive.
        """
        vector = np.asarray(normal, dtype=np.float64)
        if vector.shape != (3,) or np.linalg.norm(vector) <= 1e-12:
            raise ValueError("Slice normal must be a non-zero three-vector")
        vector /= np.linalg.norm(vector)
        if point is not None and offset != 0.0:
            raise ValueError("Specify either point or offset, not both")
        plane_point = (
            vector * float(offset)
            if point is None
            else np.asarray(point, dtype=np.float64)
        )
        if plane_point.shape != (3,):
            raise ValueError("Slice point must contain three values")
        return self.transform(Clip(point=plane_point, normal=vector))

    def isolate_component(
        self,
        component: int,
        *,
        attribute: str = "component",
        compute: bool = True,
    ) -> "Layer":
        """Keep one connected component or one existing scalar component label.

        With ``compute=True``, connected facet components are first written to
        ``attribute``. With ``compute=False``, that attribute must already exist.
        """
        if not attribute:
            raise ValueError("Component attribute name must not be empty")
        from ...spec.expression import compile_expression

        selection = Filter(
            data=attribute,
            condition=compile_expression(f"value == {int(component)}"),
        )
        transform = selection * Compute(component=attribute) if compute else selection
        return self.transform(transform)

    def compare(
        self,
        other: "Layer",
        *,
        axis: int | Literal["x", "y", "z"] = "x",
        gap: float = 0.05,
        normalize: bool = False,
        labels: tuple[str, str] | None = None,
    ) -> "Layer":
        """Juxtapose this layer and ``other`` with optional labels.

        Labels name WebGL layer controls; static backends do not draw them.
        This shorthand delegates to :meth:`juxtapose` and preserves the
        canonical layout representation.
        """
        left, right = self, other
        if labels is not None:
            if len(labels) != 2:
                raise ValueError("Comparison labels must contain exactly two values")
            left = left.name(labels[0])
            right = right.name(labels[1])
        return left.juxtapose(right, axis=axis, gap=gap, normalize=normalize)

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

    def to_spec(self, *, data_ids=None, function_ids=None):
        """Convert this layer tree to a canonical, validated specification."""
        from ...spec import to_spec

        return to_spec(self, data_ids=data_ids, function_ids=function_ids)

    def to_json(
        self,
        *,
        data_ids=None,
        function_ids=None,
        indent: int | None = 2,
        canonical: bool = False,
    ) -> str:
        """Serialize this layer tree as canonical Hakowan JSON."""
        return self.to_spec(data_ids=data_ids, function_ids=function_ids).to_json(
            indent=indent, canonical=canonical
        )

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
