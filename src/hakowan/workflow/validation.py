"""Static, structured validation for Hakowan layer specifications."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import lagrange
import numpy as np

from ..backends import BackendCapabilities, BackendName, get_backend_capabilities
from ..grammar.channel import (
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
from ..grammar.channel.material import Hair, Material
from ..grammar.dataframe import DataFrame
from ..grammar.layer import Layer
from ..grammar.mark import Mark
from ..grammar.scale import Attribute, Custom, Log, Norm as NormScale, Scale
from ..grammar.texture import Checkerboard, Image, Isocontour, ScalarField, Texture
from ..grammar.transform import (
    Boundary,
    Compute,
    Explode,
    Filter,
    Fur,
    Norm as NormTransform,
    Streamline,
    Transform,
    UVMesh,
)


Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One actionable validation finding."""

    code: str
    severity: Severity
    path: str
    message: str
    hint: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Return this diagnostic as a JSON-safe mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Validation result for a layer and target backend."""

    backend: str
    strict: bool
    diagnostics: tuple[Diagnostic, ...]

    @property
    def valid(self) -> bool:
        """Return whether the report contains no error diagnostics."""
        return not any(item.severity == "error" for item in self.diagnostics)

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        """Return error diagnostics in their original order."""
        return tuple(item for item in self.diagnostics if item.severity == "error")

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        """Return warning diagnostics in their original order."""
        return tuple(item for item in self.diagnostics if item.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report including the derived validity flag."""
        return {
            "backend": self.backend,
            "strict": self.strict,
            "valid": self.valid,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    def raise_for_errors(self) -> None:
        """Raise :class:`ValidationError` when the report is invalid."""
        if not self.valid:
            raise ValidationError(self)


class ValidationError(ValueError):
    """Raised by :meth:`ValidationReport.raise_for_errors`."""

    def __init__(self, report: ValidationReport):
        """Initialize the exception from a structured validation report."""
        self.report = report
        detail = "; ".join(f"{item.path}: {item.message}" for item in report.errors)
        super().__init__(detail or "Hakowan validation failed")


@dataclass(slots=True)
class _ResolvedView:
    data: DataFrame | None
    mark: Mark
    channels: list[Channel]
    transforms: list[Transform]


@dataclass(frozen=True, slots=True)
class _AttributeInfo:
    channels: int
    element: str
    minimum: float | None = None
    has_nonfinite: bool = False


_CHANNEL_NAMES: dict[type[Channel], str] = {
    Position: "position",
    Normal: "normal",
    Size: "size",
    Shape: "shape",
    VectorField: "vector_field",
    Covariance: "covariance",
    Material: "material",
    BumpMap: "bump_map",
    NormalMap: "normal_map",
}

_MARK_CHANNELS: dict[type[Channel], frozenset[Mark]] = {
    Normal: frozenset({Mark.Surface}),
    Size: frozenset({Mark.Point, Mark.Curve}),
    Shape: frozenset({Mark.Point}),
    VectorField: frozenset({Mark.Curve}),
    Covariance: frozenset({Mark.Point}),
    BumpMap: frozenset({Mark.Surface}),
    NormalMap: frozenset({Mark.Surface}),
}


def _channel_name(channel: Channel) -> str:
    for cls, name in _CHANNEL_NAMES.items():
        if isinstance(channel, cls):
            return name
    return type(channel).__name__.lower()


def _flatten_views(root: Layer) -> list[_ResolvedView]:
    resolved: list[_ResolvedView] = []

    def visit(node: Layer, ancestors: list[Layer]) -> None:
        ancestors.append(node)
        if node._children:
            for child in node._children:
                visit(child, ancestors)
        else:
            data = None
            mark = None
            channels: list[Channel] = []
            transforms: list[Transform] = []
            for ancestor in ancestors:
                if data is None and ancestor._spec.data is not None:
                    data = ancestor._spec.data
                if mark is None and ancestor._spec.mark is not None:
                    mark = ancestor._spec.mark
                channels.extend(ancestor._spec.channels)
                if ancestor._spec.transform is not None:
                    transforms.append(ancestor._spec.transform)
            if mark is None:
                mark = (
                    Mark.Point
                    if data is not None and data.mesh.num_facets == 0
                    else Mark.Surface
                )
            resolved.append(
                _ResolvedView(
                    data=data,
                    mark=mark,
                    channels=channels,
                    transforms=transforms,
                )
            )
        ancestors.pop()

    visit(root, [])
    return resolved


def _transform_nodes(transform: Transform) -> list[Transform]:
    result: list[Transform] = []
    current: Transform | None = transform
    while current is not None:
        result.append(current)
        current = current._child
    return result


def _world_points(view: Any) -> np.ndarray:
    if view.data_frame is None or view.data_frame.mesh.num_vertices == 0:
        return np.empty((0, 3), dtype=np.float64)
    points = np.asarray(view.data_frame.mesh.vertices, dtype=np.float64)
    transform = np.asarray(view.global_transform, dtype=np.float64)
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def _camera_coordinates(
    points: np.ndarray, camera: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eye = np.asarray(camera.eye, dtype=np.float64)
    forward = np.asarray(camera.target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    up = np.asarray(camera.up, dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    relative = points - eye
    return relative @ right, relative @ camera_up, relative @ forward


def _perspective_tangents(camera: Any, width: int, height: int) -> tuple[float, float]:
    tangent = float(np.tan(np.radians(camera.fov) / 2.0))
    aspect = width / height
    axis = camera.fov_axis
    if (
        axis == "x"
        or (axis == "smaller" and width <= height)
        or (axis == "larger" and width >= height)
    ):
        return tangent, tangent / aspect
    if axis == "diagonal":
        denominator = float(np.sqrt(aspect * aspect + 1.0))
        return tangent * aspect / denominator, tangent / denominator
    return tangent * aspect, tangent


def _projected_bounds(
    points: np.ndarray, camera: Any
) -> tuple[float, float, float, float] | None:
    from ..grammar.figure import OrthographicCamera

    x, y, depth = _camera_coordinates(points, camera)
    visible = (depth >= camera.near) & (depth <= camera.far)
    if not np.any(visible):
        return None
    if not isinstance(camera, OrthographicCamera):
        x = x[visible] / depth[visible]
        y = y[visible] / depth[visible]
    else:
        x = x[visible]
        y = y[visible]
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def _rect_coverage(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
) -> float:
    inner_area = max(inner[1] - inner[0], 1e-12) * max(inner[3] - inner[2], 1e-12)
    intersection = max(0.0, min(outer[1], inner[1]) - max(outer[0], inner[0])) * max(
        0.0, min(outer[3], inner[3]) - max(outer[2], inner[2])
    )
    return intersection / inner_area


def _validate_compiled_scene(scene: Any, figure: Any, validator: "_Validator") -> None:
    for index, view in enumerate(scene):
        assert view.data_frame is not None
        mesh = view.data_frame.mesh
        empty = mesh.num_vertices == 0 or (
            view.mark is Mark.Surface and mesh.num_facets == 0
        )
        if empty:
            validator.issue(
                "geometry.empty_after_transform",
                f"views[{index}].geometry",
                "The compiled view contains no renderable geometry.",
                hint=(
                    "Inspect filter, clip, and selection transforms; relax their conditions "
                    "or verify that the source data contains elements for this mark."
                ),
            )

    if figure is None or figure.scene.camera is None:
        return
    camera = figure.scene.camera
    view_points = [_world_points(view) for view in scene]
    nonempty = [points for points in view_points if points.size]
    if not nonempty:
        return
    points = np.concatenate(nonempty, axis=0)
    x, y, depth = _camera_coordinates(points, camera)
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    center = (bounds_min + bounds_max) / 2.0
    radius = float(np.linalg.norm(points - center, axis=1).max(initial=0.0))
    center_x, center_y, center_depth = (
        values[0] for values in _camera_coordinates(center.reshape(1, 3), camera)
    )
    center_x = float(center_x)
    center_y = float(center_y)
    center_depth = float(center_depth)
    min_depth = float(depth.min())
    max_depth = float(depth.max())

    if max_depth <= 0.0:
        validator.issue(
            "camera.scene_behind",
            "scene.camera",
            "All scene geometry is behind the camera.",
            hint="Aim target at the scene bounds or move eye to the opposite side of target.",
        )
        return
    if max_depth < camera.near or min_depth > camera.far:
        plane = "near" if max_depth < camera.near else "far"
        validator.issue(
            "camera.clipping.outside",
            f"scene.camera.{plane}",
            f"The complete scene lies outside the camera {plane} clipping plane.",
            hint=(
                f"Set {plane} to include the scene depth range "
                f"[{max(min_depth, 0.0):.4g}, {max_depth:.4g}]."
            ),
        )
        return
    if min_depth < camera.near:
        validator.warning(
            "camera.clipping.near",
            "scene.camera.near",
            "The near clipping plane intersects scene geometry.",
            hint=f"Reduce near below {max(min_depth, 0.0):.4g} or move the camera back.",
        )
    if max_depth > camera.far:
        validator.warning(
            "camera.clipping.far",
            "scene.camera.far",
            "The far clipping plane intersects scene geometry.",
            hint=f"Increase far above {max_depth:.4g} or move the camera closer.",
        )

    from ..grammar.figure import OrthographicCamera

    output = figure.scene.output
    width = output.width if output is not None else 1024
    height = output.height if output is not None else 800
    if isinstance(camera, OrthographicCamera):
        half_height = camera.scale * 0.5
        half_width = half_height * width / height
        outside_x = abs(center_x) - half_width > radius
        outside_y = abs(center_y) - half_height > radius
    else:
        tangent_x, tangent_y = _perspective_tangents(camera, width, height)
        outside_x = abs(center_x) - center_depth * tangent_x > radius * np.sqrt(
            1.0 + tangent_x * tangent_x
        )
        outside_y = abs(center_y) - center_depth * tangent_y > radius * np.sqrt(
            1.0 + tangent_y * tangent_y
        )
    if outside_x or outside_y:
        validator.issue(
            "camera.scene_outside_view",
            "scene.camera.target",
            "The scene lies outside the camera field of view.",
            hint="Aim target at the scene center, widen fov/scale, or move eye farther away.",
        )

    projected = [_projected_bounds(points, camera) for points in view_points]
    depth_ranges: list[tuple[float, float] | None] = []
    for values in view_points:
        if not values.size:
            depth_ranges.append(None)
            continue
        view_depth = _camera_coordinates(values, camera)[2]
        depth_ranges.append((float(view_depth.min()), float(view_depth.max())))
    for back_index, back_bounds in enumerate(projected):
        if back_bounds is None or scene[back_index].mark is not Mark.Surface:
            continue
        back_depth = depth_ranges[back_index]
        assert back_depth is not None
        for front_index, front_bounds in enumerate(projected):
            if front_index == back_index or front_bounds is None:
                continue
            if scene[front_index].mark is not Mark.Surface:
                continue
            front_depth = depth_ranges[front_index]
            assert front_depth is not None
            if (
                front_depth[1] < back_depth[0]
                and _rect_coverage(front_bounds, back_bounds) >= 0.98
            ):
                validator.warning(
                    "layer.possible_occlusion",
                    f"views[{back_index}]",
                    f"View {front_index} is entirely closer and covers this view's projected bounds.",
                    hint=(
                        "Hide or separate the front layer, use transparency, or inspect the "
                        "occluded layer from another camera view."
                    ),
                )
                break


def _attribute_name(value: str | Attribute) -> str:
    return value if isinstance(value, str) else value.name


class _Validator:
    def __init__(self, capabilities: BackendCapabilities, strict: bool):
        self.capabilities = capabilities
        self.strict = strict
        self.diagnostics: list[Diagnostic] = []

    def issue(
        self,
        code: str,
        path: str,
        message: str,
        *,
        hint: str | None = None,
        degradation: bool = False,
    ) -> None:
        severity: Severity = "error" if (not degradation or self.strict) else "warning"
        self.diagnostics.append(Diagnostic(code, severity, path, message, hint))

    def warning(
        self, code: str, path: str, message: str, hint: str | None = None
    ) -> None:
        self.diagnostics.append(Diagnostic(code, "warning", path, message, hint))

    def validate_view(self, view: _ResolvedView, index: int) -> None:
        base = f"views[{index}]"
        if view.data is None:
            self.issue(
                "data.missing",
                f"{base}.data",
                "No data component is specified.",
                hint=(
                    "Attach a mesh, point array, table, or supported geometry object "
                    "with hkw.layer(data) or .data(data)."
                ),
            )
            return

        mark_name = view.mark.name.lower()
        if mark_name not in self.capabilities.marks:
            self.issue(
                "backend.mark.unsupported",
                f"{base}.mark",
                f"Backend '{self.capabilities.name}' does not support mark '{mark_name}'.",
                hint=f"Choose one of the backend's supported marks: {sorted(self.capabilities.marks)}.",
                degradation=True,
            )

        mesh = view.data.mesh
        generated = self._generated_attributes(view.transforms)
        active: dict[str, Channel] = {}
        for channel_index, channel in enumerate(view.channels):
            name = _channel_name(channel)
            path = f"{base}.channels.{name}"
            if name in active:
                continue
            active[name] = channel
            allowed = next(
                (
                    marks
                    for cls, marks in _MARK_CHANNELS.items()
                    if isinstance(channel, cls)
                ),
                None,
            )
            if allowed is not None and view.mark not in allowed:
                valid_marks = ", ".join(sorted(mark.name for mark in allowed))
                self.issue(
                    "channel.mark_incompatible",
                    path,
                    f"Channel '{name}' has no effect on the {view.mark.name} mark.",
                    hint=f"Use one of these marks: {valid_marks}.",
                    degradation=True,
                )
            self._validate_channel(channel, mesh, generated, path)
            if (
                isinstance(channel, Material)
                and channel.back_side is not None
                and view.mark is not Mark.Surface
            ):
                self.issue(
                    "material.back_side.mark_incompatible",
                    f"{path}.back_side",
                    "Back-face material has no effect on non-surface marks.",
                    hint="Use the Surface mark or remove the back-side material.",
                    degradation=True,
                )

        for transform_index, transform in enumerate(view.transforms):
            for chain_index, node in enumerate(_transform_nodes(transform)):
                self._validate_transform(
                    node,
                    mesh,
                    generated,
                    view.mark,
                    f"{base}.transforms[{transform_index}].chain[{chain_index}]",
                )

    def _generated_attributes(
        self, transforms: list[Transform]
    ) -> dict[str, _AttributeInfo]:
        generated: dict[str, _AttributeInfo] = {}
        for transform in transforms:
            for node in _transform_nodes(transform):
                if isinstance(node, Compute):
                    for name in (node.x, node.y, node.z):
                        if name:
                            generated[name] = _AttributeInfo(1, "vertex")
                    if node.vertex_normal:
                        generated[node.vertex_normal] = _AttributeInfo(3, "vertex")
                    if node.facet_normal:
                        generated[node.facet_normal] = _AttributeInfo(3, "facet")
                    if node.normal:
                        generated[node.normal] = _AttributeInfo(3, "indexed")
                    if node.component:
                        generated[node.component] = _AttributeInfo(1, "facet")
                elif isinstance(node, NormTransform):
                    generated[node.norm_attr_name] = _AttributeInfo(1, "derived")
        return generated

    def _info(
        self,
        value: str | Attribute,
        mesh: lagrange.SurfaceMesh,
        generated: dict[str, _AttributeInfo],
        path: str,
    ) -> _AttributeInfo | None:
        name = _attribute_name(value)
        if name in generated:
            return generated[name]
        if not mesh.has_attribute(name):
            available = sorted(
                mesh.get_attribute_name(attribute_id)
                for attribute_id in mesh.get_matching_attribute_ids()
            )
            self.issue(
                "attribute.missing",
                path,
                f"Attribute '{name}' does not exist.",
                hint=f"Available attributes: {available}",
            )
            return None
        if mesh.is_attribute_indexed(name):
            attribute = mesh.indexed_attribute(name)
            values = np.asarray(attribute.values.data)
        else:
            attribute = mesh.attribute(name)
            values = np.asarray(attribute.data)
        minimum = None
        has_nonfinite = False
        if values.size and np.issubdtype(values.dtype, np.number):
            finite_mask = np.isfinite(values)
            has_nonfinite = not bool(np.all(finite_mask))
            finite = values[finite_mask]
            if finite.size:
                minimum = float(np.min(finite))
        return _AttributeInfo(
            channels=int(attribute.num_channels),
            element=attribute.element_type.name.lower(),
            minimum=minimum,
            has_nonfinite=has_nonfinite,
        )

    def _check_attribute(
        self,
        value: str | Attribute,
        mesh: lagrange.SurfaceMesh,
        generated: dict[str, _AttributeInfo],
        path: str,
        *,
        channels: int | None = None,
        minimum_channels: int | None = None,
        elements: frozenset[str] | None = None,
    ) -> _AttributeInfo | None:
        info = self._info(value, mesh, generated, path)
        if info is None:
            return None
        if info.has_nonfinite:
            self.issue(
                "attribute.nonfinite",
                path,
                "Attribute contains NaN or infinite values.",
                hint="Clean or mask non-finite values before visualization.",
            )
        if channels is not None and info.channels != channels:
            self.issue(
                "attribute.channels",
                path,
                f"Expected {channels} channel(s), but attribute has {info.channels}.",
                hint=f"Choose an attribute with exactly {channels} channels or derive one with a transform.",
            )
        if minimum_channels is not None and info.channels < minimum_channels:
            self.issue(
                "attribute.channels",
                path,
                f"Expected at least {minimum_channels} channels, but attribute has {info.channels}.",
                hint=f"Choose an attribute with at least {minimum_channels} channels.",
            )
        if elements is not None and info.element not in elements:
            self.issue(
                "attribute.element",
                path,
                f"Attribute element '{info.element}' is unsupported here.",
                hint=f"Supported elements: {sorted(elements)}",
            )
        if isinstance(value, Attribute):
            self._validate_scale(value.scale, info, f"{path}.scale")
        return info

    def _validate_scale(
        self, scale: Scale | float | None, info: _AttributeInfo, path: str
    ) -> None:
        current = scale if isinstance(scale, Scale) else None
        first = True
        nonnegative = False
        while current is not None:
            if isinstance(current, NormScale):
                if not first:
                    self.issue(
                        "scale.norm.order",
                        path,
                        "Norm must be the first scale because it changes vector data to scalar data.",
                        hint="Reorder the scale chain so Norm receives the original vector field.",
                    )
                nonnegative = True
            if (
                isinstance(current, Log)
                and not nonnegative
                and info.minimum is not None
                and info.minimum <= 0
            ):
                self.issue(
                    "scale.log.nonpositive",
                    path,
                    f"Log scale input contains values <= 0 (minimum {info.minimum:g}).",
                    hint="Filter, mask, or explicitly transform non-positive values before applying Log.",
                    degradation=True,
                )
            if isinstance(current, Custom):
                self.warning(
                    "scale.custom.unstructured",
                    path,
                    "Custom callable scales cannot be serialized or statically validated.",
                    hint="Use declarative built-in scales when the layer must round-trip through JSON.",
                )
            first = False
            current = current._child

    def _validate_channel(
        self,
        channel: Channel,
        mesh: lagrange.SurfaceMesh,
        generated: dict[str, _AttributeInfo],
        path: str,
    ) -> None:
        if isinstance(channel, Position):
            self._check_attribute(
                channel.data, mesh, generated, f"{path}.data", channels=mesh.dimension
            )
        elif isinstance(channel, Normal):
            self._check_attribute(
                channel.data, mesh, generated, f"{path}.data", channels=mesh.dimension
            )
        elif isinstance(channel, Size) and not isinstance(channel.data, (int, float)):
            info = self._check_attribute(channel.data, mesh, generated, f"{path}.data")
            has_norm = isinstance(channel.data, Attribute) and isinstance(
                channel.data.scale, NormScale
            )
            if info is not None and info.channels != 1 and not has_norm:
                self.issue(
                    "channel.size.scalar_required",
                    f"{path}.data",
                    f"Size requires scalar data, but attribute has {info.channels} channels.",
                    hint="Use hkw.norm(field) to map vector magnitude to size.",
                )
        elif isinstance(channel, VectorField):
            self._check_attribute(
                channel.data,
                mesh,
                generated,
                f"{path}.data",
                channels=mesh.dimension,
                elements=frozenset({"vertex", "facet", "indexed"}),
            )
        elif isinstance(channel, Covariance):
            self._check_attribute(
                channel.data,
                mesh,
                generated,
                f"{path}.data",
                channels=mesh.dimension * mesh.dimension,
                elements=frozenset({"vertex"}),
            )
        elif isinstance(channel, Shape):
            if channel.base_shape not in {"sphere", "cube", "disk"}:
                self.issue(
                    "channel.shape.unknown",
                    f"{path}.base_shape",
                    f"Unknown point shape '{channel.base_shape}'.",
                    hint="Choose sphere, cube, or disk.",
                )
            if channel.orientation is not None:
                self._check_attribute(
                    channel.orientation,
                    mesh,
                    generated,
                    f"{path}.orientation",
                    channels=mesh.dimension,
                    elements=frozenset({"vertex"}),
                )
        elif isinstance(channel, (BumpMap, NormalMap)):
            self._validate_texture(channel.texture, mesh, generated, f"{path}.texture")
            if self.capabilities.name in {"webgl", "blender"} and not isinstance(
                channel.texture, Image
            ):
                self.issue(
                    "backend.map.image_required",
                    f"{path}.texture",
                    f"Backend '{self.capabilities.name}' only supports image-based {path.rsplit('.', 1)[-1]} textures.",
                    hint="Provide hkw.texture.Image(...) or choose Mitsuba for procedural map textures.",
                    degradation=True,
                )
        elif isinstance(channel, Material):
            self._validate_material(channel, mesh, generated, path)

    def _validate_material(
        self,
        material: Material,
        mesh: lagrange.SurfaceMesh,
        generated: dict[str, _AttributeInfo],
        path: str,
    ) -> None:
        if material.back_side is not None:
            self._validate_material(
                material.back_side, mesh, generated, f"{path}.back_side"
            )
        if isinstance(material, Hair):
            if isinstance(material.color, Texture):
                self.issue(
                    "backend.hair.data_color",
                    f"{path}.color",
                    "Data-driven Hair color is unsupported by the rendering backends.",
                    hint="Use a Diffuse or Principled material for attribute-driven strand color.",
                    degradation=True,
                )
            if (material.root_color is not None or material.tip_color is not None) and (
                "hair_gradient" not in self.capabilities.features
            ):
                self.issue(
                    "backend.hair.gradient",
                    path,
                    f"Backend '{self.capabilities.name}' does not preserve root/tip hair gradients.",
                    hint="Use one uniform Hair color or select a backend with hair_gradient support.",
                    degradation=True,
                )
        for field in fields(material):
            if field.name.startswith("_") or field.name == "back_side":
                continue
            value = getattr(material, field.name)
            if isinstance(value, Texture):
                self._validate_texture(value, mesh, generated, f"{path}.{field.name}")

    def _validate_texture(
        self,
        texture: Any,
        mesh: lagrange.SurfaceMesh,
        generated: dict[str, _AttributeInfo],
        path: str,
    ) -> None:
        if isinstance(texture, ScalarField):
            info = self._check_attribute(texture.data, mesh, generated, f"{path}.data")
            if (
                info is not None
                and texture.colormap != "identity"
                and info.channels != 1
            ):
                self.issue(
                    "texture.scalar_field.scalar_required",
                    f"{path}.data",
                    f"ScalarField requires scalar data, but attribute has {info.channels} channels.",
                    hint="Use hkw.norm(field) for vector magnitudes or colormap='identity' for RGB data.",
                )
            if texture.domain is not None and texture.domain[0] > texture.domain[1]:
                self.issue(
                    "texture.domain.order",
                    f"{path}.domain",
                    "ScalarField domain minimum exceeds its maximum.",
                    hint="Swap the two domain endpoints so the minimum comes first.",
                )
        elif isinstance(texture, Image):
            if not Path(texture.filename).is_file():
                self.issue(
                    "texture.image.missing",
                    f"{path}.filename",
                    f"Texture file '{texture.filename}' does not exist.",
                    hint="Correct the texture path or place the image beside the specification file.",
                )
            if texture.uv is not None:
                self._check_attribute(
                    texture.uv, mesh, generated, f"{path}.uv", channels=2
                )
            elif not mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV):
                self.issue(
                    "texture.uv.missing",
                    f"{path}.uv",
                    "Image texture requires UV coordinates, but the mesh has no UV attribute.",
                    hint="Provide a two-channel UV attribute or generate UV coordinates before texturing.",
                )
        elif isinstance(texture, Checkerboard):
            if texture.uv is not None:
                self._check_attribute(
                    texture.uv, mesh, generated, f"{path}.uv", channels=2
                )
            elif not mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV):
                self.issue(
                    "texture.uv.missing",
                    f"{path}.uv",
                    "Checkerboard texture requires UV coordinates, but the mesh has no UV attribute.",
                    hint="Provide a two-channel UV attribute or generate UV coordinates before texturing.",
                )
            self._validate_texture(
                texture.texture1, mesh, generated, f"{path}.texture1"
            )
            self._validate_texture(
                texture.texture2, mesh, generated, f"{path}.texture2"
            )
        elif isinstance(texture, Isocontour):
            self._check_attribute(
                texture.data, mesh, generated, f"{path}.data", channels=1
            )
            if texture.num_contours <= 0:
                self.issue(
                    "texture.isocontour.count",
                    f"{path}.num_contours",
                    "Isocontour num_contours must be positive.",
                    hint="Set num_contours to an integer greater than zero.",
                )
            self._validate_texture(
                texture.texture1, mesh, generated, f"{path}.texture1"
            )
            self._validate_texture(
                texture.texture2, mesh, generated, f"{path}.texture2"
            )

    def _validate_transform(
        self,
        transform: Transform,
        mesh: lagrange.SurfaceMesh,
        generated: dict[str, _AttributeInfo],
        mark: Mark,
        path: str,
    ) -> None:
        if isinstance(transform, Filter):
            if transform.data is not None:
                info = self._check_attribute(
                    transform.data, mesh, generated, f"{path}.data"
                )
                supported = {"vertex", "facet"}
                if mark is Mark.Surface:
                    supported.add("indexed")
                if info is not None and info.element not in supported:
                    self.issue(
                        "transform.filter.element",
                        f"{path}.data",
                        f"Filter does not support '{info.element}' attributes.",
                        hint=f"Supported elements for {mark.name}: {sorted(supported)}",
                    )
                if info is not None and mark is Mark.Curve and info.element == "vertex":
                    self.issue(
                        "transform.filter.curve_vertex",
                        path,
                        "Filter does not support vertex filtering for Curve marks.",
                        hint="Use a facet attribute, or filter the source geometry before applying the Curve mark.",
                    )
            condition_name = getattr(transform.condition, "__name__", None)
            if condition_name not in {"_default_condition", "hakowan_expression"}:
                self.warning(
                    "transform.filter.callable",
                    f"{path}.condition",
                    "Filter callable cannot be serialized or statically validated.",
                    hint="Use a serializable filter expression when the layer must round-trip through JSON.",
                )
        elif isinstance(transform, UVMesh):
            if transform.uv is not None:
                self._check_attribute(
                    transform.uv, mesh, generated, f"{path}.uv", channels=2
                )
            elif not mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV):
                self.issue(
                    "transform.uv.missing",
                    f"{path}.uv",
                    "UVMesh requires a UV attribute, but none is available.",
                    hint="Set transform.uv to a two-channel UV attribute or add UV coordinates to the mesh.",
                )
        elif isinstance(transform, Explode):
            self._check_attribute(
                transform.pieces,
                mesh,
                generated,
                f"{path}.pieces",
                channels=1,
                elements=frozenset({"facet"}),
            )
        elif isinstance(transform, NormTransform):
            self._check_attribute(
                transform.data,
                mesh,
                generated,
                f"{path}.data",
                minimum_channels=2,
            )
        elif isinstance(transform, Boundary):
            for index, name in enumerate(transform.attributes):
                self._check_attribute(
                    name, mesh, generated, f"{path}.attributes[{index}]"
                )
        elif isinstance(transform, (Streamline, Fur)):
            self._check_attribute(
                transform.vec_field,
                mesh,
                generated,
                f"{path}.vec_field",
                channels=mesh.dimension,
                elements=frozenset({"vertex", "facet", "corner"}),
            )
            if (
                isinstance(transform, Fur)
                and transform.children > 0
                and ("fur_children" not in self.capabilities.features)
            ):
                self.issue(
                    "backend.fur.children",
                    f"{path}.children",
                    f"Backend '{self.capabilities.name}' ignores Fur child hairs.",
                    hint="Set children=0 or choose a backend advertising fur_children support.",
                    degradation=True,
                )


def validate(
    root: Layer | object,
    backend: BackendName | None = None,
    *,
    strict: bool = True,
    compile_check: bool = True,
) -> ValidationReport:
    """Validate a Layer or Figure without rendering or mutating it.

    Static checks cover attributes, channels, scales, transforms, resources,
    scene settings, and backend capabilities. ``compile_check=True`` then runs
    the real compiler on deep-copied data and checks empty geometry, camera
    framing/clipping, and likely layer occlusion.

    Args:
        root: Layer or Figure to validate.
        backend: Target backend, or the configured default when omitted.
        strict: Promote backend approximations and ignored features to errors.
        compile_check: Compile and inspect the resolved scene after static checks.

    Returns:
        A structured report; intrinsic failures are errors in every mode.

    """
    figure = None
    if not isinstance(root, Layer):
        from ..grammar.figure import Figure

        if not isinstance(root, Figure):
            raise TypeError(f"Expected a Layer or Figure, got {type(root)!r}")
        figure = root
        root = root.layer
    try:
        capabilities = get_backend_capabilities(backend)
    except ValueError as exc:
        backend_name = str(backend) if backend is not None else "unknown"
        return ValidationReport(
            backend=backend_name,
            strict=strict,
            diagnostics=(
                Diagnostic(
                    "backend.unknown",
                    "error",
                    "backend",
                    str(exc),
                    "Choose a backend returned by hkw.list_backend_capabilities().",
                ),
            ),
        )
    validator = _Validator(capabilities, strict)
    if figure is not None:
        from ..grammar.figure import ThinLensCamera

        if (
            isinstance(figure.scene.camera, ThinLensCamera)
            and capabilities.name == "webgl"
        ):
            validator.issue(
                "backend.camera.thin_lens",
                "scene.camera",
                "WebGL renders a thin-lens camera as standard perspective.",
                hint="Use a perspective camera on WebGL or render with Mitsuba/Blender for depth of field.",
                degradation=True,
            )
        environment = figure.scene.environment
        if (
            environment is not None
            and environment.enabled
            and environment.path is not None
            and not environment.path.is_file()
        ):
            validator.issue(
                "environment.path.missing",
                "scene.environment.path",
                f"Environment map '{environment.path}' does not exist.",
                hint="Correct the path or place the environment asset beside the specification file.",
            )
        output = figure.scene.output
        if output is not None:
            requested = {str(item) for item in output.passes if item != "beauty"}
            unsupported = requested - capabilities.render_passes
            if unsupported:
                validator.issue(
                    "backend.passes.unsupported",
                    "scene.output.passes",
                    f"Backend '{capabilities.name}' does not support passes {sorted(unsupported)}.",
                    hint="Remove unsupported passes or choose a backend that advertises them.",
                    degradation=True,
                )
    for index, view in enumerate(_flatten_views(root)):
        validator.validate_view(view, index)
    if compile_check and not any(
        item.severity == "error" for item in validator.diagnostics
    ):
        try:
            from ..compiler import compile as compile_layer

            scene = compile_layer(root, preserve_attributes=True)
            _validate_compiled_scene(scene, figure, validator)
        except Exception as exc:
            message = str(exc).strip() or type(exc).__name__
            validator.issue(
                "compile.failed",
                "layer",
                f"Layer compilation failed: {message}",
                hint=f"Fix the reported {type(exc).__name__} in the layer or transform configuration.",
            )
    return ValidationReport(
        backend=capabilities.name,
        strict=strict,
        diagnostics=tuple(validator.diagnostics),
    )


__all__ = [
    "Diagnostic",
    "ValidationError",
    "ValidationReport",
    "validate",
]
