"""Static, structured validation for Hakowan layer specifications."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import lagrange
import numpy as np

from .backends import BackendCapabilities, BackendName, get_backend_capabilities
from .grammar.channel import (
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
from .grammar.channel.material import Hair, Material
from .grammar.dataframe import DataFrame
from .grammar.layer import Layer
from .grammar.mark import Mark
from .grammar.scale import Attribute, Custom, Log, Norm as NormScale, Scale
from .grammar.texture import Checkerboard, Image, Isocontour, ScalarField, Texture
from .grammar.transform import (
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
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Validation result for a layer and target backend."""

    backend: str
    strict: bool
    diagnostics: tuple[Diagnostic, ...]

    @property
    def valid(self) -> bool:
        return not any(item.severity == "error" for item in self.diagnostics)

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "error")

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
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
        self.report = report
        detail = "; ".join(
            f"{item.path}: {item.message}" for item in report.errors
        )
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
            resolved.append(
                _ResolvedView(
                    data=data,
                    mark=mark or Mark.Surface,
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
                hint="Attach a mesh or mesh filename with hkw.layer(data) or .data(data).",
            )
            return

        mark_name = view.mark.name.lower()
        if mark_name not in self.capabilities.marks:
            self.issue(
                "backend.mark.unsupported",
                f"{base}.mark",
                f"Backend '{self.capabilities.name}' does not support mark '{mark_name}'.",
                degradation=True,
            )

        mesh = view.data.mesh
        generated = self._generated_attributes(view.transforms)
        active: dict[str, Channel] = {}
        for channel_index, channel in enumerate(view.channels):
            name = _channel_name(channel)
            path = f"{base}.channels.{name}"
            if name in active:
                self.warning(
                    "channel.shadowed",
                    path,
                    f"This {name} channel is shadowed by an earlier effective channel.",
                    hint="Remove the shadowed channel or place the intended override higher in the layer tree.",
                )
                continue
            active[name] = channel
            allowed = next(
                (marks for cls, marks in _MARK_CHANNELS.items() if isinstance(channel, cls)),
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
            )
        if minimum_channels is not None and info.channels < minimum_channels:
            self.issue(
                "attribute.channels",
                path,
                f"Expected at least {minimum_channels} channels, but attribute has {info.channels}.",
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
            self._check_attribute(channel.data, mesh, generated, f"{path}.data", channels=mesh.dimension)
        elif isinstance(channel, Normal):
            self._check_attribute(channel.data, mesh, generated, f"{path}.data", channels=mesh.dimension)
        elif isinstance(channel, Size) and not isinstance(channel.data, (int, float)):
            info = self._check_attribute(channel.data, mesh, generated, f"{path}.data")
            has_norm = isinstance(channel.data, Attribute) and isinstance(channel.data.scale, NormScale)
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
                elements=frozenset({"vertex", "facet"}),
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
            self._validate_material(material.back_side, mesh, generated, f"{path}.back_side")
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
            if info is not None and texture.colormap != "identity" and info.channels != 1:
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
                )
        elif isinstance(texture, Image):
            if not Path(texture.filename).is_file():
                self.issue(
                    "texture.image.missing",
                    f"{path}.filename",
                    f"Texture file '{texture.filename}' does not exist.",
                )
            if texture.uv is not None:
                self._check_attribute(texture.uv, mesh, generated, f"{path}.uv", channels=2)
            elif not mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV):
                self.issue(
                    "texture.uv.missing",
                    f"{path}.uv",
                    "Image texture requires UV coordinates, but the mesh has no UV attribute.",
                )
        elif isinstance(texture, Checkerboard):
            if texture.uv is not None:
                self._check_attribute(texture.uv, mesh, generated, f"{path}.uv", channels=2)
            elif not mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV):
                self.issue(
                    "texture.uv.missing",
                    f"{path}.uv",
                    "Checkerboard texture requires UV coordinates, but the mesh has no UV attribute.",
                )
            self._validate_texture(texture.texture1, mesh, generated, f"{path}.texture1")
            self._validate_texture(texture.texture2, mesh, generated, f"{path}.texture2")
        elif isinstance(texture, Isocontour):
            self._check_attribute(texture.data, mesh, generated, f"{path}.data", channels=1)
            if texture.num_contours <= 0:
                self.issue(
                    "texture.isocontour.count",
                    f"{path}.num_contours",
                    "Isocontour num_contours must be positive.",
                )
            self._validate_texture(texture.texture1, mesh, generated, f"{path}.texture1")
            self._validate_texture(texture.texture2, mesh, generated, f"{path}.texture2")

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
                info = self._check_attribute(transform.data, mesh, generated, f"{path}.data")
                if info is not None and info.element not in {"vertex", "facet"}:
                    self.issue(
                        "transform.filter.element",
                        f"{path}.data",
                        f"Filter does not support '{info.element}' attributes.",
                    )
                if info is not None and mark is Mark.Curve and info.element == "vertex":
                    self.issue(
                        "transform.filter.curve_vertex",
                        path,
                        "Filter does not support vertex filtering for Curve marks.",
                    )
            self.warning(
                "transform.filter.callable",
                f"{path}.condition",
                "Filter callable cannot be serialized or statically validated.",
            )
        elif isinstance(transform, UVMesh):
            if transform.uv is not None:
                self._check_attribute(transform.uv, mesh, generated, f"{path}.uv", channels=2)
            elif not mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV):
                self.issue(
                    "transform.uv.missing",
                    f"{path}.uv",
                    "UVMesh requires a UV attribute, but none is available.",
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
                self._check_attribute(name, mesh, generated, f"{path}.attributes[{index}]")
        elif isinstance(transform, (Streamline, Fur)):
            self._check_attribute(
                transform.vec_field,
                mesh,
                generated,
                f"{path}.vec_field",
                channels=mesh.dimension,
                elements=frozenset({"vertex", "facet", "corner"}),
            )
            if isinstance(transform, Fur) and transform.children > 0 and (
                "fur_children" not in self.capabilities.features
            ):
                self.issue(
                    "backend.fur.children",
                    f"{path}.children",
                    f"Backend '{self.capabilities.name}' ignores Fur child hairs.",
                    degradation=True,
                )


def validate(
    root: Layer,
    backend: BackendName | None = None,
    *,
    strict: bool = True,
    compile_check: bool = True,
) -> ValidationReport:
    """Validate a layer without rendering or mutating it.

    ``strict=True`` promotes backend approximations and ignored features to
    errors. With ``strict=False`` they remain warnings; intrinsic problems such
    as missing attributes are always errors. ``compile_check=True`` also runs
    the real compiler on deep-copied view data after static validation succeeds.
    """
    if not isinstance(root, Layer):
        raise TypeError(f"Expected a Layer, got {type(root)!r}")
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
    for index, view in enumerate(_flatten_views(root)):
        validator.validate_view(view, index)
    if compile_check and not any(
        item.severity == "error" for item in validator.diagnostics
    ):
        try:
            from .compiler import compile as compile_layer

            compile_layer(root)
        except Exception as exc:
            message = str(exc).strip() or type(exc).__name__
            validator.issue(
                "compile.failed",
                "layer",
                f"Layer compilation failed: {message}",
                hint=f"Original exception type: {type(exc).__name__}",
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
