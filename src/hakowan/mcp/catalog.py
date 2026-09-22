"""Compact schema fragments and canonical FigureSpec templates for MCP clients."""

from __future__ import annotations

import copy
from typing import Any

from ..spec import FigureSpec, json_schema
from ..spec.model import SCHEMA_URL, SCHEMA_VERSION

_FRAGMENT_MODELS = {
    "attribute": "AttributeSpec",
    "channel.position": "PositionChannelSpec",
    "channel.normal": "NormalChannelSpec",
    "channel.size": "SizeChannelSpec",
    "channel.shape": "ShapeChannelSpec",
    "channel.vector_field": "VectorFieldChannelSpec",
    "channel.covariance": "CovarianceChannelSpec",
    "channel.bump_map": "BumpMapChannelSpec",
    "channel.normal_map": "NormalMapChannelSpec",
    "channels": "ChannelsSpec",
    "data.external": "ExternalDataSpec",
    "data.mesh_file": "MeshFileDataSpec",
    "layer": "LayerPropertiesSpec",
    "material.diffuse": "DiffuseMaterialSpec",
    "material.principled": "PrincipledMaterialSpec",
    "material.rough_plastic": "RoughPlasticMaterialSpec",
    "node.inherit": "InheritNodeSpec",
    "node.layer": "LayerNodeSpec",
    "node.layout": "LayoutNodeSpec",
    "node.overlay": "OverlayNodeSpec",
    "scene": "SceneSettingsSpec",
    "scene.camera.orthographic": "OrthographicCameraSpec",
    "scene.camera.perspective": "PerspectiveCameraSpec",
    "scene.camera.thin_lens": "ThinLensCameraSpec",
    "scene.environment": "EnvironmentSpec",
    "scene.output": "OutputSettingsSpec",
    "texture.image": "ImageTextureSpec",
    "texture.scalar_field": "ScalarFieldTextureSpec",
    "texture.uniform": "UniformTextureSpec",
    "transform.boundary": "BoundaryTransformSpec",
    "transform.clip": "ClipTransformSpec",
    "transform.compute": "ComputeTransformSpec",
    "transform.explode": "ExplodeTransformSpec",
    "transform.filter": "FilterTransformSpec",
    "transform.normalize": "NormalizeTransformSpec",
    "transform.norm": "NormTransformSpec",
    "transform.principal_axes": "PrincipalAxesTransformSpec",
    "transform.streamline": "StreamlineTransformSpec",
}

_TEMPLATE_DESCRIPTIONS = {
    "surface": "One unstyled surface layer.",
    "surface-scalar": "Surface colored by a scalar attribute with a legend.",
    "point-scalar": "Point glyphs colored by a scalar attribute.",
    "point-vector-size": "Point glyph sizes driven by a vector magnitude.",
    "vector-glyphs": "Arrow glyphs driven by a vector attribute.",
    "wireframe-overlay": "Surface with black boundary curves overlaid.",
    "clip-plane": "Surface clipped to one plane half-space.",
    "side-by-side": "Two copies of a source arranged along the x axis.",
}


def fragment_names() -> tuple[str, ...]:
    """Return stable names accepted by :func:`schema_fragment`."""
    return tuple(sorted(_FRAGMENT_MODELS))


def _references(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            result.add(reference.removeprefix("#/$defs/"))
        for child in value.values():
            result.update(_references(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_references(child))
    return result


def schema_fragment(name: str) -> dict[str, Any]:
    """Return one self-contained JSON Schema fragment and its dependencies."""
    normalized = name.strip().lower().replace("/", ".")
    model_name = _FRAGMENT_MODELS.get(normalized)
    schema = json_schema()
    definitions = schema.get("$defs", {})
    if model_name is None:
        model_name = next(
            (candidate for candidate in definitions if candidate.lower() == normalized),
            None,
        )
    if model_name is None or model_name not in definitions:
        raise KeyError(name)
    selected: dict[str, Any] = {}
    pending = [model_name]
    while pending:
        current = pending.pop()
        if current in selected:
            continue
        definition = definitions[current]
        selected[current] = copy.deepcopy(definition)
        pending.extend(sorted(_references(definition) - selected.keys()))
    return {
        "$schema": schema.get(
            "$schema", "https://json-schema.org/draft/2020-12/schema"
        ),
        "title": f"Hakowan {normalized} schema fragment",
        "$ref": f"#/$defs/{model_name}",
        "$defs": selected,
    }


def template_names() -> tuple[str, ...]:
    """Return stable names accepted by :func:`spec_template`."""
    return tuple(_TEMPLATE_DESCRIPTIONS)


def _layer(data_id: str, *, mark: str) -> dict[str, Any]:
    return {
        "kind": "layer",
        "spec": {
            "data": {"kind": "external", "id": data_id},
            "mark": mark,
        },
    }


def spec_template(
    name: str, *, data_id: str = "data", attribute: str = "value"
) -> dict[str, Any]:
    """Return a minimal validated FigureSpec template with optional fields omitted."""
    if not data_id:
        raise ValueError("data_id must not be empty")
    normalized = name.strip().lower()
    if (
        normalized
        in {
            "surface-scalar",
            "point-scalar",
            "point-vector-size",
            "vector-glyphs",
        }
        and not attribute
    ):
        raise ValueError(f"attribute must not be empty for template {normalized!r}")
    root: dict[str, Any]
    if normalized == "surface":
        root = _layer(data_id, mark="surface")
    elif normalized in {"surface-scalar", "point-scalar"}:
        mark = "surface" if normalized == "surface-scalar" else "point"
        root = _layer(data_id, mark=mark)
        root["spec"]["channels"] = {
            "material": {
                "kind": "diffuse",
                "reflectance": {
                    "kind": "scalar_field",
                    "data": {"name": attribute},
                    "colormap": "viridis",
                    "legend": True,
                },
            }
        }
    elif normalized == "point-vector-size":
        root = _layer(data_id, mark="point")
        root["spec"]["channels"] = {
            "size": {
                "kind": "size",
                "data": {
                    "name": attribute,
                    "scales": [{"kind": "norm", "order": 2.0}],
                },
            }
        }
    elif normalized == "vector-glyphs":
        root = _layer(data_id, mark="curve")
        root["spec"]["channels"] = {
            "size": {"kind": "size", "data": 0.01},
            "vector_field": {
                "kind": "vector_field",
                "data": {
                    "name": attribute,
                    "scales": [{"kind": "uniform", "factor": 0.2}],
                },
                "end_type": "arrow",
            },
            "material": {"kind": "diffuse", "reflectance": "black"},
        }
    elif normalized == "wireframe-overlay":
        surface = _layer(data_id, mark="surface")
        wireframe = _layer(data_id, mark="curve")
        wireframe["spec"]["transforms"] = [{"kind": "boundary"}]
        wireframe["spec"]["channels"] = {
            "size": {"kind": "size", "data": 0.01},
            "material": {"kind": "diffuse", "reflectance": "black"},
        }
        root = {"kind": "overlay", "children": [surface, wireframe]}
    elif normalized == "clip-plane":
        root = _layer(data_id, mark="surface")
        root["spec"]["transforms"] = [
            {"kind": "clip", "point": [0, 0, 0], "normal": [1, 0, 0]}
        ]
    elif normalized == "side-by-side":
        root = {
            "kind": "layout",
            "axis": "x",
            "gap": 0.05,
            "children": [
                _layer(data_id, mark="surface"),
                _layer(data_id, mark="surface"),
            ],
        }
    else:
        raise KeyError(name)
    result = {"$schema": SCHEMA_URL, "version": SCHEMA_VERSION, "root": root}
    FigureSpec.model_validate(result)
    return result


def template_catalog() -> list[dict[str, str]]:
    """Return compact descriptions of available canonical templates."""
    return [
        {"name": name, "description": description}
        for name, description in _TEMPLATE_DESCRIPTIONS.items()
    ]


__all__ = [
    "fragment_names",
    "schema_fragment",
    "spec_template",
    "template_catalog",
    "template_names",
]
