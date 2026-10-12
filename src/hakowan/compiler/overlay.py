"""Collect backend-neutral legend and annotation metadata from compiled views."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from typing import Any

import numpy as np

from ..common.to_color import to_color
from ..grammar.overlay import Annotation, Legend, WorldAnnotation
from ..grammar.texture import ScalarField, Texture
from .scene import Scene


@dataclass(frozen=True, slots=True)
class CompiledLegend:
    title: str
    units: str | None
    categories: bool
    domain: tuple[float, float] | None
    values: tuple[float, ...]
    labels: tuple[str, ...]
    colors: tuple[tuple[float, float, float], ...]
    position: str
    width: int
    scale: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompiledAnnotation:
    text: str
    position: tuple[float, float]
    color: tuple[float, float, float]
    font_size: int
    anchor: str
    background: tuple[float, float, float] | None
    padding: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompiledWorldAnnotation:
    """World-space anchor and its owning rendered view."""

    text: str
    position: tuple[float, float, float]
    layer: int
    cell: str | None
    color: tuple[float, float, float]
    font_size: int
    background: tuple[float, float, float] | None
    padding: int
    offset: tuple[float, float] | None = None
    arrow: bool = True

    layers: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _format(value: float, spec: str) -> str:
    try:
        return format(value, spec)
    except ValueError:
        return f"{value:.3g}"


def _category_label(value: float, settings: Legend) -> str:
    fallback = _format(value, settings.format)
    if not settings.category_labels:
        return fallback
    return settings.category_labels.get(
        str(value), settings.category_labels.get(format(value, "g"), fallback)
    )


def _compiled_legend(texture: ScalarField) -> CompiledLegend | None:
    if texture.legend is False or texture._legend_colors is None:
        return None
    assert not isinstance(texture.legend, bool) or texture.legend is True
    settings = texture.legend if isinstance(texture.legend, Legend) else Legend()
    attribute = texture.data
    title = settings.title or str(getattr(attribute, "name", "value"))
    units = settings.units or getattr(attribute, "unit", None)
    if texture.categories:
        values = texture._legend_values or ()
        labels = tuple(_category_label(value, settings) for value in values)
        colors = texture._legend_colors[: len(values)]
        domain = None
    else:
        domain = texture._legend_domain
        if domain is None:
            return None
        values = tuple(float(value) for value in np.linspace(*domain, settings.ticks))
        labels = tuple(_format(value, settings.format) for value in values)
        colors = texture._legend_colors
    return CompiledLegend(
        title=title,
        units=units,
        categories=texture.categories,
        domain=domain,
        values=values,
        labels=labels,
        colors=colors,
        position=settings.position,
        width=settings.width,
        scale=texture._legend_scale,
    )


def _walk_textures(value: Any, seen: set[int]) -> list[ScalarField]:
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, ScalarField):
        return [value]
    result: list[ScalarField] = []
    if isinstance(value, Texture) or is_dataclass(value):
        for item in fields(value):
            if item.name.startswith("_"):
                continue
            child = getattr(value, item.name)
            if isinstance(child, (Texture, list, tuple)) or is_dataclass(child):
                result.extend(_walk_textures(child, seen))
    elif isinstance(value, (list, tuple)):
        for child in value:
            result.extend(_walk_textures(child, seen))
    return result


def _compiled_annotation(annotation: Annotation) -> CompiledAnnotation:
    raw_color = to_color(annotation.color).data
    color = (float(raw_color[0]), float(raw_color[1]), float(raw_color[2]))
    if annotation.background is not None:
        raw_background = to_color(annotation.background).data
        background = (
            float(raw_background[0]),
            float(raw_background[1]),
            float(raw_background[2]),
        )
    else:
        background = None

    return CompiledAnnotation(
        text=annotation.text,
        position=annotation.position,
        color=color,
        font_size=annotation.font_size,
        anchor=annotation.anchor,
        background=background,
        padding=annotation.padding,
    )


def collect_overlays(scene: Scene) -> None:
    """Populate unique scene-level legends and annotations from compiled views."""
    legends: list[CompiledLegend] = []
    annotations: list[CompiledAnnotation] = []
    for view in scene:
        for channel in (
            view.material_channel,
            view.bump_map,
            view.normal_map,
        ):
            if channel is None:
                continue
            for texture in _walk_textures(channel, set()):
                legend = _compiled_legend(texture)
                if legend is not None and legend not in legends:
                    legends.append(legend)
        for annotation in view.annotations:
            if isinstance(annotation, Annotation):
                compiled = _compiled_annotation(annotation)
                if compiled not in annotations:
                    annotations.append(compiled)
    scene.legends = legends
    scene.annotations = annotations


def collect_world_annotations(scene: Scene) -> None:
    """Resolve local annotation points after layout and scene transforms."""
    view_indices: dict[tuple[Any, ...], int] = {}
    result: list[CompiledWorldAnnotation] = []
    for index, view in enumerate(scene):
        cell = (
            "/".join(f"{node_id}.{branch}" for node_id, branch in view._layout_cell)
            if view._layout_cell
            else None
        )
        for annotation in view.annotations:
            if not isinstance(annotation, WorldAnnotation):
                continue
            point = np.asarray((*annotation.position, 1.0), dtype=np.float64)
            transformed = view.global_transform @ point
            color = to_color(annotation.color).data
            background = (
                to_color(annotation.background).data
                if annotation.background is not None
                else None
            )
            position = (
                float(transformed[0]),
                float(transformed[1]),
                float(transformed[2]),
            )
            rgb = (float(color[0]), float(color[1]), float(color[2]))
            rgb_background = (
                (float(background[0]), float(background[1]), float(background[2]))
                if background is not None
                else None
            )
            key = (
                cell,
                annotation.text,
                position,
                rgb,
                annotation.font_size,
                rgb_background,
                annotation.padding,
                annotation.offset,
                annotation.arrow,
            )
            if key in view_indices:
                result_index = view_indices[key]
                existing = result[result_index]
                owners = existing.layers or (existing.layer,)
                if index not in owners:
                    result[result_index] = replace(existing, layers=owners + (index,))
                continue
            view_indices[key] = len(result)
            result.append(
                CompiledWorldAnnotation(
                    text=annotation.text,
                    position=position,
                    layer=index,
                    layers=(index,),
                    cell=cell,
                    color=rgb,
                    font_size=annotation.font_size,
                    background=rgb_background,
                    padding=annotation.padding,
                    offset=annotation.offset,
                    arrow=annotation.arrow,
                )
            )
    scene.world_annotations = result


__all__ = [
    "CompiledAnnotation",
    "CompiledWorldAnnotation",
    "CompiledLegend",
    "collect_overlays",
    "collect_world_annotations",
]
