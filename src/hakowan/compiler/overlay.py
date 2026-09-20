"""Collect backend-neutral legend and annotation metadata from compiled views."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any

import numpy as np

from ..common.to_color import to_color
from ..grammar.overlay import Annotation, Legend
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


def _format(value: float, spec: str) -> str:
    try:
        return format(value, spec)
    except ValueError:
        return f"{value:.3g}"


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
        labels = tuple(
            settings.category_labels.get(str(value), _format(value, settings.format))
            if settings.category_labels
            else _format(value, settings.format)
            for value in values
        )
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
            compiled = _compiled_annotation(annotation)
            if compiled not in annotations:
                annotations.append(compiled)
    scene.legends = legends
    scene.annotations = annotations


__all__ = [
    "CompiledAnnotation",
    "CompiledLegend",
    "collect_overlays",
]
