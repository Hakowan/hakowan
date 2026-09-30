"""Pillow compositor for backend-neutral legends and annotations."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..compiler.overlay import CompiledAnnotation, CompiledLegend, CompiledWorldAnnotation
from ..setup.sensor import Orthographic, Perspective, Sensor
from .image_io import _write


@lru_cache(maxsize=None)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Pillow's built-in Aileron fallback is soft at small annotation sizes.
    # Prefer each platform's native UI font, then the common Linux font.
    candidates = (
        "/System/Library/Fonts/SFNS.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _rgb(color: tuple[float, float, float]) -> tuple[int, int, int]:
    return (
        int(round(np.clip(color[0], 0.0, 1.0) * 255)),
        int(round(np.clip(color[1], 0.0, 1.0) * 255)),
        int(round(np.clip(color[2], 0.0, 1.0) * 255)),
    )


def _gradient(
    colors: tuple[tuple[float, float, float], ...], width: int, height: int
) -> Image.Image:
    stops = np.asarray(colors, dtype=np.float64)
    coordinates = np.linspace(0.0, 1.0, len(stops))
    samples = np.linspace(1.0, 0.0, height)
    result = np.empty((height, width, 3), dtype=np.uint8)
    for channel in range(3):
        values = np.interp(samples, coordinates, stops[:, channel])
        result[:, :, channel] = np.clip(values[:, None] * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def _rounded_panel(width: int, height: int) -> Image.Image:
    scale = 4
    mask = Image.new("L", (width * scale, height * scale), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, width * scale - 1, height * scale - 1),
        radius=4 * scale,
        fill=255,
    )
    mask = mask.resize((width, height), Image.Resampling.LANCZOS)
    mask = mask.point(lambda value: round(value * 148 / 255))
    panel = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    panel.putalpha(mask)
    return panel


def _draw_legend_panel(
    legends: Sequence[CompiledLegend], width: int, height: int
) -> Image.Image:
    content = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(content)
    title_font = _font(14)
    label_font = _font(12)
    text_color = (238, 238, 238, 255)
    muted_color = (187, 187, 187, 255)
    outline_color = (204, 204, 204, 255)
    y = 12
    for legend in legends:
        title = legend.title + (f" ({legend.units})" if legend.units else "")
        draw.text((12, y), title, fill=text_color, font=title_font)
        y += 24
        if legend.scale:
            draw.text(
                (12, y),
                " → ".join(legend.scale),
                fill=muted_color,
                font=label_font,
            )
            y += 18
        if legend.categories:
            for label, color in zip(legend.labels, legend.colors):
                draw.rectangle(
                    (12, y + 2, 28, y + 18),
                    fill=(*_rgb(color), 255),
                    outline=outline_color,
                )
                draw.text((36, y + 2), label, fill=text_color, font=label_font)
                y += 22
        else:
            bar_height = max(80, min(180, height - y - 40))
            bar = _gradient(legend.colors, 18, bar_height).convert("RGBA")
            content.alpha_composite(bar, (12, y))
            draw.rectangle((12, y, 30, y + bar_height), outline=outline_color)
            count = max(len(legend.labels), 1)
            for index, label in enumerate(reversed(legend.labels)):
                fraction = index / max(count - 1, 1)
                tick_y = int(round(y + fraction * bar_height))
                draw.line((30, tick_y, 35, tick_y), fill=outline_color)
                draw.text((40, tick_y - 7), label, fill=text_color, font=label_font)
            y += bar_height + 22
        y += 10

    panel_height = min(y + 2, height)
    panel = _rounded_panel(width, panel_height)
    panel.alpha_composite(content.crop((0, 0, width, panel_height)))
    return panel


def _draw_annotations(
    image: Image.Image,
    annotations: Sequence[CompiledAnnotation],
    image_origin_x: int,
    image_width: int,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    for annotation in annotations:
        font = _font(annotation.font_size)
        x = image_origin_x + annotation.position[0] * image_width
        y = annotation.position[1] * image.height
        bbox = draw.textbbox((0, 0), annotation.text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if annotation.anchor == "center":
            x -= text_width / 2
        elif annotation.anchor == "right":
            x -= text_width
        padding = annotation.padding
        if annotation.background is not None:
            draw.rectangle(
                (
                    x - padding,
                    y - padding,
                    x + text_width + padding,
                    y + text_height + padding,
                ),
                fill=(*_rgb(annotation.background), 220),
            )
        draw.text(
            (x, y), annotation.text, fill=(*_rgb(annotation.color), 255), font=font
        )


def _project_world_annotations(
    annotations: Sequence[CompiledWorldAnnotation],
    size: tuple[int, int],
    *,
    sensor: Sensor | None,
    world_to_camera: np.ndarray | None,
    projection: np.ndarray | None,
) -> list[tuple[CompiledWorldAnnotation, float, float]]:
    """Project world anchors into image pixels; reject clipped and rear-facing points."""
    width, height = size
    result: list[tuple[CompiledWorldAnnotation, float, float]] = []
    if world_to_camera is not None and projection is not None:
        transform = np.asarray(projection, dtype=np.float64) @ np.asarray(
            world_to_camera, dtype=np.float64
        )
        for annotation in annotations:
            clip = transform @ np.array((*annotation.position, 1.0), dtype=np.float64)
            if not np.all(np.isfinite(clip)) or clip[3] <= 1e-10:
                continue
            ndc = clip[:3] / clip[3]
            if np.any(np.abs(ndc) > 1.0):
                continue
            result.append(
                (annotation, (ndc[0] + 1) * width / 2, (1 - ndc[1]) * height / 2)
            )
        return result
    if sensor is None:
        raise ValueError("World annotations require a sensor or camera matrices.")

    # Mitsuba's camera looks along -Z in its local coordinates, with +X right.
    eye = np.asarray(sensor.location, dtype=np.float64)
    backward = eye - np.asarray(sensor.target, dtype=np.float64)
    backward /= np.linalg.norm(backward)
    right = np.cross(np.asarray(sensor.up, dtype=np.float64), backward)
    if np.linalg.norm(right) < 1e-9:
        fallback = (
            np.array([1.0, 0.0, 0.0])
            if abs(backward[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        right = np.cross(fallback, backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    if isinstance(sensor, Orthographic):
        # Mitsuba's film covers [-1, 1] horizontally before the configured
        # scale transform; image aspect changes the vertical extent instead.
        horizontal_extent = float(sensor.scale)
        vertical_extent = horizontal_extent * height / width
    elif isinstance(sensor, Perspective):
        axis = sensor.fov_axis
        reference = {
            "x": width,
            "y": height,
            "larger": max(width, height),
            "diagonal": np.hypot(width, height),
        }.get(axis, min(width, height))
        tangent_per_pixel = 2 * np.tan(np.radians(float(sensor.fov)) / 2) / reference
    else:
        raise TypeError(f"Unsupported world-annotation sensor: {type(sensor)!r}")
    for annotation in annotations:
        displacement = np.asarray(annotation.position, dtype=np.float64) - eye
        depth = -float(np.dot(displacement, backward))
        if not np.isfinite(depth) or not (sensor.near_clip <= depth <= sensor.far_clip):
            continue
        x = float(np.dot(displacement, right))
        y = float(np.dot(displacement, up))
        if isinstance(sensor, Orthographic):
            px = (0.5 + x / horizontal_extent) * width
            py = (0.5 - y / vertical_extent) * height
        else:
            px = width / 2 + x / (depth * tangent_per_pixel)
            py = height / 2 - y / (depth * tangent_per_pixel)
        if np.isfinite(px) and np.isfinite(py) and 0 <= px < width and 0 <= py < height:
            result.append((annotation, px, py))
    return result


def _draw_world_annotations(
    image: Image.Image,
    annotations: Sequence[tuple[CompiledWorldAnnotation, float, float]],
) -> None:
    """Lay out labels in screen space while retaining leaders to their anchors."""
    draw = ImageDraw.Draw(image, "RGBA")
    occupied: list[tuple[float, float, float, float]] = []
    anchors = [(x, y) for _, x, y in annotations]
    for annotation, ax, ay in annotations:
        font = _font(annotation.font_size)
        bbox = draw.multiline_textbbox((0, 0), annotation.text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        panel_width = text_width + annotation.padding * 2
        panel_height = text_height + annotation.padding * 2
        placement = None
        # Prefer adjacent diagonals, then search farther away when labels collide.
        directions = (
            (1, -1), (1, 1), (-1, -1), (-1, 1),
            (1, 0), (-1, 0), (0, -1), (0, 1),
        )
        for distance in (18, 30, 46, 68, 96, 136, 188):
            for dx, dy in directions:
                if dx > 0:
                    left = ax + distance
                elif dx < 0:
                    left = ax - distance - panel_width
                else:
                    left = ax - panel_width / 2
                if dy > 0:
                    top = ay + distance
                elif dy < 0:
                    top = ay - distance - panel_height
                else:
                    top = ay - panel_height / 2
                right, bottom = left + panel_width, top + panel_height
                if (
                    left < 1 or top < 1
                    or right >= image.width - 1 or bottom >= image.height - 1
                ):
                    continue
                if any(
                    left - 4 < x < right + 4 and top - 4 < y < bottom + 4
                    for x, y in anchors
                ):
                    continue
                if any(
                    left - 4 < existing_right and right + 4 > existing_left
                    and top - 4 < existing_bottom and bottom + 4 > existing_top
                    for existing_left, existing_top, existing_right, existing_bottom
                    in occupied
                ):
                    continue
                placement = (left, top, right, bottom)
                break
            if placement is not None:
                break
        if placement is None:
            continue
        left, top, right, bottom = placement
        occupied.append(placement)
        # Connect the closest point of the panel to the source point. The tip
        # points toward the anchor rather than intruding into the text panel.
        end_x = min(max(ax, left), right)
        end_y = min(max(ay, top), bottom)
        vx, vy = ax - end_x, ay - end_y
        length = float(np.hypot(vx, vy))
        color = (*_rgb(annotation.color), 255)
        # Thin contrast strokes keep default white labels visible on light
        # renders without imposing a background panel on transparent images.
        contrast = (0, 0, 0, 255) if sum(color[:3]) > 382 else (255, 255, 255, 255)
        if length > 0:
            ux, uy = vx / length, vy / length
            draw.line((end_x, end_y, ax, ay), fill=contrast, width=4)
            draw.line((end_x, end_y, ax, ay), fill=color, width=2)
            base_x, base_y = ax - ux * 8, ay - uy * 8
            draw.polygon(
                ((ax, ay), (base_x - uy * 6, base_y + ux * 6),
                 (base_x + uy * 6, base_y - ux * 6)),
                fill=contrast,
            )
            draw.polygon(
                ((ax, ay), (base_x - uy * 4, base_y + ux * 4),
                 (base_x + uy * 4, base_y - ux * 4)),
                fill=color,
            )
        if annotation.background is not None:
            draw.rectangle(placement, fill=(*_rgb(annotation.background), 220))
        draw.multiline_text(
            (left + annotation.padding - bbox[0], top + annotation.padding - bbox[1]),
            annotation.text, fill=color, font=font,
            stroke_width=1, stroke_fill=contrast,
        )


def composite_overlays(
    image: Image.Image,
    legends: Sequence[CompiledLegend],
    annotations: Sequence[CompiledAnnotation],
    background: str | None = None,
    *,
    world_annotations: Sequence[CompiledWorldAnnotation] = (),
    sensor: Sensor | None = None,
    world_to_camera: np.ndarray | None = None,
    projection: np.ndarray | None = None,
) -> Image.Image:
    """Overlay legends, fixed annotations and camera-projected callouts."""
    source = image.convert("RGBA")
    if background is None:
        output = source.copy()
    else:
        canvas_color = (
            (18, 18, 18, 255) if background == "dark" else (255, 255, 255, 255)
        )
        output = Image.new("RGBA", source.size, canvas_color)
        output.alpha_composite(source)


    if world_annotations:
        projected = _project_world_annotations(
            world_annotations, source.size, sensor=sensor,
            world_to_camera=world_to_camera, projection=projection,
        )
        _draw_world_annotations(output, projected)
    left = [legend for legend in legends if legend.position == "left"]
    right = [legend for legend in legends if legend.position == "right"]
    margin = 8 if source.width > 16 and source.height > 16 else 0
    available_width = source.width - 2 * margin
    available_height = source.height - 2 * margin
    if left:
        width = min(max(legend.width for legend in left), available_width)
        panel = _draw_legend_panel(left, width, available_height)
        output.alpha_composite(panel, (margin, margin))
    if right:
        width = min(max(legend.width for legend in right), available_width)
        panel = _draw_legend_panel(right, width, available_height)
        output.alpha_composite(panel, (source.width - width - margin, margin))
    _draw_annotations(output, annotations, 0, source.width)
    return output


def composite_overlay_file(
    path: str | Path,
    legends: Sequence[CompiledLegend],
    annotations: Sequence[CompiledAnnotation],
    background: str | None = None,
    *,
    world_annotations: Sequence[CompiledWorldAnnotation] = (),
    sensor: Sensor | None = None,
    world_to_camera: np.ndarray | None = None,
    projection: np.ndarray | None = None,
) -> bool:
    """Composite overlays into a Pillow-readable image file.

    Returns ``False`` for HDR formats that cannot carry a raster text overlay.
    """
    filename = Path(path)
    if not legends and not annotations and not world_annotations and background is None:
        return True
    if filename.suffix.lower() in {".exr", ".hdr"}:
        return False
    with Image.open(filename) as source:
        result = composite_overlays(
            source, legends, annotations, background,
            world_annotations=world_annotations, sensor=sensor,
            world_to_camera=world_to_camera, projection=projection,
        )
    _write(result, filename)
    return True


__all__ = ["composite_overlay_file", "composite_overlays"]
