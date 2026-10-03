"""Pillow compositor for backend-neutral legends and annotations."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence, TypedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..compiler.overlay import (
    CompiledAnnotation,
    CompiledLegend,
    CompiledWorldAnnotation,
)
from ..setup.sensor import Orthographic, Perspective, Sensor
from .image_io import _write


class _CalloutPolicy(TypedDict):
    edge: int
    anchor_radius: int
    gap: int
    gap_growth: float
    gap_step: int
    footprint: list[tuple[float, float]]


CALLOUT_POLICY: _CalloutPolicy = {
    "edge": 6,
    "anchor_radius": 12,
    "gap": 20,
    "gap_growth": 1.5,
    "gap_step": 15,
    "footprint": [
        (0.03, 0.03),
        (0.5, 0.03),
        (0.97, 0.03),
        (0.03, 0.5),
        (0.5, 0.5),
        (0.97, 0.5),
        (0.03, 0.97),
        (0.5, 0.97),
        (0.97, 0.97),
    ],
}


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


def _alpha_composite_rectangle(
    image: Image.Image,
    bounds: tuple[float, float, float, float],
    color: tuple[int, int, int],
    alpha: int,
) -> None:
    """Blend a translucent rectangle over the existing RGBA pixels."""
    left, top, right, bottom = bounds
    x0, y0 = max(0, int(left)), max(0, int(top))
    x1, y1 = min(image.width, int(right) + 1), min(image.height, int(bottom) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    bounds = (x0, y0, x1, y1)
    base = image.crop(bounds)
    panel = Image.new("RGBA", base.size, (*color, alpha))
    image.paste(Image.alpha_composite(base, panel), (x0, y0))


def _draw_annotations(
    image: Image.Image,
    annotations: Sequence[CompiledAnnotation],
    image_origin_x: int,
    image_width: int,
    occupied: list[tuple[float, float, float, float]] | None = None,
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
        if occupied is not None:
            occupied.append(
                (
                    x - padding,
                    y - padding,
                    x + text_width + padding,
                    y + max(text_height, bbox[3]) + padding,
                )
            )
        if annotation.background is not None:
            _alpha_composite_rectangle(
                image,
                (
                    x - padding,
                    y - padding,
                    x + text_width + padding,
                    y + text_height + padding,
                ),
                _rgb(annotation.background),
                220,
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
    film_size: tuple[int, int] | None = None,
    crop_offset: tuple[int, int] = (0, 0),
) -> list[tuple[CompiledWorldAnnotation, float, float]]:
    """Project anchors into output pixels, accounting for a cropped film."""
    width, height = size
    film_width, film_height = film_size or size
    crop_x, crop_y = crop_offset
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
            if not np.all(np.isfinite(ndc)) or abs(ndc[2]) > 1.0:
                continue
            x = (ndc[0] + 1) * film_width / 2 - crop_x
            y = (1 - ndc[1]) * film_height / 2 - crop_y
            if 0 <= x < width and 0 <= y < height:
                result.append((annotation, x, y))
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
        # Orthographic scale is the full vertical camera extent. The same
        # projection is used by the WebGL and Mitsuba backends.
        vertical_extent = float(sensor.scale)
        horizontal_extent = vertical_extent * film_width / film_height
    elif isinstance(sensor, Perspective):
        axis = sensor.fov_axis
        reference = {
            "x": film_width,
            "y": film_height,
            "larger": max(film_width, film_height),
            "diagonal": np.hypot(film_width, film_height),
        }.get(axis, min(film_width, film_height))
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
            px = (0.5 + x / horizontal_extent) * film_width
            py = (0.5 - y / vertical_extent) * film_height
        else:
            px = film_width / 2 + x / (depth * tangent_per_pixel)
            py = film_height / 2 - y / (depth * tangent_per_pixel)
        px -= crop_x
        py -= crop_y
        if np.isfinite(px) and np.isfinite(py) and 0 <= px < width and 0 <= py < height:
            result.append((annotation, px, py))
    return result


def _offset_label_origin(
    anchor: tuple[float, float],
    panel_size: tuple[float, float],
    offset: tuple[float, float],
) -> tuple[float, float]:
    """Return the panel top-left for an explicit screen-space offset.

    ``offset`` is the vector from the projected anchor to the panel edge that
    faces it. ``+x`` is right and ``+y`` is down. A zero component centers the
    panel on that axis, so ``(0, -10)`` puts the bottom edge 10 px above the
    anchor. The WebGL viewer duplicates this rule in ``offsetLabelOrigin``.
    """
    ax, ay = anchor
    width, height = panel_size
    dx, dy = offset
    if dx > 0:
        left = ax + dx
    elif dx < 0:
        left = ax + dx - width
    else:
        left = ax - width / 2
    if dy > 0:
        top = ay + dy
    elif dy < 0:
        top = ay + dy - height
    else:
        top = ay - height / 2
    return left, top


def _place_world_label(
    anchor: tuple[float, float],
    panel_size: tuple[float, float],
    viewport: tuple[int, int],
    occupied: Sequence[tuple[float, float, float, float]],
    geometry_at: Callable[[float, float], bool] | None = None,
    anchors: Sequence[tuple[float, float]] = (),
) -> tuple[float, float, float, float] | None:
    """Return the first legal static callout rectangle (left, top, right, bottom)."""
    ax, ay = anchor
    width, height = panel_size
    viewport_width, viewport_height = viewport
    edge = CALLOUT_POLICY["edge"]
    if width + 2 * edge > viewport_width or height + 2 * edge > viewport_height:
        return None
    radius = CALLOUT_POLICY["anchor_radius"]
    gap = CALLOUT_POLICY["gap"]
    while gap <= np.hypot(viewport_width, viewport_height):
        candidates = (
            (ax + gap, ay - gap - height),
            (ax - gap - width, ay - gap - height),
            (ax + gap, ay + gap),
            (ax - gap - width, ay + gap),
            (ax - width / 2, ay - gap - height),
            (ax - width / 2, ay + gap),
            (ax - gap - width, ay - height / 2),
            (ax + gap, ay - height / 2),
        )
        for x, y in candidates:
            left = max(edge, min(viewport_width - width - edge, x))
            top = max(edge, min(viewport_height - height - edge, y))
            right, bottom = left + width, top + height
            if (
                left < ax + radius
                and right > ax - radius
                and top < ay + radius
                and bottom > ay - radius
            ):
                continue
            if any(
                other != anchor
                and left < other[0] + radius
                and right > other[0] - radius
                and top < other[1] + radius
                and bottom > other[1] - radius
                for other in anchors
            ):
                continue
            if any(
                left < other_right
                and right > other_left
                and top < other_bottom
                and bottom > other_top
                for other_left, other_top, other_right, other_bottom in occupied
            ):
                continue
            if geometry_at is not None and any(
                geometry_at(left + width * u, top + height * v)
                for u, v in CALLOUT_POLICY["footprint"]
            ):
                continue
            return left, top, right, bottom
        # Match JavaScript Math.round, including positive half-integers.
        gap = int(gap * CALLOUT_POLICY["gap_growth"] + CALLOUT_POLICY["gap_step"] + 0.5)
    return None


def _draw_world_annotations(
    image: Image.Image,
    annotations: Sequence[tuple[CompiledWorldAnnotation, float, float]],
    occupied: list[tuple[float, float, float, float]],
    geometry_at: Callable[[float, float], bool] | None = None,
) -> None:
    """Lay out labels in screen space while retaining leaders to their anchors."""
    draw = ImageDraw.Draw(image, "RGBA")
    anchor_points = tuple((x, y) for _, x, y in annotations)
    for annotation_index, (annotation, ax, ay) in enumerate(annotations):
        font = _font(annotation.font_size)
        max_text_width = min(260, image.width * 0.35)
        lines = []
        for paragraph in annotation.text.split("\n"):
            remaining = paragraph
            while remaining and draw.textlength(remaining, font=font) > max_text_width:
                end = 1
                while (
                    end < len(remaining)
                    and draw.textlength(remaining[: end + 1], font=font)
                    <= max_text_width
                ):
                    end += 1
                # Prefer a word boundary, but split long words like overflow-wrap:anywhere.
                boundary = max(
                    remaining.rfind(" ", 0, end), remaining.rfind("\t", 0, end)
                )
                if boundary > 0:
                    end = boundary + 1
                lines.append(remaining[:end])
                remaining = remaining[end:]
            lines.append(remaining)
        line_height = annotation.font_size * 1.25
        text_width = max(draw.textlength(line, font=font) for line in lines)
        panel_width = int(np.ceil(text_width + annotation.padding * 2))
        panel_height = int(np.ceil(len(lines) * line_height + annotation.padding * 2))
        if annotation.offset is not None:
            left, top = _offset_label_origin(
                (ax, ay), (panel_width, panel_height), annotation.offset
            )
            placement = (left, top, left + panel_width, top + panel_height)
        else:
            placement = _place_world_label(
                (ax, ay),
                (panel_width, panel_height),
                image.size,
                occupied,
                geometry_at,
                tuple(
                    point
                    for i, point in enumerate(anchor_points)
                    if i != annotation_index
                ),
            )
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
        if annotation.arrow and length > 0:
            ux, uy = vx / length, vy / length
            draw.line((end_x, end_y, ax, ay), fill=contrast, width=4)
            draw.line((end_x, end_y, ax, ay), fill=color, width=2)
            base_x, base_y = ax - ux * 8, ay - uy * 8
            draw.polygon(
                (
                    (ax, ay),
                    (base_x - uy * 6, base_y + ux * 6),
                    (base_x + uy * 6, base_y - ux * 6),
                ),
                fill=contrast,
            )
            draw.polygon(
                (
                    (ax, ay),
                    (base_x - uy * 4, base_y + ux * 4),
                    (base_x + uy * 4, base_y - ux * 4),
                ),
                fill=color,
            )
        if annotation.background is not None:
            _alpha_composite_rectangle(
                image, placement, _rgb(annotation.background), 220
            )
        for index, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            draw.text(
                (
                    left + annotation.padding - bbox[0],
                    top
                    + annotation.padding
                    + index * line_height
                    + (line_height - (bbox[3] - bbox[1])) / 2
                    - bbox[1],
                ),
                line,
                fill=color,
                font=font,
                stroke_width=1,
                stroke_fill=contrast,
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
    geometry_mask: np.ndarray | None = None,
    geometry_at: Callable[[float, float], bool] | None = None,
    film_size: tuple[int, int] | None = None,
    crop_offset: tuple[int, int] = (0, 0),
) -> Image.Image:
    """Overlay legends, fixed annotations and camera-projected callouts.

    Geometry is sampled at nine panel footprint points, using ``geometry_at``
    preferentially or ``geometry_mask`` in source-image pixel coordinates.
    Transparent sources supply an alpha foreground mask by default; opaque
    sources do not imply that the whole viewport contains geometry.
    """
    source = image.convert("RGBA")
    if background is None:
        output = source.copy()
    else:
        canvas_color = {
            "dark": (18, 18, 18, 255),
            "light": (255, 255, 255, 255),
            "black": (0, 0, 0, 255),
        }[background]
        output = Image.new("RGBA", source.size, canvas_color)
        output.alpha_composite(source)

    occupied: list[tuple[float, float, float, float]] = []
    left = [legend for legend in legends if legend.position == "left"]
    right = [legend for legend in legends if legend.position == "right"]
    margin = 8 if source.width > 16 and source.height > 16 else 0
    available_width = source.width - 2 * margin
    available_height = source.height - 2 * margin
    if left:
        width = min(max(legend.width for legend in left), available_width)
        panel = _draw_legend_panel(left, width, available_height)
        output.alpha_composite(panel, (margin, margin))
        occupied.append((margin, margin, margin + panel.width, margin + panel.height))
    if right:
        width = min(max(legend.width for legend in right), available_width)
        panel = _draw_legend_panel(right, width, available_height)
        output.alpha_composite(panel, (source.width - width - margin, margin))
        occupied.append(
            (
                source.width - width - margin,
                margin,
                source.width - margin,
                margin + panel.height,
            )
        )
    _draw_annotations(output, annotations, 0, source.width, occupied)
    if world_annotations:
        if geometry_at is None:
            if geometry_mask is None:
                alpha = np.asarray(source.getchannel("A"))
                if np.any(alpha < 255):
                    geometry_mask = alpha > 0
            if geometry_mask is not None:

                def geometry_at(x: float, y: float) -> bool:
                    return bool(
                        geometry_mask[
                            min(max(int(y), 0), geometry_mask.shape[0] - 1),
                            min(max(int(x), 0), geometry_mask.shape[1] - 1),
                        ]
                    )

        projected = _project_world_annotations(
            world_annotations,
            source.size,
            sensor=sensor,
            world_to_camera=world_to_camera,
            projection=projection,
            film_size=film_size,
            crop_offset=crop_offset,
        )
        _draw_world_annotations(output, projected, occupied, geometry_at)
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
    geometry_mask: np.ndarray | None = None,
    geometry_at: Callable[[float, float], bool] | None = None,
    film_size: tuple[int, int] | None = None,
    crop_offset: tuple[int, int] = (0, 0),
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
            source,
            legends,
            annotations,
            background,
            world_annotations=world_annotations,
            sensor=sensor,
            world_to_camera=world_to_camera,
            projection=projection,
            geometry_mask=geometry_mask,
            geometry_at=geometry_at,
            film_size=film_size,
            crop_offset=crop_offset,
        )
    _write(result, filename)
    return True


__all__ = ["CALLOUT_POLICY", "composite_overlay_file", "composite_overlays"]
