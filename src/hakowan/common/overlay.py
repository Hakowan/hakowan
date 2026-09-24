"""Pillow compositor for backend-neutral legends and annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..compiler.overlay import CompiledAnnotation, CompiledLegend
from .image_io import _write


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
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


def composite_overlays(
    image: Image.Image,
    legends: Sequence[CompiledLegend],
    annotations: Sequence[CompiledAnnotation],
    background: str | None = None,
) -> Image.Image:
    """Overlay legend panels and screen annotations within the source image."""
    source = image.convert("RGBA")
    if background is None:
        output = source.copy()
    else:
        canvas_color = (
            (18, 18, 18, 255) if background == "dark" else (255, 255, 255, 255)
        )
        output = Image.new("RGBA", source.size, canvas_color)
        output.alpha_composite(source)

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
) -> bool:
    """Composite overlays into a Pillow-readable image file.

    Returns ``False`` for HDR formats that cannot carry a raster text overlay.
    """
    filename = Path(path)
    if not legends and not annotations and background is None:
        return True
    if filename.suffix.lower() in {".exr", ".hdr"}:
        return False
    with Image.open(filename) as source:
        result = composite_overlays(source, legends, annotations, background)
    _write(result, filename)
    return True


__all__ = ["composite_overlay_file", "composite_overlays"]
