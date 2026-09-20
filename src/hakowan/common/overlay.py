"""Pillow compositor for backend-neutral legends and annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..compiler.overlay import CompiledAnnotation, CompiledLegend


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


def _draw_legend_panel(
    legends: Sequence[CompiledLegend], width: int, height: int
) -> Image.Image:
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    title_font = _font(14)
    label_font = _font(12)
    y = 12
    for legend in legends:
        title = legend.title + (f" ({legend.units})" if legend.units else "")
        draw.text((12, y), title, fill="black", font=title_font)
        y += 24
        if legend.scale:
            draw.text(
                (12, y),
                " → ".join(legend.scale),
                fill=(80, 80, 80),
                font=label_font,
            )
            y += 18
        if legend.categories:
            for label, color in zip(legend.labels, legend.colors):
                draw.rectangle(
                    (12, y + 2, 28, y + 18), fill=_rgb(color), outline="black"
                )
                draw.text((36, y + 2), label, fill="black", font=label_font)
                y += 22
        else:
            bar_height = max(80, min(180, height - y - 40))
            bar = _gradient(legend.colors, 18, bar_height)
            panel.paste(bar, (12, y))
            draw.rectangle((12, y, 30, y + bar_height), outline="black")
            count = max(len(legend.labels), 1)
            for index, label in enumerate(reversed(legend.labels)):
                fraction = index / max(count - 1, 1)
                tick_y = int(round(y + fraction * bar_height))
                draw.line((30, tick_y, 35, tick_y), fill="black")
                draw.text((40, tick_y - 7), label, fill="black", font=label_font)
            y += bar_height + 22
        y += 10
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
) -> Image.Image:
    """Return an image with legend side panels and screen annotations."""
    source = image.convert("RGBA")
    left = [legend for legend in legends if legend.position == "left"]
    right = [legend for legend in legends if legend.position == "right"]
    left_width = max((legend.width for legend in left), default=0)
    right_width = max((legend.width for legend in right), default=0)
    output = Image.new(
        "RGBA", (left_width + source.width + right_width, source.height), "white"
    )
    if left:
        output.paste(_draw_legend_panel(left, left_width, source.height), (0, 0))
    output.alpha_composite(source, (left_width, 0))
    if right:
        output.paste(
            _draw_legend_panel(right, right_width, source.height),
            (left_width + source.width, 0),
        )
    _draw_annotations(output, annotations, left_width, source.width)
    return output


def composite_overlay_file(
    path: str | Path,
    legends: Sequence[CompiledLegend],
    annotations: Sequence[CompiledAnnotation],
) -> bool:
    """Composite overlays into a Pillow-readable image file.

    Returns ``False`` for HDR formats that cannot carry a raster text overlay.
    """
    filename = Path(path)
    if not legends and not annotations:
        return True
    if filename.suffix.lower() in {".exr", ".hdr"}:
        return False
    with Image.open(filename) as source:
        result = composite_overlays(source, legends, annotations)
    if filename.suffix.lower() in {".jpg", ".jpeg"}:
        result = result.convert("RGB")
    result.save(filename)
    return True


__all__ = ["composite_overlay_file", "composite_overlays"]
