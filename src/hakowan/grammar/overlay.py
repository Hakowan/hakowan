"""Semantic 2D overlays shared by rendering backends."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from ..common.color import ColorLike


@dataclass(frozen=True, slots=True)
class Legend:
    """Configuration for a scalar-field colorbar or categorical legend.

    Attributes:
        title: Display title. ``None`` uses the attribute name.
        units: Optional physical units appended to the title.
        ticks: Number of continuous ticks. Ignored for categorical legends.
        format: Python numeric format specifier, such as ``".3g"``.
        position: Side of the image or viewer containing the legend.
        category_labels: Optional mapping from stringified values to labels.
        width: Raster legend panel width in pixels.

    """

    title: str | None = None
    units: str | None = None
    ticks: int = 5
    format: str = ".3g"
    position: Literal["left", "right"] = "right"
    category_labels: dict[str, str] | None = None
    width: int = 180

    def __post_init__(self) -> None:
        """Validate tick count, panel width, and numeric format."""
        if self.ticks < 2:
            raise ValueError("Legend.ticks must be at least 2.")
        if self.width < 80:
            raise ValueError("Legend.width must be at least 80 pixels.")
        try:
            format(1.0, self.format)
        except ValueError as exc:
            raise ValueError(f"Invalid Legend.format: {self.format!r}") from exc


@dataclass(frozen=True, slots=True)
class Annotation:
    """Screen-space text annotation drawn over the rendered figure.

    ``position`` uses normalized image coordinates with origin at the top-left.
    Values outside ``[0, 1]`` are rejected to keep output deterministic.
    """

    text: str
    position: tuple[float, float] = (0.02, 0.02)
    color: ColorLike = "white"
    font_size: int = 16
    anchor: Literal["left", "center", "right"] = "left"
    background: ColorLike | None = None
    padding: int = 4

    def __post_init__(self) -> None:
        """Validate annotation text, normalized position, and pixel sizes."""
        if not self.text:
            raise ValueError("Annotation.text must not be empty.")
        if len(self.position) != 2:
            raise ValueError(
                "Annotation.position must contain exactly two coordinates."
            )
        if not all(0.0 <= value <= 1.0 for value in self.position):
            raise ValueError("Annotation.position values must be in [0, 1].")
        if self.font_size <= 0:
            raise ValueError("Annotation.font_size must be positive.")
        if self.padding < 0:
            raise ValueError("Annotation.padding must be non-negative.")


@dataclass(frozen=True, slots=True)
class WorldAnnotation:
    """Text callout anchored to a point in a view's local 3D coordinates.

    The label remains screen-facing; renderers project ``position`` through
    the view's geometry transform and the active camera.

    Parameters:
        text: Text string to display.
        position: 3D point in the layer's local coordinates.
        color: Text and arrow color.
        font_size: Font size in pixels.
        background: Background color of the text callout panel.
        padding: Internal padding in pixels around the text.
        offset: Optional screen-space pixel offset ``(dx, dy)`` from the projected
            anchor to the panel edge that faces it. ``+x`` is right and ``+y`` is
            down. A zero component centers the panel on that axis, so
            ``(0, -10)`` places the label 10 pixels above the anchor. ``None``
            keeps automatic placement. An explicit offset is not collision-checked.
        arrow: Whether to draw a leader from the panel to the anchor (default True).
    """

    text: str
    position: tuple[float, float, float]
    color: ColorLike = "white"
    font_size: int = 16
    background: ColorLike | None = "black"
    padding: int = 4
    offset: tuple[float, float] | None = None
    arrow: bool = True

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("WorldAnnotation.text must not be empty.")
        if len(self.position) != 3 or not all(
            math.isfinite(value) for value in self.position
        ):
            raise ValueError(
                "WorldAnnotation.position must contain three finite coordinates."
            )
        if self.font_size <= 0:
            raise ValueError("WorldAnnotation.font_size must be positive.")
        if self.padding < 0:
            raise ValueError("WorldAnnotation.padding must be non-negative.")
        if self.offset is not None:
            if len(self.offset) != 2 or not all(
                math.isfinite(value) for value in self.offset
            ):
                raise ValueError(
                    "WorldAnnotation.offset must contain two finite coordinates."
                )


__all__ = ["Annotation", "Legend", "WorldAnnotation"]
