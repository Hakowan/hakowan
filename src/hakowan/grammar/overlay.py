"""Semantic 2D overlays shared by rendering backends."""

from __future__ import annotations

from dataclasses import dataclass
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


__all__ = ["Annotation", "Legend"]
