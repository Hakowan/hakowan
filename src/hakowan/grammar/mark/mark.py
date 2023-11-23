from enum import Enum


class Mark(Enum):
    """Mark represents the way data is visualized."""

    Point = 0
    Curve = 1
    Surface = 2


Point = Mark.Point
"""Point is a mark for visualizing data as a point."""

Curve = Mark.Curve
"""Curve is a mark for visualizing data as a curve."""

Surface = Mark.Surface
"""Surface is a mark for visualizing data as a surface."""
