from enum import Enum


class Mark(Enum):
    """Mark represents the way data is visualized."""

    Point = 0
    Curve = 1
    Surface = 2


Point = Mark.Point
Curve = Mark.Curve
Surface = Mark.Surface
