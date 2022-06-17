"""Scene module"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import numpy.typing as npt


@dataclass
class Point:
    """A Point with radius and color."""

    center: npt.ArrayLike
    radius: float
    color: npt.ArrayLike
    material: str = "diffuse"


@dataclass
class Segment:
    """A line segment with radius and color."""

    vertices: npt.ArrayLike
    radii: npt.ArrayLike
    colors: npt.ArrayLike
    material: str = "diffuse"


@dataclass
class Surface:
    """Generic surface represented by triangle mesh."""

    vertices: npt.ArrayLike
    triangles: npt.ArrayLike
    normals: Optional[npt.ArrayLike] = None
    uvs: Optional[npt.ArrayLike] = None
    colors: Optional[npt.ArrayLike] = None
    material: str = "diffuse"


@dataclass
class Scene:
    """A scene consists of a list of points, segments and surfaces."""

    points: list[Point] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    surfaces: list[Surface] = field(default_factory=list)
