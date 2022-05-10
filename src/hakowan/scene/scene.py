"""Scene module"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Point:
    """A Point with radius and color."""

    center: np.ndarray
    radius: float
    color: Optional[np.ndarray] = None
    material: str = "diffuse"


@dataclass
class Segment:
    """A line segment with radius and color."""

    p0: np.ndarray
    p1: np.ndarray
    radius: float
    color: Optional[np.ndarray] = None
    material: str = "diffuse"


@dataclass
class Surface:
    """Generic surface represented by triangle mesh."""

    vertices: np.ndarray
    triangles: np.ndarray
    normals: Optional[np.ndarray] = None
    uvs: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    material: str = "diffuse"


@dataclass
class Scene:
    """A scene consists of a list of points, segments and surfaces."""

    points: list[Point] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    surfaces: list[Surface] = field(default_factory=list)
