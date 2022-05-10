"""Scene module"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Shape:
    """The `Shape` class represents something that can be rendred."""

    material: str = None


@dataclass
class Point(Shape):
    """A Point with radius and color."""

    center: np.ndarray = None
    radius: float = None
    color: np.ndarray = None


@dataclass
class Segment(Shape):
    """A line segment with radius and color."""

    p0: np.ndarray = None
    p1: np.ndarray = None
    radius: float = None
    color: np.ndarray = None


@dataclass
class Surface(Shape):
    """Generic surface represented by triangle mesh."""

    vertices: np.ndarray = None
    triangles: np.ndarray = None
    normals: np.ndarray = None
    uvs: np.ndarray = None
    colors: np.ndarray = None


@dataclass
class Scene:
    """A scene consists of a list of points, segments and surfaces."""

    points: list[Point] = None
    segments: list[Segment] = None
    surfaces: list[Surface] = None
