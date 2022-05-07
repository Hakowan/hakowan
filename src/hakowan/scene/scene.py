"""Flattened scene module"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Shape:
    material: str = None


@dataclass
class Point(Shape):
    center: np.ndarray
    radius: float
    color: np.ndarray = None


@dataclass
class Edge(Shape):
    p0: np.ndarray
    p1: np.ndarray
    radius: float
    color: np.ndarray = None


@dataclass
class Surface(Shape):
    vertices: np.ndarray
    triangles: np.ndarray
    normal: np.ndarray = None
    uv: np.ndarray = None
    color: np.ndarray = None


@dataclass
class Scene:
    points: list[Point]
    edges: list[Edge]
    surfaces: list[Surface]
