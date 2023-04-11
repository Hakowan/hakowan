"""Scene module"""

from dataclasses import dataclass, field
from typing import Optional, Union
import math
import numpy as np
import numpy.typing as npt

from ..common.color import Color

@dataclass
class Point:
    """A Point with radius and color."""

    center: npt.NDArray
    radius: float
    color: npt.NDArray
    material: str = "diffuse"
    material_preset: str = "none"


@dataclass
class Segment:
    """A line segment with radius and color."""

    vertices: npt.NDArray
    radii: npt.NDArray
    colors: npt.NDArray
    material: str = "diffuse"
    material_preset: str = "none"


@dataclass
class Surface:
    """Generic surface represented by triangle mesh."""

    vertices: npt.NDArray
    triangles: npt.NDArray
    normals: Optional[npt.NDArray] = None
    uvs: Optional[npt.NDArray] = None
    colors: Union[Color, npt.NDArray, None] = None
    material: str = "diffuse"
    material_preset: str = "none"


@dataclass
class Scene:
    """A scene consists of a list of points, segments and surfaces."""

    points: list[Point] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    surfaces: list[Surface] = field(default_factory=list)

    @property
    def bbox(self):
        """Compute the axis-aligned bounding box of the scene"""
        assert len(self.points) > 0 or len(self.segments) > 0 or len(self.surfaces) > 0
        bbox_min = np.array([math.inf, math.inf, math.inf])
        bbox_max = np.array([-math.inf, -math.inf, -math.inf])

        for p in self.points:
            bbox_min = np.minimum(bbox_min, p.center)
            bbox_max = np.maximum(bbox_max, p.center)

        for s in self.segments:
            bbox_min = np.minimum(bbox_min, np.amin(s.vertices, axis=0))
            bbox_max = np.maximum(bbox_max, np.amax(s.vertices, axis=0))

        for m in self.surfaces:
            bbox_min = np.minimum(bbox_min, np.amin(m.vertices, axis=0))
            bbox_max = np.maximum(bbox_max, np.amax(m.vertices, axis=0))

        return bbox_min, bbox_max