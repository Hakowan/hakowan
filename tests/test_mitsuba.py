import numpy as np
import hakowan
from hakowan.scene.scene import Scene
from hakowan.scene.scene_utils import generate_scene
from hakowan.grammar.layer_data import Attribute
from hakowan.backend.render import render

from .test_utils import (
    triangle_data_frame,
    quad_data_frame,
    triangle_boundary_data_frame,
)

class TestMitsuba:
    def test_triangle(self, triangle_data_frame):
        base = hakowan.layer(data=triangle_data_frame)
        l0 = base.mark(hakowan.POINT)
        render(l0, "mitsuba", "tmp.png")
