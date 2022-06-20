import numpy as np
import hakowan

from .test_utils import (
    triangle_data_frame,
    quad_data_frame,
    triangle_boundary_data_frame,
)

class TestMitsuba:
    def test_points(self, triangle_data_frame):
        base = hakowan.layer(data=triangle_data_frame)
        l0 = base.mark(hakowan.POINT)
        config = hakowan.RenderConfig()
        config.filename = "point.png"
        hakowan.render(l0, config)

    def test_segments(self, triangle_boundary_data_frame):
        base = hakowan.layer(data=triangle_boundary_data_frame)
        l0 = base.mark(hakowan.CURVE)
        config = hakowan.RenderConfig()
        config.filename = "curve.png"
        hakowan.render(l0, config)
