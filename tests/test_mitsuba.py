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
        config.dry_run = True
        hakowan.render(l0, config)

    def test_segments(self, triangle_boundary_data_frame):
        base = hakowan.layer(data=triangle_boundary_data_frame)
        l0 = base.mark(hakowan.CURVE)
        config = hakowan.RenderConfig()
        config.filename = "curve.png"
        config.dry_run = True
        hakowan.render(l0, config)

    def test_combined(self, triangle_data_frame, triangle_boundary_data_frame):
        l0 = hakowan.layer(data=triangle_boundary_data_frame)\
                .mark(hakowan.CURVE)
        l1 = hakowan.layer(data = triangle_data_frame)
        l2 = l1.mark(hakowan.POINT)
        l3 = l1.mark(hakowan.SURFACE)

        scene = l0 + l2 + l3

        config = hakowan.RenderConfig()
        config.filename = "combined.png"
        config.dry_run = True
        hakowan.render(scene, config)
