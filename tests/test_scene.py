import numpy as np
import hakowan
from hakowan.scene.scene import Scene
from hakowan.scene.scene_utils import generate_scene

from .test_utils import triangle_data_frame, quad_data_frame


class TestScene:
    def test_triangle(self, triangle_data_frame):
        base = hakowan.layer(data=triangle_data_frame)
        l0 = base.mark(hakowan.SURFACE)
        scene = generate_scene(l0)

        assert len(scene.surfaces) == 1
        assert len(scene.surfaces[0].triangles) == 1
        assert len(scene.surfaces[0].vertices) == 3
        assert len(scene.surfaces[0].colors) == 3

    def test_quad(self, quad_data_frame):
        base = hakowan.layer(data=quad_data_frame)
        l0 = base.mark(hakowan.SURFACE).transform(np.identity(4) * 2)
        scene = generate_scene(l0)

        assert len(scene.surfaces) == 1
        assert len(scene.surfaces[0].triangles) == 2
        assert len(scene.surfaces[0].vertices) == 6
        assert len(scene.surfaces[0].colors) == 6

    def test_composite(self, triangle_data_frame, quad_data_frame):
        base = hakowan.layer(mark=hakowan.SURFACE)
        l0 = base.data(triangle_data_frame).channel(color="black")
        l1 = base.data(quad_data_frame).channel(color="white")

        l2 = l0 + l1
        scene = generate_scene(l2)

        assert len(scene.surfaces) == 2
        assert len(scene.surfaces[0].triangles) == 1
        assert len(scene.surfaces[1].triangles) == 2
        assert len(scene.surfaces[0].vertices) == 3
        assert len(scene.surfaces[1].vertices) == 6
        assert np.all(scene.surfaces[0].colors[0] == [0, 0, 0])
        assert np.all(scene.surfaces[1].colors[0] == [1, 1, 1])
