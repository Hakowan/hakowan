import numpy as np
import hakowan
from hakowan.scene.scene import Scene
from hakowan.scene.scene_utils import generate_scene
from hakowan.grammar.layer_data import Attribute

from .test_utils import (
    triangle_data_frame,
    quad_data_frame,
    triangle_boundary_data_frame,
)


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

    def test_tri_quad_composite(self, triangle_data_frame, quad_data_frame):
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

    def test_points(self, triangle_data_frame):
        base = hakowan.layer(data=triangle_data_frame)
        l0 = base.mark(hakowan.POINT)
        scene = generate_scene(l0)

        assert len(scene.points) == 3
        assert len(scene.surfaces) == 0

        for p in scene.points:
            assert p.radius == hakowan.common.default.DEFAULT_SIZE

        l1 = l0.channel(size=0.5)
        scene = generate_scene(l1)
        for p in scene.points:
            assert p.radius == 0.5

        triangle_data_frame.attributes["my_size"] = Attribute(
            np.array([0, 1, 2], dtype=float), np.array([0, 1, 2])
        )
        l2 = l0.channel(size="my_size", size_map=lambda x: x / 2)
        scene = generate_scene(l2)
        assert scene.points[0].radius == 0
        assert scene.points[1].radius == 0.5
        assert scene.points[2].radius == 1

    def test_segments(self, triangle_boundary_data_frame):
        base = hakowan.layer(data=triangle_boundary_data_frame)
        l0 = base.mark(hakowan.CURVE)
        scene = generate_scene(l0)
        assert len(scene.points) == 0
        assert len(scene.surfaces) == 0
        assert len(scene.segments) == 3
        assert np.all(scene.segments[0].radii == 1)
        assert np.all(scene.segments[1].radii == 1)
        assert np.all(scene.segments[2].radii == 1)

        l1 = l0.channel(size=0.5)
        scene = generate_scene(l1)
        assert np.all(scene.segments[0].radii == 0.5)
        assert np.all(scene.segments[1].radii == 0.5)
        assert np.all(scene.segments[2].radii == 0.5)

        triangle_boundary_data_frame.attributes["my_size"] = Attribute(
            np.array([1, 2, 3], dtype=float),
            np.array([[0, 0], [1, 1], [2, 2]], dtype=int),
        )
        l2 = l0.channel(size="my_size", size_map="identity")
        scene = generate_scene(l2)
        assert np.all(scene.segments[0].radii == 1)
        assert np.all(scene.segments[1].radii == 2)
        assert np.all(scene.segments[2].radii == 3)
