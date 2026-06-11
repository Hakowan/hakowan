import numpy as np

from hakowan.compiler.scene import Scene
from hakowan.compiler.view import View


class TestComputeGlobalTransform:
    def test_does_not_mutate_view_bbox(self):
        # Two views with distinct, non-overlapping bounding boxes.
        v0 = View()
        v0.bbox = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        v1 = View()
        v1.bbox = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])

        b0 = v0.bbox.copy()
        b1 = v1.bbox.copy()

        scene = Scene(views=[v0, v1])
        scene.compute_global_transform()

        # The first view's stored bbox must remain its own geometry box, not be
        # clobbered with the whole-scene union by the accumulation loop.
        assert np.array_equal(v0.bbox, b0)
        assert np.array_equal(v1.bbox, b1)

    def test_fits_scene_into_unit_sphere(self):
        v0 = View()
        v0.bbox = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        v1 = View()
        v1.bbox = np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])

        scene = Scene(views=[v0, v1])
        scene.compute_global_transform()

        # The union box centre maps to the origin for every view.
        union_center = np.array([1.5, 1.5, 1.5])
        for v in (v0, v1):
            mapped = (
                v.global_transform[:3, :3] @ union_center + v.global_transform[:3, 3]
            )
            assert np.allclose(mapped, 0.0)
