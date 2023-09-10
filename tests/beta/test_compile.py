import pytest
from hakowan.beta import layer, mark, dataframe, compiler, scale, transform
import lagrange

import numpy as np


class TestCompile:
    def test_compile(self):
        data = lagrange.SurfaceMesh()
        l1 = layer.Layer().data(data)
        root = l1.mark(mark.Surface)
        scene = compiler.compile(root)
        assert len(scene.views) == 1

    def test_apply_transform(self):
        mesh = lagrange.SurfaceMesh()
        mesh.add_vertices(np.eye(3))
        mesh.add_triangle(0, 1, 2)
        mesh.add_triangle(2, 1, 0)
        mesh.create_attribute(
            "index",
            element=lagrange.AttributeElement.Facet,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=np.arange(mesh.num_facets, dtype=np.uint32),
        )

        l1 = layer.Layer().data(mesh).mark(mark.Surface)
        l1.transform(
            transform.Filter(
                data=scale.Attribute(name="index"), condition=lambda v: v % 2 == 0
            )
        )
        scene = compiler.compile(l1)
        assert len(scene) == 1
        assert scene[0].data.mesh.num_facets == 2
        scene[0].apply_transform()
        assert scene[0].data.mesh.num_facets == 1
