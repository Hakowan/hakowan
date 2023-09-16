import pytest
from hakowan.beta import layer, mark, dataframe, compiler, scale, transform
import lagrange
import numpy as np
import numpy.typing as npt

from .asset import triangle, two_triangles


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
                data=scale.Attribute(name="index"),
                condition=lambda v: v % 2 == 0,
            )
        )
        scene = compiler.compile(l1)
        assert len(scene) == 1
        assert scene[0].data_frame.mesh.num_facets == 1


class TestScale:
    import hakowan.beta.compiler as compiler

    def __apply_scale(
        self,
        df: dataframe.DataFrame,
        name: str,
        sc: scale.Scale,
        expected_result: npt.NDArray,
    ):
        mesh = df.mesh
        assert mesh.has_attribute(name)
        compiler.attribute.apply_scale(df, name, sc)

        assert mesh.has_attribute(name)
        vertex_index = mesh.attribute(name).data
        assert np.all(vertex_index == pytest.approx(expected_result))

    def test_uniform(self, triangle):
        mesh = triangle
        df = dataframe.DataFrame(mesh=mesh)
        sc = scale.Uniform(factor=2.0)
        name = "vertex_index"
        self.__apply_scale(df, name, sc, np.arange(3) * 2)

    def test_log(self, triangle):
        mesh = triangle
        df = dataframe.DataFrame(mesh=mesh)
        sc = scale.Log(base=np.e)
        name = "vertex_data"
        self.__apply_scale(df, name, sc, np.log([1, 2, 3]))

    def test_normalize(self, triangle):
        mesh = triangle
        df = dataframe.DataFrame(mesh=mesh)
        sc = scale.Normalize(bbox_min=0, bbox_max=1)
        compiler.attribute.update_scale(df, "vertex_data", sc)
        assert sc._value_min == pytest.approx(1)
        assert sc._value_max == pytest.approx(3)

        self.__apply_scale(df, "vertex_data", sc, np.array([0, 0.5, 1]))

    def test_custom(self, triangle):
        mesh = triangle
        df = dataframe.DataFrame(mesh=mesh)
        sc = scale.Custom(function=lambda x: x**2)
        self.__apply_scale(df, "vertex_data", sc, np.array([1, 4, 9]))
