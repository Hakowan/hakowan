import pytest
import hakowan.beta as hkw
import hakowan.beta.compiler
import lagrange
import numpy as np
import numpy.typing as npt
import pathlib

from .asset import triangle, two_triangles


class TestCompile:
    def test_compile(self):
        data = lagrange.SurfaceMesh()
        l1 = hkw.layer.Layer().data(data)
        root = l1.mark(hkw.mark.Surface)
        scene = hkw.compiler.compile(root)
        assert len(scene.views) == 1

    def test_apply_transform(self, two_triangles):
        mesh = two_triangles

        base = hkw.layer.Layer().data(mesh).mark(hkw.mark.Surface)
        l1 = base.transform(
            hkw.transform.Filter(
                data=hkw.scale.Attribute(name="facet_index"),
                condition=lambda v: v % 2 == 0,
            )
        )
        scene = hkw.compiler.compile(l1)
        assert len(scene) == 1
        assert scene[0].data_frame.mesh.num_facets == 1

    def test_position_with_scale(self, triangle):
        mesh = triangle
        bbox_min = np.amin(mesh.vertices, axis=0)
        bbox_max = np.amax(mesh.vertices, axis=0)

        base = hkw.layer.Layer(data=mesh, mark=hkw.mark.Surface)
        l1 = base.channel(
            position=hkw.channel.Position(
                data=hkw.Attribute(
                    name=mesh.attr_name_vertex_to_position,
                    scale=hkw.scale.Uniform(factor=2.0),
                )
            )
        )

        scene = hkw.compiler.compile(l1)
        assert len(scene) == 1
        out_mesh = scene[0].data_frame.mesh

        out_bbox_min = np.amin(out_mesh.vertices, axis=0)
        out_bbox_max = np.amax(out_mesh.vertices, axis=0)

        assert bbox_min * 2 == pytest.approx(out_bbox_min)
        assert bbox_max * 2 == pytest.approx(out_bbox_max)

    def test_uv_with_scale(self, triangle):
        mesh = triangle
        attr_ids = mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV)
        assert len(attr_ids) > 0

        uv_attr_id = attr_ids[0]
        uv_attr_name = mesh.get_attribute_name(uv_attr_id)
        uv_attr = mesh.indexed_attribute(uv_attr_id)
        uv_values = uv_attr.values.data
        bbox_min = np.amin(uv_values, axis=0)
        bbox_max = np.amax(uv_values, axis=0)

        base = hkw.layer.Layer(data=mesh, mark=hkw.mark.Surface)
        light_material = hkw.texture.Uniform(color=0.2)
        dark_material = hkw.texture.Uniform(color=0.8)
        checkerboard = hkw.texture.CheckerBoard(
            uv=hkw.Attribute(name=uv_attr_name, scale=hkw.scale.Uniform(factor=2.0)),
            texture1 = light_material,
            texture2 = dark_material)
        l1 = base.channel(material=hkw.channel.Diffuse(reflectance=checkerboard))

        scene = hkw.compiler.compile(l1)
        assert len(scene) == 1
        out_mesh = scene[0].data_frame.mesh

        for attr_id in out_mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV):
            attr_name = out_mesh.get_attribute_name(attr_id)
            if attr_name.startswith("_hakowan"):

                assert out_mesh.has_attribute(attr_name)
                assert not out_mesh.is_attribute_indexed(attr_name)
                out_uv_values = out_mesh.attribute(attr_name).data
                out_bbox_min = np.amin(out_uv_values, axis=0)
                out_bbox_max = np.amax(out_uv_values, axis=0)

                assert bbox_min * 2 == pytest.approx(out_bbox_min)
                assert bbox_max * 2 == pytest.approx(out_bbox_max)


class TestScale:
    def __apply_scale(
        self,
        df: hkw.dataframe.DataFrame,
        name: str,
        sc: hkw.scale.Scale,
        expected_result: npt.NDArray,
    ):
        mesh = df.mesh
        assert mesh.has_attribute(name)
        hkw.compiler.attribute.apply_scale(df, name, sc)

        assert mesh.has_attribute(name)
        data = mesh.attribute(name).data
        assert np.all(data == pytest.approx(expected_result))

    def test_uniform(self, triangle):
        mesh = triangle
        df = hkw.dataframe.DataFrame(mesh=mesh)
        sc = hkw.scale.Uniform(factor=2.0)
        name = "vertex_index"
        self.__apply_scale(df, name, sc, np.arange(3) * 2)

    def test_log(self, triangle):
        mesh = triangle
        df = hkw.dataframe.DataFrame(mesh=mesh)
        sc = hkw.scale.Log(base=np.e)
        name = "vertex_data"
        self.__apply_scale(df, name, sc, np.log([1, 2, 3]))

    def test_normalize(self, triangle):
        mesh = triangle
        df = hkw.dataframe.DataFrame(mesh=mesh)
        sc = hkw.scale.Normalize(bbox_min=0, bbox_max=1)
        self.__apply_scale(df, "vertex_data", sc, np.array([0, 0.5, 1]))

    def test_custom(self, triangle):
        mesh = triangle
        df = hkw.dataframe.DataFrame(mesh=mesh)
        sc = hkw.scale.Custom(function=lambda x: x**2)
        self.__apply_scale(df, "vertex_data", sc, np.array([1, 4, 9]))

    def test_affine_scaling(self, triangle):
        mesh = triangle
        df = hkw.dataframe.DataFrame(mesh=mesh)
        sc = hkw.scale.Affine(matrix=np.eye(3) * 2)
        self.__apply_scale(df, mesh.attr_name_vertex_to_position, sc, mesh.vertices * 2)

    def test_affine_translating(self, triangle):
        mesh = triangle
        df = hkw.dataframe.DataFrame(mesh=mesh)
        M = np.eye(4)
        translation = np.array([1, 1, 1])
        M[0:3, 3] = translation
        sc = hkw.scale.Affine(matrix=M)
        self.__apply_scale(
            df, mesh.attr_name_vertex_to_position, sc, mesh.vertices + translation
        )

    def test_offset(self, triangle):
        mesh = triangle
        df = hkw.dataframe.DataFrame(mesh=mesh)
        mesh.create_attribute(
            "offset",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Vector,
            initial_values=np.ones((triangle.num_vertices, 3)),
        )
        offset_attr = hkw.scale.Attribute(name="offset")
        sc = hkw.scale.Offset(offset=offset_attr)
        self.__apply_scale(df, mesh.attr_name_vertex_to_position, sc, mesh.vertices + 1)

    def test_scaled_offset(self, triangle):
        mesh = triangle
        df = hkw.dataframe.DataFrame(mesh=mesh)
        mesh.create_attribute(
            "offset",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Vector,
            initial_values=np.ones((triangle.num_vertices, 3)),
        )
        uniform_scale = hkw.scale.Uniform(factor=2.0)
        offset_attr = hkw.scale.Attribute(name="offset", scale=uniform_scale)
        sc = hkw.scale.Offset(offset=offset_attr)
        self.__apply_scale(df, mesh.attr_name_vertex_to_position, sc, mesh.vertices + 2)


class TestTexture:
    def test_scalar_field(self, triangle):
        mesh = triangle
        df = hkw.dataframe.DataFrame(mesh=mesh)

        attr = hkw.scale.Attribute(name="vertex_index")
        tex = hkw.texture.ScalarField(data=attr)
        hkw.compiler.texture.apply_texture(df, tex)
        assert attr._internal_name is not None
        assert mesh.has_attribute(attr._internal_name)

        data = mesh.attribute(attr._internal_name).data
        assert np.amax(data) == pytest.approx(1.0)
        assert np.amin(data) == pytest.approx(0.0)

    def test_image(self, triangle):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            mesh = triangle
            df = hkw.dataframe.DataFrame(mesh=mesh)

            attr = hkw.scale.Attribute(name="uv")
            tex = hkw.texture.Image(uv=attr, filename=pathlib.Path(f.name))

            hkw.compiler.texture.apply_texture(df, tex)
            assert attr._internal_name is not None
            assert mesh.has_attribute(attr._internal_name)

    def test_image_with_scale(self, triangle):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            mesh = triangle
            df = hkw.dataframe.DataFrame(mesh=mesh)

            s = hkw.scale.Uniform(factor=2.0)
            attr = hkw.scale.Attribute(name="uv", scale=s)
            tex = hkw.texture.Image(uv=attr, filename=pathlib.Path(f.name))

            hkw.compiler.texture.apply_texture(df, tex)
            assert attr._internal_name is not None
            assert mesh.has_attribute(attr._internal_name)
            assert attr._internal_name != attr.name

    def test_checker_board(self, triangle):
        mesh = triangle
        df = hkw.dataframe.DataFrame(mesh=mesh)

        attr = hkw.scale.Attribute(name="uv")
        attr2 = hkw.scale.Attribute(name="vertex_index")
        tex = hkw.texture.CheckerBoard(
            uv=attr,
            texture1=hkw.texture.ScalarField(data=attr),
            texture2=hkw.texture.ScalarField(data=attr2),
        )

        hkw.compiler.texture.apply_texture(df, tex)
        assert attr._internal_name is not None
        assert mesh.has_attribute(attr._internal_name)
        assert attr2._internal_name is not None
        assert mesh.has_attribute(attr2._internal_name)

    def test_isocontour(self, triangle):
        mesh = triangle
        df = hkw.dataframe.DataFrame(mesh=mesh)

        attr = hkw.scale.Attribute(name="vertex_index")
        tex = hkw.texture.Isocontour(
            data=attr,
            ratio=0.1,
            texture1=hkw.texture.Uniform(color=0.2),
            texture2=hkw.texture.ScalarField(data=attr),
        )
        hkw.compiler.texture.apply_texture(df, tex)

        assert attr._internal_name is not None
        assert mesh.has_attribute(attr._internal_name)
