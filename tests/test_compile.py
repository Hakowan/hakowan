import pytest
import hakowan as hkw
import hakowan.compiler
import lagrange
import numpy as np
import numpy.typing as npt
import pathlib

from .asset import triangle, two_triangles


class TestCompile:
    def test_compile(self):
        data = lagrange.SurfaceMesh()
        l1 = hkw.layer().data(data)
        root = l1.mark(hkw.mark.Surface)
        scene = hkw.compiler.compile(root)
        assert len(scene.views) == 1

    def test_apply_transform(self, two_triangles):
        mesh = two_triangles

        base = hkw.layer().data(mesh).mark(hkw.mark.Surface)
        l1 = base.transform(
            hkw.transform.Filter(
                data=hkw.attribute(name="facet_index"),
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

        base = hkw.layer(data=mesh, mark=hkw.mark.Surface)
        l1 = base.channel(
            position=hkw.channel.Position(
                data=hkw.attribute(
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

        base = hkw.layer(data=mesh, mark=hkw.mark.Surface)
        light_material = hkw.texture.Uniform(color=0.2)
        dark_material = hkw.texture.Uniform(color=0.8)
        checkerboard = hkw.texture.Checkerboard(
            uv=hkw.attribute(name=uv_attr_name, scale=hkw.scale.Uniform(factor=2.0)),
            texture1=light_material,
            texture2=dark_material,
        )
        l1 = base.channel(material=hkw.material.Diffuse(reflectance=checkerboard))

        scene = hkw.compiler.compile(l1)
        assert len(scene) == 1
        out_mesh = scene[0].data_frame.mesh

        for attr_id in out_mesh.get_matching_attribute_ids(
            usage=lagrange.AttributeUsage.UV
        ):
            attr_name = out_mesh.get_attribute_name(attr_id)
            if attr_name.startswith("_hakowan"):
                assert out_mesh.has_attribute(attr_name)
                assert not out_mesh.is_attribute_indexed(attr_name)
                out_uv_values = out_mesh.attribute(attr_name).data
                out_bbox_min = np.amin(out_uv_values, axis=0)
                out_bbox_max = np.amax(out_uv_values, axis=0)

                assert bbox_min * 2 == pytest.approx(out_bbox_min)
                assert bbox_max * 2 == pytest.approx(out_bbox_max)

    def test_scalar_field(self, triangle):
        mesh = triangle

        base = hkw.layer(data=mesh, mark=hkw.mark.Surface)
        mat = hkw.material.Diffuse(
            reflectance=hkw.texture.ScalarField(
                data=hkw.attribute(name="vertex_data"), colormap="viridis"
            )
        )
        l1 = base.channel(material=mat)

        scene = hkw.compiler.compile(l1)
        assert len(scene) == 1
        out_mesh = scene[0].data_frame.mesh

        color_attr_ids = out_mesh.get_matching_attribute_ids(
            usage=lagrange.AttributeUsage.Color
        )
        assert len(color_attr_ids) == 1
        color_attr_id = color_attr_ids[0]
        color_attr_name = out_mesh.get_attribute_name(color_attr_id)
        assert color_attr_name.startswith("vertex_")
        color_attr = out_mesh.attribute(color_attr_id)
        assert color_attr.usage == lagrange.AttributeUsage.Color
        assert color_attr.element_type == lagrange.AttributeElement.Vertex
        assert color_attr.num_channels == 3

        colors = color_attr.data
        assert colors[0] == pytest.approx([0.267004, 0.004874, 0.329415])
        assert colors[2] == pytest.approx([0.993248, 0.906157, 0.143936])
        assert np.amax(np.absolute(colors[1] - colors[0])) > 0.1
        assert np.amax(np.absolute(colors[1] - colors[2])) > 0.1

    def test_size_field(self, triangle):
        mesh = triangle

        base = hkw.layer(data=mesh, mark=hkw.mark.Point)
        l1 = base.channel(size=0.1)

        scene = hkw.compiler.compile(l1)
        assert len(scene) == 1
        view = scene[0]
        assert view.size_channel is not None
        assert view.size_channel.data == 0.1

    def test_multiple_views(self, triangle, two_triangles):
        mesh1 = triangle
        mesh2 = two_triangles

        l1 = hkw.layer(data=mesh1)
        l2 = hkw.layer(data=mesh2)
        combined = (l1 + l2).mark(hkw.mark.Surface)

        scene = hkw.compiler.compile(combined)
        assert len(scene) == 2

        assert scene[0].data_frame.mesh.num_facets == 1
        assert scene[1].data_frame.mesh.num_facets == 2

    def test_vector_field(self, triangle):
        mesh = triangle
        attr_id = lagrange.compute_vertex_normal(mesh)
        attr_name = mesh.get_attribute_name(attr_id)

        base = hkw.layer(data=mesh, mark=hkw.mark.Curve)
        base = base.channel(vector_field=attr_name)

        scene = hkw.compiler.compile(base)
        assert len(scene) == 1

        assert scene[0].vector_field_channel is not None

    def test_filter_transform(self, two_triangles):
        mesh = two_triangles
        bbox_min = np.amin(mesh.vertices, axis=0)
        bbox_max = np.amax(mesh.vertices, axis=0)
        base = (
            hkw.layer()
            .data(mesh)
            .mark(hkw.mark.Surface)
            .transform(
                hkw.transform.Filter(
                    data=hkw.attribute(name="facet_index"),
                    condition=lambda x: x % 2 == 0,
                )
            )
        )
        scene = hkw.compiler.compile(base)

        assert len(scene) == 1
        assert scene[0].data_frame.mesh.num_facets == 1
        bbox = scene[0].bbox
        assert np.all(bbox[0] == pytest.approx(bbox_min))
        assert np.all(bbox[1] == pytest.approx(bbox_max))

    def test_uv_mesh_transform(self, triangle):
        mesh = triangle
        mesh.vertices[:, 2] = 1
        assert mesh.has_attribute("uv")
        assert mesh.has_attribute("vertex_index")
        assert mesh.has_attribute("facet_index")
        assert mesh.has_attribute("corner_index")

        base = hkw.layer(mesh).transform(hkw.transform.UVMesh(uv="uv"))
        base = base.channel(
            material=hkw.material.Principled(
                color=hkw.texture.ScalarField(data="vertex_index"),
                roughness=hkw.texture.ScalarField(data="facet_index"),
                metallic=hkw.texture.ScalarField(data="corner_index"),
            )
        )
        scene = hkw.compiler.compile(base)
        out_mesh = scene[0].data_frame.mesh

        assert len(scene) == 1
        assert out_mesh.num_facets == 1
        assert np.allclose(out_mesh.vertices[:, 2], 0)
        assert np.allclose(out_mesh.vertices[:, :2], mesh.vertices[:, :2])

    def test_affine_transform(self, triangle):
        mesh = triangle
        normal_attr_id = lagrange.compute_facet_normal(mesh)
        normal_attr_name = mesh.get_attribute_name(normal_attr_id)
        matrix = np.eye(4)
        matrix[0, 0] = 2.0  # Stretch in X direction
        base = (
            hkw.layer(mesh)
            .transform(hkw.transform.Affine(matrix))
            .channel(normal=normal_attr_name)
        )
        scene = hkw.compiler.compile(base)

        assert len(scene) == 1
        view = scene[0]
        out_mesh = view.data_frame.mesh
        assert np.all(np.amax(out_mesh.vertices, axis=0) == pytest.approx([2, 1, 1]))

        n = out_mesh.attribute(normal_attr_name).data
        d = np.dot(out_mesh.vertices, n.T)
        assert np.allclose(d, d[[2, 0, 1]])

    def test_compute_transform_component(self, triangle):
        mesh = lagrange.combine_meshes([triangle, triangle])
        base = (
            hkw.layer(mesh)
            .transform(hkw.transform.Compute(component="component"))
            .channel(
                material=hkw.material.Diffuse(
                    hkw.texture.ScalarField("component", colormap=[0.0, 1.0])
                )
            )
        )
        scene = hkw.compiler.compile(base)

        assert len(scene) == 1
        view = scene[0]
        out_mesh = view.data_frame.mesh
        color_attr_ids = out_mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.Color)
        assert len(color_attr_ids) == 1


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
        sc = hkw.scale.Normalize(range_min=0, range_max=1)
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
        offset_attr = hkw.attribute(name="offset")
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
        offset_attr = hkw.attribute(name="offset", scale=uniform_scale)
        sc = hkw.scale.Offset(offset=offset_attr)
        self.__apply_scale(df, mesh.attr_name_vertex_to_position, sc, mesh.vertices + 2)


class TestTexture:
    def test_scalar_field(self, triangle):
        mesh = triangle
        df = hkw.dataframe.DataFrame(mesh=mesh)

        attr = hkw.attribute(name="vertex_data")
        tex = hkw.texture.ScalarField(data=attr)
        hkw.compiler.texture.apply_texture(df, tex)
        assert attr._internal_name is not None
        assert mesh.has_attribute(attr._internal_name)

        data = mesh.attribute(attr._internal_name).data
        assert data[0] == pytest.approx(0.0)
        assert data[1] == pytest.approx(0.5)
        assert data[2] == pytest.approx(1.0)

    def test_scalar_field_with_custom_colormap(self, triangle):
        mesh = triangle
        df = hkw.dataframe.DataFrame(mesh=mesh)
        attr = hkw.attribute(name="vertex_data")
        tex = hkw.texture.ScalarField(data=attr, colormap=["black", "white"])
        hkw.compiler.texture.apply_texture(df, tex)
        hkw.compiler.color.apply_colormap(df, tex)
        assert attr._internal_name is not None
        assert attr._internal_color_field is not None
        assert mesh.has_attribute(attr._internal_name)
        assert mesh.has_attribute(attr._internal_color_field)
        data = mesh.attribute(attr._internal_color_field).data
        assert np.allclose(data[:, 0], data[:, 1])
        assert np.allclose(data[:, 0], data[:, 2])
        assert not np.allclose(data[:, 0], 0)
        assert not np.allclose(data[:, 0], 1)

    def test_image(self, triangle):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            mesh = triangle
            df = hkw.dataframe.DataFrame(mesh=mesh)

            attr = hkw.attribute(name="uv")
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
            attr = hkw.attribute(name="uv", scale=s)
            tex = hkw.texture.Image(uv=attr, filename=pathlib.Path(f.name))

            hkw.compiler.texture.apply_texture(df, tex)
            assert attr._internal_name is not None
            assert mesh.has_attribute(attr._internal_name)
            assert attr._internal_name != attr.name

    def test_checker_board(self, triangle):
        mesh = triangle
        df = hkw.dataframe.DataFrame(mesh=mesh)

        attr = hkw.attribute(name="uv")
        attr2 = hkw.attribute(name="vertex_data")
        tex = hkw.texture.Checkerboard(
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

        attr = hkw.attribute(name="vertex_index")
        tex = hkw.texture.Isocontour(
            data=attr,
            ratio=0.1,
            texture1=hkw.texture.Uniform(color=0.2),
            texture2=hkw.texture.ScalarField(data=attr),
        )
        hkw.compiler.texture.apply_texture(df, tex)

        assert attr._internal_name is not None
        assert mesh.has_attribute(attr._internal_name)
