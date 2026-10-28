from __future__ import annotations

import lagrange
import numpy as np
import pytest

import hakowan as hkw


def _attribute(mesh, name):
    return np.asarray(mesh.attribute(name).data)


def test_numpy_positions_create_point_mesh_and_render(tmp_path):
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    layer = hkw.layer(points)

    scene = hkw.compile(layer)
    mesh = scene[0].data_frame.mesh
    assert scene[0].mark is hkw.mark.Point
    result = hkw.render(layer, filename=tmp_path / "points.html", backend="webgl")

    assert mesh.num_vertices == 3
    assert mesh.num_facets == 0
    assert np.asarray(mesh.vertices)[:, 2] == pytest.approx(0.0)
    assert result.outputs["main"] == tmp_path / "points.html"


def test_pandas_dataframe_infers_positions_and_imports_numeric_columns():
    pandas = pytest.importorskip("pandas")
    table = pandas.DataFrame(
        {
            "x": [0.0, 1.0, 0.0],
            "y": [0.0, 0.0, 1.0],
            "z": [0.0, 0.0, 0.0],
            "temperature": [10.0, 20.0, 30.0],
            "label": ["a", "b", "c"],
        }
    )

    frame = hkw.dataframe.to_dataframe(table)

    assert frame.mesh.num_vertices == 3
    assert frame.mesh.has_attribute("temperature")
    assert not frame.mesh.has_attribute("label")
    assert _attribute(frame.mesh, "temperature").reshape(-1) == pytest.approx(
        [10.0, 20.0, 30.0]
    )


def test_pandas_dataframe_preserves_boolean_columns_as_integer_attributes():
    pandas = pytest.importorskip("pandas")
    table = pandas.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "z": [0.0, 0.0],
            "selected": [True, False],
        }
    )

    frame = hkw.dataframe.to_dataframe(table)

    np.testing.assert_array_equal(
        _attribute(frame.mesh, "selected").reshape(-1), [1, 0]
    )


def test_pandas_dataframe_accepts_explicit_position_columns():
    pandas = pytest.importorskip("pandas")
    table = pandas.DataFrame(
        {
            "east": [0.0, 1.0],
            "north": [2.0, 3.0],
            "height": [4.0, 5.0],
            "value": [6.0, 7.0],
        }
    )

    layer = hkw.layer(table, positions=("east", "north", "height"))
    mesh = hkw.compile(layer.mark("point").channel(size="value"))[0].data_frame.mesh

    np.testing.assert_allclose(
        np.asarray(mesh.vertices), [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]
    )
    assert mesh.has_attribute("value")


def test_inspect_accepts_tabular_input():
    pandas = pytest.importorskip("pandas")
    table = pandas.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "value": [2.0, 4.0]})

    summary = hkw.inspect(table)

    assert summary.vertex_count == 2
    assert summary.facet_count == 0
    assert summary.attribute("value").minimum == 2.0


def test_xarray_dataset_adapter_uses_point_dimension():
    class Variable:
        def __init__(self, values):
            self.values = np.asarray(values)

    class Dataset:
        def __init__(self):
            self.data_vars = {"velocity": Variable([[1, 0, 0], [0, 1, 0]])}
            self.coords = {
                "x": Variable([0.0, 1.0]),
                "y": Variable([2.0, 3.0]),
                "z": Variable([4.0, 5.0]),
            }

        def __getitem__(self, name):
            return self.coords.get(name, self.data_vars.get(name))

    Dataset.__module__ = "xarray.core.dataset"
    frame = hkw.dataframe.to_dataframe(Dataset())

    np.testing.assert_allclose(
        np.asarray(frame.mesh.vertices), [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]
    )
    np.testing.assert_allclose(
        _attribute(frame.mesh, "velocity"), [[1, 0, 0], [0, 1, 0]]
    )


def test_real_xarray_dataset_is_not_dispatched_as_a_mesh_mapping():
    xarray = pytest.importorskip("xarray")
    dataset = xarray.Dataset(
        {"temperature": ("point", [10.0, 20.0, 30.0])},
        coords={
            "x": ("point", [0.0, 1.0, 0.0]),
            "y": ("point", [0.0, 0.0, 1.0]),
        },
    )

    frame = hkw.dataframe.to_dataframe(dataset)

    np.testing.assert_allclose(frame.mesh.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(
        _attribute(frame.mesh, "temperature").reshape(-1), [10, 20, 30]
    )


def test_pyvista_polydata_preserves_points_faces_and_attributes():
    pyvista = pytest.importorskip("pyvista")
    source = pyvista.PolyData(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([3, 0, 1, 2]),
    )
    source.point_data["value"] = np.array([1.0, 2.0, 3.0])

    frame = hkw.dataframe.to_dataframe(source)

    assert frame.mesh.num_vertices == 3
    assert frame.mesh.num_facets == 1
    assert _attribute(frame.mesh, "value").reshape(-1) == pytest.approx([1, 2, 3])


def test_pyvista_vertex_cells_preserve_point_attributes():
    pyvista = pytest.importorskip("pyvista")
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    source = pyvista.UnstructuredGrid(
        np.array([1, 0, 1, 1]),
        np.array([pyvista.CellType.VERTEX, pyvista.CellType.VERTEX]),
        points,
    )
    source.point_data["velocity"] = np.array([[1.0, 0, 0], [0, 2.0, 0]])

    frame = hkw.dataframe.to_dataframe(source)

    assert frame.mesh.num_vertices == 2
    assert frame.mesh.num_facets == 0
    np.testing.assert_allclose(
        _attribute(frame.mesh, "velocity"), [[1.0, 0, 0], [0, 2.0, 0]]
    )
    colored = (
        hkw.layer(frame)
        .mark("point")
        .color_by("speed", domain=(0, 3))
        .transform(hkw.transform.Norm("velocity", "speed"))
    )
    compiled = hkw.compile(colored, preserve_attributes=True)[0].data_frame.mesh
    colors = _attribute(compiled, "vertex_color")
    assert len(np.unique(colors, axis=0)) == 2


def test_trimesh_preserves_topology_and_attribute_domains():
    trimesh = pytest.importorskip("trimesh")
    source = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    source.vertex_attributes["value"] = np.array([1.0, 2.0, 3.0])
    source.face_attributes["quality"] = np.array([0.5])

    frame = hkw.dataframe.to_dataframe(source)

    assert frame.mesh.num_facets == 1
    assert frame.mesh.has_attribute("value")
    assert frame.mesh.has_attribute("quality")


def test_2d_mesh_mapping_preserves_topology_attributes_and_renders(tmp_path):
    mesh_data = {
        "vertices": [[0, 0], [2, 0], [2, 1], [0, 1]],
        "facets": [[0, 1, 2], [0, 2, 3]],
        "vertex_attributes": {"temperature": [10.0, 20.0, 30.0, 40.0]},
        "facet_attributes": {"region": [1, 2]},
    }

    frame = hkw.dataframe.to_dataframe(mesh_data)
    np.testing.assert_allclose(
        frame.mesh.vertices, [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]]
    )
    np.testing.assert_array_equal(frame.mesh.facets, [[0, 1, 2], [0, 2, 3]])
    np.testing.assert_allclose(
        _attribute(frame.mesh, "temperature").reshape(-1), [10, 20, 30, 40]
    )
    np.testing.assert_array_equal(_attribute(frame.mesh, "region").reshape(-1), [1, 2])

    result = hkw.render(
        hkw.layer(mesh_data).mark("surface"),
        filename=tmp_path / "planar-mesh.html",
        backend="webgl",
    )
    assert result.outputs["main"] == tmp_path / "planar-mesh.html"
    assert result.outputs["main"].is_file()


def test_2d_lagrange_surface_mesh_is_embedded_and_renders(tmp_path):
    source = lagrange.SurfaceMesh(dimension=2)
    source.add_vertices(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))
    source.add_triangles(np.array([[0, 1, 2]], dtype=np.uint32))
    source.create_attribute(
        "temperature",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.array([10.0, 20.0, 30.0]),
    )

    frame = hkw.dataframe.to_dataframe(source)

    assert source.dimension == 2
    assert frame.mesh.dimension == 3
    np.testing.assert_allclose(frame.mesh.vertices, [[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    np.testing.assert_array_equal(frame.mesh.facets, [[0, 1, 2]])
    np.testing.assert_allclose(
        _attribute(frame.mesh, "temperature").reshape(-1), [10, 20, 30]
    )

    result = hkw.render(
        hkw.layer(source).mark("surface"),
        filename=tmp_path / "lagrange-2d-mesh.html",
        backend="webgl",
    )
    assert result.outputs["main"] == tmp_path / "lagrange-2d-mesh.html"
    assert result.outputs["main"].is_file()


def test_2d_lagrange_quad_mesh_preserves_polygon_facets():
    source = lagrange.SurfaceMesh(dimension=2)
    source.add_vertices(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    source.add_quads(np.array([[0, 1, 2, 3]], dtype=np.uint32))

    embedded = hkw.dataframe.to_dataframe(source).mesh

    assert embedded.dimension == 3
    np.testing.assert_array_equal(embedded.facets, [[0, 1, 2, 3]])


def test_2d_facet_free_lagrange_mesh_embeds_as_point_mesh():
    source = lagrange.SurfaceMesh(dimension=2)
    source.add_vertices(np.array([[0.0, 0.0], [1.0, 0.0]]))
    source.create_attribute(
        "temperature",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.array([10.0, 20.0]),
    )

    embedded = hkw.dataframe.to_dataframe(source).mesh

    assert embedded.dimension == 3
    assert embedded.num_facets == 0
    np.testing.assert_allclose(embedded.vertices, [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(
        _attribute(embedded, "temperature").reshape(-1), [10, 20]
    )


def test_2d_lagrange_edge_attributes_are_preserved():
    source = lagrange.SurfaceMesh(dimension=2)
    source.add_vertices(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    source.add_triangles(np.array([[0, 1, 2]], dtype=np.uint32))
    source.initialize_edges()
    edge_values = np.arange(source.num_edges, dtype=np.float64)
    source.create_attribute(
        "edge_value",
        element=lagrange.AttributeElement.Edge,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=edge_values,
    )

    embedded = hkw.dataframe.to_dataframe(source).mesh

    np.testing.assert_array_equal(embedded.edges, source.edges)
    np.testing.assert_array_equal(_attribute(embedded, "edge_value"), edge_values)


def test_2d_vector_position_attributes_are_embedded_for_compilation():
    source = lagrange.SurfaceMesh(dimension=2)
    source.add_vertices(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    source.add_triangles(np.array([[0, 1, 2]], dtype=np.uint32))
    alternate = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    source.create_attribute(
        "alternate",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Position,
        initial_values=alternate,
    )

    scene = hkw.compile(
        hkw.layer(source)
        .mark("surface")
        .channel(position=hkw.channel.Position(data="alternate"))
    )

    np.testing.assert_allclose(
        scene[0].data_frame.mesh.vertices,
        [[0, 0, 0], [2, 0, 0], [0, 3, 0]],
    )


def test_2d_normal_attribute_is_embedded_and_renders(tmp_path):
    source = lagrange.SurfaceMesh(dimension=2)
    source.add_vertices(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    source.add_triangles(np.array([[0, 1, 2]], dtype=np.uint32))
    source.create_attribute(
        "normal2",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Normal,
        initial_values=np.tile([0.0, 1.0], (3, 1)),
    )

    frame = hkw.dataframe.to_dataframe(source)
    np.testing.assert_allclose(
        _attribute(frame.mesh, "normal2"), np.tile([0.0, 1.0, 0.0], (3, 1))
    )
    result = hkw.render(
        hkw.layer(source).mark("surface"),
        filename=tmp_path / "2d-normal.html",
        backend="webgl",
    )
    assert result.outputs["main"].is_file()


def test_2d_mesh_mapping_embeds_two_component_vertex_attributes():
    mesh_data = {
        "vertices": [[0, 0], [1, 0], [0, 1]],
        "facets": [[0, 1, 2]],
        "vertex_attributes": {"alternate": [[0, 0], [2, 0], [0, 3]]},
    }

    scene = hkw.compile(
        hkw.layer(mesh_data)
        .mark("surface")
        .channel(position=hkw.channel.Position(data="alternate"))
    )

    np.testing.assert_allclose(
        scene[0].data_frame.mesh.vertices,
        [[0, 0, 0], [2, 0, 0], [0, 3, 0]],
    )


def test_2d_mesh_embedding_copies_attribute_storage():
    source = lagrange.SurfaceMesh(dimension=2)
    source.add_vertices(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    source.add_triangles(np.array([[0, 1, 2]], dtype=np.uint32))
    source.create_attribute(
        "temperature",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.array([10.0, 20.0, 30.0]),
    )
    source.create_attribute(
        "uv",
        element=lagrange.AttributeElement.Indexed,
        usage=lagrange.AttributeUsage.UV,
        initial_values=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        initial_indices=np.array([0, 1, 2], dtype=np.uint32),
    )

    embedded = hkw.dataframe.to_dataframe(source).mesh
    source.vertices[0, 0] = 9.0
    source.attribute("temperature").data[0] = 99.0
    source.indexed_attribute("uv").values.data[0, 0] = 99.0

    np.testing.assert_allclose(embedded.vertices[0], [0, 0, 0])
    np.testing.assert_allclose(
        _attribute(embedded, "temperature").reshape(-1), [10, 20, 30]
    )
    np.testing.assert_allclose(
        embedded.indexed_attribute("uv").values.data,
        [[0, 0], [1, 0], [0, 1]],
    )


@pytest.mark.parametrize(
    "mesh_data",
    [
        {
            "vertices": [[0, 0], [1, 0], [0, 1]],
            "facets": [[0, 1, 2]],
            "vertex_attributes": {"temperature": [1, 2]},
        },
        {
            "vertices": [[0, 0], [1, 0], [0, 1]],
            "facets": [[0, 1, 2]],
            "facet_attributes": {"region": []},
        },
        {"vertices": [[0, 0], [1, 0], [0, 1]], "facets": [[]]},
    ],
)
def test_mesh_mapping_rejects_mismatched_attributes_and_empty_facets(mesh_data):
    with pytest.raises(ValueError):
        hkw.dataframe.to_dataframe(mesh_data)


def test_adapter_rejects_ambiguous_or_invalid_positions():
    pandas = pytest.importorskip("pandas")
    with pytest.raises(ValueError, match="infer position"):
        hkw.layer(pandas.DataFrame({"a": [1.0], "b": [2.0]}))
    with pytest.raises(ValueError, match="shape"):
        hkw.layer(np.ones((3, 4)))
    with pytest.raises(ValueError, match="finite"):
        hkw.layer(np.array([[0.0, np.nan, 0.0]]))


def test_adapter_rejects_empty_positions():
    with pytest.raises(ValueError, match="at least one point"):
        hkw.layer(np.zeros((0, 3)))
