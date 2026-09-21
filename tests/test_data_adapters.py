from __future__ import annotations

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
    table = pandas.DataFrame(
        {"x": [0.0, 1.0], "y": [0.0, 1.0], "value": [2.0, 4.0]}
    )

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
            self.data_vars = {
                "velocity": Variable([[1, 0, 0], [0, 1, 0]])
            }
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


def test_adapter_rejects_ambiguous_or_invalid_positions():
    pandas = pytest.importorskip("pandas")
    with pytest.raises(ValueError, match="infer position"):
        hkw.layer(pandas.DataFrame({"a": [1.0], "b": [2.0]}))
    with pytest.raises(ValueError, match="shape"):
        hkw.layer(np.ones((3, 4)))
    with pytest.raises(ValueError, match="finite"):
        hkw.layer(np.array([[0.0, np.nan, 0.0]]))
