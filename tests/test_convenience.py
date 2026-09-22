from __future__ import annotations

import lagrange
import numpy as np
import pytest

import hakowan as hkw


def _triangle():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    mesh.add_triangle(0, 1, 2)
    mesh.create_attribute(
        "temperature",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.array([[1.0], [2.0], [3.0]]),
    )
    mesh.create_attribute(
        "velocity",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Vector,
        initial_values=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
    )
    return mesh


def _two_components():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
            ]
        )
    )
    mesh.add_triangles(np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32))
    return mesh


def test_color_by_builds_scalar_material_and_legend():
    mesh = _triangle()
    base = hkw.layer(mesh)

    colored = base.color_by(
        "temperature", colormap="viridis", domain=(1, 3), reverse=True
    )
    scene = hkw.compile(colored)
    material = scene[0].material_channel

    assert base._spec.channels == []
    assert isinstance(material, hkw.material.Diffuse)
    assert isinstance(material.reflectance, hkw.texture.ScalarField)
    assert material.reflectance.domain == (1, 3)
    assert material.reflectance.reverse
    assert len(scene.legends) == 1


def test_color_by_serializes_to_minimal_canonical_scalar_field():
    mesh = _triangle()

    spec = hkw.to_spec(
        hkw.layer(mesh).color_by("temperature"),
        data_ids={id(mesh): "mesh"},
    ).to_dict()

    material = spec["root"]["spec"]["channels"]["material"]
    assert material["kind"] == "diffuse"
    assert material["reflectance"]["kind"] == "scalar_field"
    assert material["reflectance"]["data"]["name"] == "temperature"
    assert material["reflectance"]["colormap"] == "viridis"
    assert material["reflectance"]["legend"] is True


def test_show_edges_overlays_surface_and_curve():
    mesh = _triangle()

    scene = hkw.compile(hkw.layer(mesh).show_edges(color="red", width=0.03))

    assert [view.mark for view in scene] == [hkw.mark.Surface, hkw.mark.Curve]
    assert scene[1].name == "Edges"
    assert scene[1].size_channel.data == pytest.approx(0.03)
    assert scene[1].material_channel.reflectance == "red"


def test_glyph_vectors_overlays_scaled_arrow_field():
    mesh = _triangle()

    scene = hkw.compile(
        hkw.layer(mesh).glyph_vectors("velocity", scale=0.5, size=0.02, normalize=True)
    )
    glyphs = scene[1]
    attribute = glyphs.vector_field_channel.data
    values = np.asarray(glyphs.data_frame.mesh.attribute(attribute._internal_name).data)

    assert glyphs.mark is hkw.mark.Curve
    assert glyphs.name == "Vectors"
    assert glyphs.vector_field_channel.end_type == "arrow"
    assert glyphs.vector_field_channel.normalize
    assert glyphs.size_channel.data == pytest.approx(0.02)
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 0.5)


def test_glyph_vectors_materializes_indexed_vector_attributes():
    mesh = _triangle()
    mesh.delete_attribute("velocity")
    mesh.create_attribute(
        "velocity",
        element=lagrange.AttributeElement.Indexed,
        usage=lagrange.AttributeUsage.Vector,
        initial_values=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
        initial_indices=np.array([0, 1, 2], dtype=np.uint32),
    )

    layer = hkw.layer(mesh).glyph_vectors(
        "velocity", scale=0.5, normalize=True, overlay=False
    )
    report = hkw.validate(layer, strict=True)
    view = hkw.compile(layer)[0]
    attribute = view.vector_field_channel.data

    assert report.valid
    assert attribute._internal_name is not None
    assert not view.data_frame.mesh.is_attribute_indexed(attribute._internal_name)
    values = np.asarray(view.data_frame.mesh.attribute(attribute._internal_name).data)
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 0.5)


def test_slice_clips_without_mutating_source():
    mesh = _triangle()

    sliced = hkw.layer(mesh).slice((1, 0, 0), offset=0)
    result = hkw.compile(sliced)[0].data_frame.mesh

    assert mesh.num_vertices == 3
    assert np.min(np.asarray(result.vertices)[:, 0]) >= -1e-12
    assert result.num_facets > 0


def test_isolate_component_computes_then_filters():
    mesh = _two_components()

    isolated = hkw.layer(mesh).isolate_component(0)
    result = hkw.compile(isolated)[0].data_frame.mesh

    assert mesh.num_facets == 2
    assert result.num_facets == 1


def test_isolate_component_uses_existing_labels():
    mesh = _two_components()
    mesh.create_attribute(
        "region",
        element=lagrange.AttributeElement.Facet,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.array([[4], [8]], dtype=np.int32),
    )

    result = hkw.compile(
        hkw.layer(mesh).isolate_component(8, attribute="region", compute=False)
    )[0].data_frame.mesh

    assert result.num_facets == 1
    assert np.min(np.asarray(result.vertices)[:, 0]) >= 3.0


def test_compare_labels_and_separates_layers():
    mesh = _triangle()

    scene = hkw.compile(
        hkw.layer(mesh).compare(hkw.layer(mesh), labels=("Before", "After"), gap=0.2)
    )

    assert [view.name for view in scene] == ["Before", "After"]
    assert scene[0].bbox[1, 0] < scene[1].bbox[0, 0]


def test_convenience_expansions_round_trip_canonically():
    mesh = _triangle()
    layer = hkw.layer(mesh).color_by("temperature").show_edges(width=0.02)

    spec = hkw.to_spec(layer, data_ids={id(mesh): "mesh"})
    restored = hkw.from_spec(spec, data_resolver={"mesh": mesh})
    rebuilt = hkw.to_spec(restored, data_ids={id(mesh): "mesh"})

    assert rebuilt.to_json(canonical=True) == spec.to_json(canonical=True)


def test_convenience_operations_reject_invalid_parameters():
    layer = hkw.layer(_triangle())

    with pytest.raises(ValueError, match="positive"):
        layer.show_edges(width=0)
    with pytest.raises(ValueError, match="positive"):
        layer.glyph_vectors("velocity", scale=0)
    with pytest.raises(ValueError, match="non-zero"):
        layer.slice((0, 0, 0))
    with pytest.raises(ValueError, match="either point or offset"):
        layer.slice((1, 0, 0), offset=1, point=(0, 0, 0))
