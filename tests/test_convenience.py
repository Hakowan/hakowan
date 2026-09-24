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


def test_color_by_uses_qualitative_default_for_categories():
    mesh = _triangle()

    spec = hkw.to_spec(
        hkw.layer(mesh).color_by("temperature", categories=True),
        data_ids={id(mesh): "mesh"},
    ).to_dict()

    field = spec["root"]["spec"]["channels"]["material"]["reflectance"]
    assert field["categories"] is True
    assert field["colormap"] == "set1"


def test_show_edges_overlays_surface_and_curve():
    mesh = _triangle()

    scene = hkw.compile(hkw.layer(mesh).show_edges(color="red", width=0.03))

    assert [view.mark for view in scene] == [hkw.mark.Surface, hkw.mark.Curve]
    assert scene[1].name == "Edges"
    assert scene[1].size_channel.data == pytest.approx(0.03)
    assert scene[1].material_channel.reflectance == "red"

    default_scene = hkw.compile(hkw.layer(mesh).show_edges())
    assert default_scene[1].size_channel.data == pytest.approx(0.005)
    assert default_scene[1].size_channel.space == "scene"
    scale = float(
        np.cbrt(abs(np.linalg.det(default_scene[1].global_transform[:3, :3])))
    )
    default_scene.resolve_size_spaces(hkw.config())
    assert default_scene[1].size_channel.space == "world"
    assert default_scene[1].size_channel.data * scale == pytest.approx(0.005)

    screen_scene = hkw.compile(
        hkw.layer(mesh).show_edges(width=2.0, width_space="screen")
    )
    screen_scale = float(
        np.cbrt(abs(np.linalg.det(screen_scene[1].global_transform[:3, :3])))
    )
    config = hkw.config()
    screen_scene.resolve_size_spaces(config)
    distance = np.linalg.norm(
        np.asarray(config.sensor.location) - np.asarray(config.sensor.target)
    )
    pixel_size = (
        2 * distance * np.tan(np.radians(config.sensor.fov) / 2) / config.film.height
    )
    assert screen_scene[1].size_channel.data * screen_scale == pytest.approx(pixel_size)


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


def test_grid_wraps_row_major_and_centers_ragged_row():
    mesh = _triangle()
    layers = [hkw.layer(mesh).name(str(index)) for index in range(5)]

    scene = hkw.compile(hkw.grid(layers, columns=3, row_axis="y"))
    assert [view.name for view in scene] == [str(index) for index in range(5)]
    centers = {view.name: view.bbox.mean(axis=0) for view in scene}

    assert centers["0"][0] < centers["1"][0] < centers["2"][0]
    assert centers["3"][0] < centers["4"][0]
    assert centers["0"][1] > centers["3"][1]
    assert np.mean([centers[str(i)][0] for i in range(3)]) == pytest.approx(
        np.mean([centers[str(i)][0] for i in range(3, 5)])
    )

    rows_scene = hkw.compile(hkw.grid(layers, rows=2, row_axis="y"))
    rows_centers = {view.name: view.bbox.mean(axis=0) for view in rows_scene}
    for name in centers:
        np.testing.assert_allclose(rows_centers[name], centers[name])


def test_grid_controls_row_and_column_spacing_independently():
    mesh = _triangle()
    layers = [hkw.layer(mesh).name(str(index)) for index in range(4)]

    scene = hkw.compile(hkw.grid(layers, columns=2, row_gap=0.1, column_gap=0.5))
    centers = {view.name: view.bbox.mean(axis=0) for view in scene}
    column_distance = abs(centers["1"][0] - centers["0"][0])
    row_distance = abs(centers["0"][1] - centers["2"][1])
    assert column_distance / row_distance == pytest.approx(1.5 / 1.1)

    shared = hkw.compile(hkw.grid(layers, columns=2, gap=0.2))
    explicit = hkw.compile(hkw.grid(layers, columns=2, row_gap=0.2, column_gap=0.2))
    for shared_view, explicit_view in zip(shared, explicit):
        np.testing.assert_allclose(shared_view.bbox, explicit_view.bbox)


def test_negative_juxtaposition_gap_reduces_spacing_and_serializes():
    mesh = _triangle()
    zero_gap = hkw.layer(mesh).juxtapose(hkw.layer(mesh), gap=0.0)
    negative_gap = hkw.layer(mesh).juxtapose(hkw.layer(mesh), gap=-0.25)

    spec = hkw.to_spec(negative_gap, data_ids={id(mesh): "mesh"})
    assert spec.root.gap == -0.25

    zero_centers = [view.bbox.mean(axis=0) for view in hkw.compile(zero_gap)]
    negative_centers = [view.bbox.mean(axis=0) for view in hkw.compile(negative_gap)]
    assert abs(negative_centers[1][0] - negative_centers[0][0]) < abs(
        zero_centers[1][0] - zero_centers[0][0]
    )


def test_grid_accepts_negative_row_and_column_gaps():
    mesh = _triangle()
    layers = [hkw.layer(mesh).name(str(index)) for index in range(4)]
    zero_gap = hkw.grid(layers, columns=2, gap=0.0)
    negative_gap = hkw.grid(
        layers, columns=2, row_gap=-0.2, column_gap=-0.3
    )

    hkw.to_spec(negative_gap, data_ids={id(mesh): "mesh"})
    zero_centers = {
        view.name: view.bbox.mean(axis=0) for view in hkw.compile(zero_gap)
    }
    negative_centers = {
        view.name: view.bbox.mean(axis=0) for view in hkw.compile(negative_gap)
    }

    assert abs(negative_centers["1"][0] - negative_centers["0"][0]) < abs(
        zero_centers["1"][0] - zero_centers["0"][0]
    )
    assert abs(negative_centers["2"][1] - negative_centers["0"][1]) < abs(
        zero_centers["2"][1] - zero_centers["0"][1]
    )


def test_grid_normalizes_cells_across_full_and_ragged_rows():
    layers = []
    for index, scale in enumerate((1.0, 2.0, 4.0)):
        mesh = _triangle()
        mesh.vertices *= scale
        layers.append(hkw.layer(mesh).name(str(index)))

    scene = hkw.compile(hkw.grid(layers, columns=2, normalize=True))
    radii = []
    for view in scene:
        points = np.asarray(view.data_frame.mesh.vertices)
        transform = view.global_transform
        world = (transform[:3, :3] @ points.T).T + transform[:3, 3]
        radii.append(np.linalg.norm(world - world.mean(axis=0), axis=1).max())

    np.testing.assert_allclose(radii, radii[0])


def test_grid_round_trips_as_standard_layout_nodes():
    mesh = _triangle()
    layer = hkw.grid(
        [hkw.layer(mesh).name(str(index)) for index in range(3)],
        columns=2,
        row_axis="z",
    )

    spec = hkw.to_spec(layer, data_ids={id(mesh): "mesh"})
    restored = hkw.from_spec(spec, data_resolver={"mesh": mesh})

    assert hkw.to_spec(restored, data_ids={id(mesh): "mesh"}).to_json(
        canonical=True
    ) == spec.to_json(canonical=True)


def test_grid_rejects_ambiguous_or_invalid_extents():
    layer = hkw.layer(_triangle())

    with pytest.raises(ValueError, match="exactly one"):
        hkw.grid([layer])
    with pytest.raises(ValueError, match="exactly one"):
        hkw.grid([layer], columns=1, rows=1)
    with pytest.raises(ValueError, match="positive integers"):
        hkw.grid([layer], columns=0)


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
