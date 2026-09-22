import json

import hakowan as hkw
import lagrange
import numpy as np
import pytest

from hakowan.spec import compile_expression


def test_inspect_returns_json_safe_geometry_and_attribute_statistics(triangle):
    summary = hkw.inspect(triangle)

    assert summary.vertex_count == 3
    assert summary.facet_count == 1
    assert summary.bounds == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]

    attribute = summary.attribute("vertex_data")
    assert attribute.element == "vertex"
    assert attribute.usage == "scalar"
    assert attribute.channels == 1
    assert attribute.minimum == 1.0
    assert attribute.maximum == 3.0
    assert attribute.quantiles["0.5"] == 2.0

    payload = summary.to_dict()
    assert isinstance(payload["attributes"], list)
    json.dumps(payload)


def test_inspect_reports_indexed_value_and_element_counts(triangle):
    uv = hkw.inspect(triangle).attribute("uv")

    assert uv.indexed
    assert uv.channels == 2
    assert uv.value_count == 3
    assert uv.element_count == triangle.num_corners


def test_inspect_loads_mesh_file(triangle, tmp_path):
    path = tmp_path / "triangle.ply"
    lagrange.io.save_mesh(path, triangle)

    summary = hkw.inspect(path)

    assert summary.source == str(path)
    assert summary.vertex_count == 3
    assert summary.facet_count == 1


def test_inspect_loads_point_cloud_attributes_from_file(tmp_path):
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    mesh.create_attribute(
        "pressure",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.array([1.0, 3.0]),
    )
    path = tmp_path / "points.ply"
    lagrange.io.save_mesh(path, mesh)

    summary = hkw.inspect(path)

    assert summary.facet_count == 0
    pressure = summary.attribute("pressure")
    assert pressure.element == "vertex"
    assert pressure.value_count == 2
    assert pressure.minimum == 1.0
    assert pressure.maximum == 3.0


def test_validate_accepts_valid_scalar_color_layer(triangle):
    layer = hkw.layer(triangle).material(
        "Diffuse", hkw.texture.ScalarField("vertex_data")
    )

    report = hkw.validate(layer, backend="webgl")

    assert report.valid
    assert report.diagnostics == ()
    json.dumps(report.to_dict())


def test_validate_reports_missing_attribute_with_path_and_candidates(triangle):
    layer = hkw.layer(triangle).mark("Point").channel(size="missing")

    report = hkw.validate(layer, backend="webgl")

    assert not report.valid
    diagnostic = report.errors[0]
    assert diagnostic.code == "attribute.missing"
    assert diagnostic.path == "views[0].channels.size.data"
    assert "vertex_data" in diagnostic.hint
    with pytest.raises(hkw.ValidationError, match="missing"):
        report.raise_for_errors()


def test_strict_mode_promotes_ignored_channel_to_error(triangle):
    layer = hkw.layer(triangle).mark("Surface").channel(size=0.1)

    strict = hkw.validate(layer, backend="webgl", strict=True)
    permissive = hkw.validate(layer, backend="webgl", strict=False)

    assert not strict.valid
    assert strict.errors[0].code == "channel.mark_incompatible"
    assert permissive.valid
    assert permissive.warnings[0].code == "channel.mark_incompatible"


def test_validate_rejects_nonpositive_log_input_in_strict_mode(triangle):
    triangle.create_attribute(
        "signed",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.array([-1.0, 0.0, 2.0]),
    )
    field = hkw.attribute("signed", scale=hkw.scale.Log())
    layer = hkw.layer(triangle).material(
        "Diffuse", hkw.texture.ScalarField(field, colormap="identity")
    )

    strict = hkw.validate(layer, backend="webgl", strict=True)
    permissive = hkw.validate(layer, backend="webgl", strict=False)

    assert any(item.code == "scale.log.nonpositive" for item in strict.errors)
    assert any(item.code == "scale.log.nonpositive" for item in permissive.warnings)


def test_validate_recognizes_attributes_generated_by_transform(triangle):
    layer = (
        hkw.layer(triangle)
        .transform(hkw.transform.Compute(component="component"))
        .material(
            "Diffuse",
            hkw.texture.ScalarField("component", categories=True),
        )
    )

    report = hkw.validate(layer, backend="webgl")

    assert report.valid
    assert not any(item.code == "attribute.missing" for item in report.diagnostics)


def test_validate_accepts_serializable_filter_expression(triangle):
    layer = hkw.layer(triangle).transform(
        hkw.transform.Filter(
            data="vertex_data", condition=compile_expression("value >= 2")
        )
    )

    report = hkw.validate(layer, backend="webgl", compile_check=False)

    assert report.valid
    assert not any(
        item.code == "transform.filter.callable" for item in report.diagnostics
    )


def test_validate_ignores_intentionally_overridden_channel(triangle):
    layer = hkw.layer(triangle).material("Diffuse", "red").material("Diffuse", "blue")

    report = hkw.validate(layer, backend="webgl", compile_check=False)

    assert report.valid
    assert not any(item.code == "channel.shadowed" for item in report.diagnostics)


def test_validate_scaled_uv_with_preserved_attributes(triangle, tmp_path):
    texture = tmp_path / "texture.png"
    texture.touch()
    layer = hkw.layer(triangle).material(
        "Principled",
        color=hkw.texture.Image(texture, uv=hkw.attribute("uv", scale=2)),
    )

    report = hkw.validate(layer, backend="webgl")

    assert report.valid
    scene = hkw.compile(layer, preserve_attributes=True)
    uv_ids = scene[0].data_frame.mesh.get_matching_attribute_ids(
        usage=lagrange.AttributeUsage.UV
    )
    assert len(uv_ids) == 1


def test_validate_compile_check_catches_runtime_failure_without_mutation(triangle):
    texture = hkw.texture.ScalarField("vertex_data")
    layer = (
        hkw.layer(triangle)
        .transform(hkw.transform.Affine(matrix=np.ones((2, 2))))
        .material("Diffuse", texture)
    )

    report = hkw.validate(layer, backend="webgl")

    assert not report.valid
    assert any(item.code == "compile.failed" for item in report.errors)
    assert texture.data == "vertex_data"


def test_backend_capabilities_are_available_without_loading_backend():
    capabilities = hkw.backend_capabilities("blender")

    assert capabilities.marks == frozenset({"point", "curve", "surface"})
    assert "facet_id" in capabilities.render_passes
    assert "fur_children" in capabilities.features
    json.dumps(capabilities.to_dict())
    assert set(hkw.list_backend_capabilities()) == {"blender", "mitsuba", "webgl"}


def test_unknown_backend_is_a_structured_error(triangle):
    report = hkw.validate(hkw.layer(triangle), backend="bogus")  # type: ignore[arg-type]

    assert not report.valid
    assert report.errors[0].code == "backend.unknown"
    assert report.errors[0].path == "backend"


def test_validate_reports_empty_geometry_after_transform(triangle):
    layer = hkw.layer(triangle).transform(
        hkw.transform.Filter(data="vertex_data", condition=lambda _: False)
    )

    report = hkw.validate(layer, backend="webgl")

    diagnostic = next(
        item for item in report.errors if item.code == "geometry.empty_after_transform"
    )
    assert diagnostic.path == "views[0].geometry"
    assert "relax" in diagnostic.hint.lower()


def test_validate_reports_scene_behind_camera(triangle):
    figure = hkw.figure(hkw.layer(triangle)).camera(
        "perspective", eye=(0, 0, 5), target=(0, 0, 10)
    )

    report = hkw.validate(figure, backend="webgl")

    diagnostic = next(
        item for item in report.errors if item.code == "camera.scene_behind"
    )
    assert diagnostic.path == "scene.camera"
    assert "target" in diagnostic.hint


def test_validate_reports_scene_outside_field_of_view(triangle):
    figure = hkw.figure(hkw.layer(triangle)).camera(
        "perspective", eye=(0, 0, 5), target=(1, 0, 5), fov=20
    )

    report = hkw.validate(figure, backend="webgl")

    assert any(item.code == "camera.scene_outside_view" for item in report.errors)


def test_validate_reports_complete_and_intersecting_clipping(triangle):
    outside = hkw.figure(hkw.layer(triangle)).camera(
        "perspective", eye=(0, 0, 5), near=7, far=10
    )
    intersecting = hkw.figure(hkw.layer(triangle)).camera(
        "perspective", eye=(0, 0, 5), near=5, far=10
    )

    outside_report = hkw.validate(outside, backend="webgl")
    intersecting_report = hkw.validate(intersecting, backend="webgl")

    assert any(item.code == "camera.clipping.outside" for item in outside_report.errors)
    near = next(
        item
        for item in intersecting_report.warnings
        if item.code == "camera.clipping.near"
    )
    assert "Reduce near" in near.hint
    far_intersecting = hkw.figure(hkw.layer(triangle)).camera(
        "perspective", eye=(0, 0, 5), near=0.01, far=5
    )
    far_report = hkw.validate(far_intersecting, backend="webgl")
    far = next(
        item for item in far_report.warnings if item.code == "camera.clipping.far"
    )
    assert "Increase far" in far.hint


def test_validate_warns_about_severe_projected_occlusion():
    front = lagrange.SurfaceMesh()
    front.add_vertices(np.array([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]]))
    front.add_triangle(0, 1, 2)
    back = lagrange.SurfaceMesh()
    back.add_vertices(np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]]))
    back.add_triangle(0, 1, 2)
    figure = hkw.figure(hkw.layer(front) + hkw.layer(back)).camera(
        "perspective", eye=(0, 0, 5), target=(0, 0, 0)
    )

    report = hkw.validate(figure, backend="webgl")

    diagnostic = next(
        item for item in report.warnings if item.code == "layer.possible_occlusion"
    )
    assert diagnostic.path == "views[1]"
    assert "another camera" in diagnostic.hint


def test_compile_check_can_disable_geometry_diagnostics(triangle):
    layer = hkw.layer(triangle).transform(
        hkw.transform.Filter(data="vertex_data", condition=lambda _: False)
    )

    report = hkw.validate(layer, backend="webgl", compile_check=False)

    assert not any(
        item.code == "geometry.empty_after_transform" for item in report.diagnostics
    )
