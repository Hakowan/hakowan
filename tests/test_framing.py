from __future__ import annotations

import json

import lagrange
import numpy as np
import pytest

import hakowan as hkw


def _point_mesh(points, **attributes):
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.asarray(points, dtype=np.float64))
    for name, values in attributes.items():
        array = np.asarray(values)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        mesh.create_attribute(
            name,
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=np.array(array, copy=True),
        )
    return mesh


def _world_points(view):
    points = np.asarray(view.data_frame.mesh.vertices, dtype=np.float64)
    transform = np.asarray(view.global_transform, dtype=np.float64)
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def _direction(camera):
    value = np.asarray(camera.eye) - np.asarray(camera.target)
    return value / np.linalg.norm(value)


def test_fit_camera_frames_scene_and_resolves_to_concrete_camera():
    layer = hkw.layer(np.array([[-4.0, -1.0, 0.0], [4.0, -1.0, 0.0], [0.0, 2.0, 0.0]]))

    figure = hkw.figure(layer).camera("fit", direction="front", margin=0.1, fov=30)
    camera = figure.scene.camera

    assert isinstance(camera, hkw.PerspectiveCamera)
    np.testing.assert_allclose(_direction(camera), [0.0, 0.0, 1.0])
    assert camera.near > 0
    assert camera.far > camera.near
    assert hkw.validate(figure, backend="webgl").valid


def test_fit_camera_targets_named_layer():
    left = hkw.layer(np.array([[-10.0, 0.0, 0.0], [-9.0, 1.0, 0.0]]), name="left")
    right = hkw.layer(np.array([[9.0, 0.0, 0.0], [10.0, 1.0, 0.0]]), name="right")
    root = left + right
    scene = hkw.compile(root)
    expected = (
        _world_points(scene[0]).min(axis=0) + _world_points(scene[0]).max(axis=0)
    ) / 2

    figure = hkw.figure(root).camera("fit", layer="left", direction="front")

    np.testing.assert_allclose(figure.scene.camera.target, expected)
    with pytest.raises(ValueError, match="available names"):
        hkw.figure(root).camera("fit", layer="missing")


def test_fit_camera_accepts_explicit_bounds():
    figure = hkw.figure(hkw.layer(np.eye(3))).camera(
        "fit", bounds=((-2, -1, 3), (4, 5, 7)), direction="front"
    )

    assert figure.scene.camera.target == pytest.approx((1.0, 2.0, 5.0))


def test_principal_axis_camera_tracks_longest_axis():
    points = np.array([[-5.0, 0.0, 0.0], [0.0, 0.1, 0.0], [5.0, 0.0, 0.0]])

    camera = (
        hkw.figure(hkw.layer(points))
        .camera("principal_axis", axis=0, sign="+")
        .scene.camera
    )

    assert _direction(camera)[0] == pytest.approx(1.0)
    assert abs(_direction(camera)[1]) < 1e-8
    assert abs(_direction(camera)[2]) < 1e-8


def test_attribute_extremum_targets_matching_vertex():
    mesh = _point_mesh(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        stress=[1.0, 9.0, 3.0],
    )
    layer = hkw.layer(mesh, name="sample")
    expected = _world_points(hkw.compile(layer, preserve_attributes=True)[0])[1]

    camera = (
        hkw.figure(layer)
        .camera(
            "attribute_extremum", attribute="stress", extremum="max", direction="front"
        )
        .scene.camera
    )

    np.testing.assert_allclose(camera.target, expected)


def test_component_selection_fits_only_requested_points():
    mesh = _point_mesh(
        [[-4.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        component=[0, 0, 1],
    )
    layer = hkw.layer(mesh)
    world = _world_points(hkw.compile(layer, preserve_attributes=True)[0])
    expected = (world[:2].min(axis=0) + world[:2].max(axis=0)) / 2

    camera = (
        hkw.figure(layer)
        .camera("fit", component=("component", 0), direction="front")
        .scene.camera
    )

    np.testing.assert_allclose(camera.target, expected)


def test_section_camera_targets_plane_and_looks_along_normal():
    camera = (
        hkw.figure(hkw.layer(np.eye(3)))
        .camera("section", normal=(0, 0, 1), offset=0.25, projection="orthographic")
        .scene.camera
    )

    assert isinstance(camera, hkw.OrthographicCamera)
    assert np.dot(camera.target, [0, 0, 1]) == pytest.approx(0.25)
    np.testing.assert_allclose(_direction(camera), [0, 0, 1])
    assert camera.scale > 0


def test_turntable_returns_evenly_spaced_immutable_figures():
    base = hkw.figure(hkw.layer(np.eye(3)))

    frames = base.turntable(count=4, elevation=0)
    directions = np.asarray([_direction(frame.scene.camera) for frame in frames])

    assert base.scene.camera is None
    assert len(frames) == 4
    np.testing.assert_allclose(
        directions,
        [[0, 0, 1], [1, 0, 0], [0, 0, -1], [-1, 0, 0]],
        atol=1e-7,
    )


def test_orthographic_scale_round_trips_and_reaches_webgl(tmp_path):
    mesh = _point_mesh([[-2.0, -1.0, 0.0], [2.0, 1.0, 0.0]])
    figure = hkw.figure(hkw.layer(mesh)).camera(
        "fit", projection="orthographic", direction="front", margin=0.2
    )
    spec = hkw.to_spec(figure, data_ids={id(mesh): "mesh"})
    restored = hkw.from_spec(spec, data_resolver={"mesh": mesh})
    output = tmp_path / "orthographic.html"
    hkw.render(restored, filename=output, backend="webgl")
    html = output.read_text(encoding="utf-8")

    assert restored.scene.camera.scale == pytest.approx(figure.scene.camera.scale)
    assert 'const INITIAL_CAMERA_MODE = "orthographic"' in html
    assert f"const INITIAL_ORTHO_SCALE = {figure.scene.camera.scale}" in html
    json.loads(spec.to_json())


def test_framing_rejects_invalid_selections():
    figure = hkw.figure(hkw.layer(np.eye(3)))

    with pytest.raises(ValueError, match="non-negative"):
        figure.camera("fit", margin=-0.1)
    with pytest.raises(ValueError, match="attribute="):
        figure.camera("attribute_extremum")
    with pytest.raises(ValueError, match="normal="):
        figure.camera("section")
    with pytest.raises(ValueError, match="positive"):
        figure.turntable(count=0)


def test_framing_handles_zero_extent_and_rejects_ambiguous_axes():
    point = hkw.figure(hkw.layer(np.array([[0.0, 0.0, 0.0]])))
    camera = point.camera("fit").scene.camera
    assert camera is not None
    assert camera.near < camera.far

    symmetric = hkw.figure(
        hkw.layer(
            np.array(
                [
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            )
        )
    )
    with pytest.raises(ValueError, match="not unique"):
        symmetric.camera("principal_axis", axis=0)
