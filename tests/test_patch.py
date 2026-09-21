from __future__ import annotations

import copy

import lagrange
import numpy as np
import pytest

import hakowan as hkw


def _mesh():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    mesh.add_triangle(0, 1, 2)
    mesh.create_attribute(
        "value",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.array([[1.0], [2.0], [3.0]]),
    )
    return mesh


def test_patch_replaces_runtime_figure_without_mutating_original():
    mesh = _mesh()
    original = hkw.figure(hkw.layer(mesh)).camera("perspective", eye=(0, 0, 5))

    updated = hkw.patch(
        original,
        [{"op": "replace", "path": "/scene/camera/eye", "value": [2, 3, 4]}],
    )

    assert isinstance(updated, hkw.Figure)
    assert updated.scene.camera.eye == (2, 3, 4)
    assert original.scene.camera.eye == (0, 0, 5)
    assert updated.layer._spec.data.mesh is mesh


def test_patch_supports_add_remove_and_array_append():
    mesh = _mesh()
    original = hkw.figure(hkw.layer(mesh)).clear_lights()

    updated = hkw.patch(
        original,
        [
            {
                "op": "add",
                "path": "/scene/lights/-",
                "value": {
                    "kind": "directional",
                    "direction": [0, 0, -1],
                    "color": "white",
                    "intensity": 2,
                },
            },
            {"op": "replace", "path": "/scene/lights/0/intensity", "value": 4},
        ],
    )
    removed = hkw.patch(
        updated,
        [{"op": "remove", "path": "/scene/lights/0"}],
    )

    assert updated.scene.lights[0].intensity == 4
    assert removed.scene.lights == ()
    assert original.scene.lights == ()


def test_patch_can_promote_layer_spec_to_figure():
    mesh = _mesh()
    layer = hkw.layer(mesh)

    updated = hkw.patch(
        layer,
        [
            {"op": "replace", "path": "/version", "value": "1.1"},
            {
                "op": "replace",
                "path": "/scene",
                "value": {
                    "camera": {
                        "kind": "perspective",
                        "eye": [0, 0, 5],
                        "target": [0, 0, 0],
                        "up": [0, 1, 0],
                        "fov": 35,
                        "fov_axis": "smaller",
                        "near": 0.01,
                        "far": 100,
                    },
                    "lights": None,
                    "environment": None,
                    "output": None,
                },
            },
        ],
    )

    assert isinstance(updated, hkw.Figure)
    assert isinstance(updated.scene.camera, hkw.PerspectiveCamera)


def test_patch_spec_returns_validated_immutable_model():
    mesh = _mesh()
    spec = hkw.to_spec(hkw.figure(hkw.layer(mesh)).output(width=100), data_ids={id(mesh): "mesh"})
    before = copy.deepcopy(spec.to_dict())

    updated = hkw.patch_spec(
        spec,
        [{"op": "replace", "path": "/scene/output/width", "value": 640}],
    )

    assert updated.scene.output.width == 640
    assert spec.to_dict() == before


def test_operation_failure_is_path_specific_and_atomic():
    mesh = _mesh()
    original = hkw.figure(hkw.layer(mesh)).output(width=100)

    with pytest.raises(hkw.PatchError) as captured:
        hkw.patch(
            original,
            [
                {"op": "replace", "path": "/scene/output/width", "value": 200},
                {"op": "remove", "path": "/scene/output/missing"},
            ],
        )

    assert captured.value.failure.code == "patch.operation"
    assert captured.value.failure.operation_index == 1
    assert captured.value.failure.path == "/scene/output/missing"
    assert original.scene.output.width == 100


def test_schema_failure_reports_json_pointer_and_is_atomic():
    mesh = _mesh()
    original = hkw.figure(hkw.layer(mesh)).camera("perspective")

    with pytest.raises(hkw.PatchError) as captured:
        hkw.patch(
            original,
            [{"op": "replace", "path": "/scene/camera/fov", "value": 200}],
        )

    assert captured.value.failure.code == "patch.schema"
    assert captured.value.failure.path.endswith("/fov")
    assert original.scene.camera.fov < 180


def test_semantic_failure_exposes_validation_report_and_is_atomic():
    mesh = _mesh()
    original = hkw.layer(mesh).mark("point").channel(size="value")

    with pytest.raises(hkw.PatchError) as captured:
        hkw.patch(
            original,
            [
                {
                    "op": "replace",
                    "path": "/root/spec/channels/size/data/name",
                    "value": "missing",
                }
            ],
        )

    error = captured.value
    assert error.failure.code == "patch.semantic"
    assert error.failure.path == "views[0].channels.size.data"
    assert error.validation_report is not None
    assert original._spec.channels[0].data.name == "value"


def test_semantic_validation_can_be_disabled_explicitly():
    mesh = _mesh()
    original = hkw.layer(mesh).mark("point").channel(size="value")

    updated = hkw.patch(
        original,
        [
            {
                "op": "replace",
                "path": "/root/spec/channels/size/data/name",
                "value": "missing",
            }
        ],
        semantic=False,
    )

    assert updated._spec.channels[0].data.name == "missing"


def test_patch_preserves_runtime_callable_bindings():
    mesh = _mesh()

    def keep(value):
        return value[0] > 0

    original = hkw.layer(mesh).transform(
        hkw.transform.Filter(data="value", condition=keep)
    )

    updated = hkw.patch(
        original,
        [{"op": "replace", "path": "/root/spec/name", "value": "filtered"}],
        semantic=False,
    )

    assert updated._spec.name == "filtered"
    assert updated._spec.transform.condition is keep


def test_patch_can_resolve_new_external_data_identifier():
    first = _mesh()
    second = _mesh()
    original = hkw.layer(first)

    updated = hkw.patch(
        original,
        [{"op": "replace", "path": "/root/spec/data/id", "value": "replacement"}],
        data_resolver={"replacement": second},
    )

    assert updated._spec.data.mesh is second


def test_patch_rejects_invalid_pointer_and_root_removal():
    mesh = _mesh()
    figure = hkw.figure(hkw.layer(mesh))

    with pytest.raises(hkw.PatchError, match="start with"):
        hkw.patch(figure, [{"op": "replace", "path": "scene", "value": None}])
    with pytest.raises(hkw.PatchError, match="document root"):
        hkw.patch(figure, [{"op": "remove", "path": ""}])
