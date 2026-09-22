from __future__ import annotations

import json

import lagrange
import numpy as np
import pytest
from PIL import Image

import hakowan as hkw


def _point_mesh(points, values):
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.asarray(points, dtype=np.float64))
    mesh.create_attribute(
        "value",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.asarray(values, dtype=np.float64).reshape(-1, 1),
    )
    return mesh


def _synthetic_observation():
    first = hkw.layer(
        _point_mesh([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [1, 2, 5]),
        name="first",
    ).mark("point")
    second = hkw.layer(
        _point_mesh([[0, 1, 0], [1, 1, 0]], [10, 20]),
        name="second",
    ).mark("point")
    scene = hkw.compile(first + second, preserve_attributes=True)
    layer = np.full((4, 6), hkw.observation.BACKGROUND_ID, dtype=np.uint32)
    element = np.full_like(layer, hkw.observation.BACKGROUND_ID)
    depth = np.full((4, 6), np.nan, dtype=np.float32)
    layer[1:3, 1:3] = 0
    element[1, 1] = 0
    element[1, 2] = 2
    element[2, 1] = 0
    element[2, 2] = 2
    depth[1:3, 1:3] = [[2.0, 2.5], [2.0, 2.5]]
    layer[0:2, 4:6] = 1
    element[0, 4] = 0
    element[0, 5] = 1
    element[1, 4] = 0
    element[1, 5] = 1
    depth[0:2, 4:6] = 3.0
    image = Image.new("RGBA", (6, 4), "black")
    camera = hkw.CameraState(eye=(0, 0, 5), target=(0, 0, 0), up=(0, 1, 0))
    snapshots = {
        ("front", "layer_id"): hkw.Snapshot(
            image=image, data=layer, view="front", pass_name="layer_id", camera=camera
        ),
        ("front", "element_id"): hkw.Snapshot(
            image=image,
            data=element,
            view="front",
            pass_name="element_id",
            camera=camera,
        ),
        ("front", "depth"): hkw.Snapshot(
            image=image, data=depth, view="front", pass_name="depth", camera=camera
        ),
    }
    return hkw.Observation(
        snapshots=snapshots,
        scene_summary=hkw.SceneSummary(
            bounds=((-1, -1, -1), (1, 1, 1)),
            center=(0, 0, 0),
            radius=1.0,
            up_axis="y",
            layers=(
                hkw.observation.LayerSummary(0, "first", "point", 3, 0),
                hkw.observation.LayerSummary(1, "second", "point", 2, 0),
            ),
        ),
        _scene=scene,
    )


def test_region_reports_background_layers_elements_bounds_and_depth():
    observation = _synthetic_observation()

    summary = observation.region(0, 0, 4, 4, view="front")

    assert summary.pixel_count == 16
    assert summary.background_pixel_count == 12
    assert summary.background_fraction == pytest.approx(0.75)
    assert len(summary.layers) == 1
    first = summary.layers[0]
    assert first.name == "first"
    assert first.visible_pixel_count == 4
    assert first.visible_element_ids == (0, 2)
    assert first.total_element_count == 3
    assert first.visible_element_fraction == pytest.approx(2 / 3)
    assert first.visible_bounds == (1, 1, 3, 3)
    assert first.projected_bounds is None
    assert first.depth_range == pytest.approx((2.0, 2.5))
    json.dumps(summary.to_dict())

    assert "visual.camera_clipping" in {
        item.code for item in observation.visual_diagnostics()
    }


def test_visible_elements_supports_layer_name_and_full_view():
    observation = _synthetic_observation()

    all_layers = observation.visible_elements(view="front")
    second = observation.visible_elements("second", view="front")

    assert [item.layer_id for item in all_layers] == [0, 1]
    assert second[0].visible_element_ids == (0, 1)
    assert second[0].visible_element_fraction == 1.0


def test_attribute_extrema_uses_only_visible_source_elements():
    observation = _synthetic_observation()

    result = observation.attribute_extrema("value", layer="first", view="front")[0]

    assert result.sample_count == 2
    assert result.minimum == 1.0
    assert result.maximum == 5.0
    assert result.mean == 3.0
    assert result.minimum_element_id == 0
    assert result.maximum_element_id == 2
    assert result.criterion == "value"
    json.dumps(result.to_dict())


def test_attribute_extrema_can_be_scoped_to_region():
    observation = _synthetic_observation()

    result = observation.attribute_extrema(
        "value", layer="first", view="front", bounds=(1, 1, 2, 3)
    )[0]

    assert result.sample_count == 1
    assert result.minimum == result.maximum == 1.0


def test_region_requires_id_passes_and_unambiguous_view():
    observation = _synthetic_observation()
    image = Image.new("RGBA", (6, 4), "black")
    observation.snapshots[("right", "layer_id")] = hkw.Snapshot(
        image=image,
        data=np.full((4, 6), hkw.observation.BACKGROUND_ID, dtype=np.uint32),
        view="right",
        pass_name="layer_id",
    )

    with pytest.raises(ValueError, match="requires view"):
        observation.region(0, 0, 2, 2)
    with pytest.raises(ValueError, match="element_id"):
        observation.region(0, 0, 2, 2, view="right")
    with pytest.raises(ValueError, match="outside"):
        observation.region(-1, 0, 2, 2, view="front")


def _triangle(z, scale):
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(
        np.array(
            [
                [-scale, -scale, z],
                [scale, -scale, z],
                [0.0, scale, z],
            ]
        )
    )
    mesh.add_triangle(0, 1, 2)
    return mesh


def test_browser_observation_queries_and_manifest():
    front_mesh = _triangle(1.0, 1.0)
    front_mesh.create_attribute(
        "stress",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.array([[1.0], [5.0], [3.0]]),
    )
    front = hkw.layer(front_mesh, name="front")
    back = hkw.layer(_triangle(0.0, 0.45), name="back")

    observation = hkw.observe(
        front + back,
        views=["front"],
        passes=["depth", "element_id", "layer_id"],
        resolution=(64, 64),
    )
    visibility = observation.visible_elements(view="front")
    occlusion = observation.occlusion_report(view="front")
    stress = observation.attribute_extrema("stress", layer="front", view="front")[0]

    assert any(
        item.name == "front" and item.visible_pixel_count > 0 for item in visibility
    )
    back_visibility = next(item for item in visibility if item.name == "back")
    assert back_visibility.visible_pixel_count == 0
    assert back_visibility.visible_element_ids == ()
    assert back_visibility.visible_element_fraction == 0.0
    assert stress.sample_count == 3
    assert stress.maximum == 5.0
    assert stress.maximum_element_id == 1
    assert any(
        item.occluded_layer_name == "back"
        and item.occluder_layer_name == "front"
        and item.projected_coverage > 0.5
        and item.fully_hidden
        for item in occlusion
    )
    assert "visibility" in observation.manifest
    assert "occlusion" in observation.manifest
    json.dumps(observation.manifest)
    evidence = observation.visual_evidence()["views"]["front"]
    back_evidence = next(item for item in evidence["layers"] if item["name"] == "back")
    assert back_evidence["occluded_fraction_estimate"] == 1.0
    assert back_evidence["visible_pixel_count"] == 0
    codes = {item.code for item in observation.visual_diagnostics()}
    assert "visual.layer_hidden" in codes
    assert "visual.layer_occluded" in codes
