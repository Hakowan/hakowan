"""World annotations follow the same geometry transforms as their owning view."""

import lagrange
import numpy as np
import pytest

import hakowan as hkw


def _triangle():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    mesh.add_triangle(0, 1, 2)
    return mesh


def test_world_callouts_follow_layout_and_remain_attached_to_views():
    mesh = _triangle()
    layer = hkw.layer(mesh).annotate(hkw.WorldAnnotation("vertex", position=(0, 0, 0)))
    scene = hkw.compile(layer | layer)

    assert len(scene.world_annotations) == 2
    assert scene.annotations == []
    assert {item.layer for item in scene.world_annotations} == {0, 1}
    assert {item.cell for item in scene.world_annotations} == {"0.0", "0.1"}
    first, second = scene.world_annotations
    assert first.position[0] < second.position[0]
    for item in scene.world_annotations:
        view = scene[item.layer]
        local_vertex = np.asarray(view.data_frame.mesh.vertices[0], dtype=float)
        world_vertex = view.global_transform @ np.append(local_vertex, 1.0)
        np.testing.assert_allclose(item.position, world_vertex[:3], atol=1e-6)


def test_world_annotations_coexist_with_fixed_screen_annotations():
    layer = (
        hkw.layer(_triangle())
        .annotate("Screen", position=(0.1, 0.2))
        .annotate(hkw.WorldAnnotation("Vertex", position=(1, 0, 0)))
    )
    scene = hkw.compile(layer)

    assert [item.text for item in scene.annotations] == ["Screen"]
    assert [item.text for item in scene.world_annotations] == ["Vertex"]


@pytest.mark.parametrize(
    "position",
    [(0, 1), (0, 1, 2, 3), (0, float("nan"), 1), (0, 1, float("inf"))],
)
def test_world_annotation_rejects_invalid_coordinates(position):
    with pytest.raises(ValueError, match="three finite coordinates"):
        hkw.WorldAnnotation("test", position=position)


def test_browser_callouts_track_rotation_and_layer_visibility(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    layer = hkw.layer(_triangle()).annotate(
        hkw.WorldAnnotation("Point", position=(0.25, 0.25, 0.04))
    )
    viewer = hkw.render(
        layer | layer, filename=tmp_path / "callouts.html", backend="webgl", offline=True
    )

    with playwright.sync_playwright() as runtime:
        try:
            browser = runtime.chromium.launch(
                headless=True, args=["--allow-file-access-from-files"]
            )
        except playwright.Error as exc:
            pytest.skip(f"Playwright Chromium is unavailable: {exc}")
        try:
            page = browser.new_page(viewport={"width": 1100, "height": 620})
            page.goto(viewer.path.resolve().as_uri(), wait_until="domcontentloaded")
            page.wait_for_function("window.__hakowan_loaded === true")
            labels = page.locator("#world-callouts .world-callout")
            assert labels.count() == 2
            assert labels.first.is_visible()
            before = page.locator("#world-callouts svg path").first.get_attribute("d")

            canvas = page.locator("canvas").bounding_box()
            assert canvas is not None
            page.mouse.move(
                canvas["x"] + canvas["width"] * 0.4,
                canvas["y"] + canvas["height"] * 0.5,
            )
            page.mouse.down()
            page.mouse.move(
                canvas["x"] + canvas["width"] * 0.45,
                canvas["y"] + canvas["height"] * 0.52,
                steps=12,
            )
            page.mouse.up()
            page.wait_for_function(
                "before => document.querySelector('#world-callouts svg path')"
                ".getAttribute('d') !== before",
                arg=before,
            )
            assert labels.first.is_visible()

            page.locator("#layers input").first.uncheck()
            assert not labels.first.is_visible()
            assert labels.nth(1).is_visible()
            page.locator('#passes button[data-pass="normal"]').click()
            assert not labels.nth(1).is_visible()
            page.locator('#passes button[data-pass="beauty"]').click()
            assert labels.nth(1).is_visible()
        finally:
            browser.close()
