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


def _box():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(
        np.array(
            [
                [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
            ]
        )
    )
    mesh.add_triangles(
        np.array(
            [
                [4, 5, 6], [4, 6, 7], [0, 2, 1], [0, 3, 2],
                [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
                [3, 7, 6], [3, 6, 2], [0, 1, 5], [0, 5, 4],
            ],
            dtype=np.uint32,
        )
    )
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


@pytest.mark.parametrize(
    "offset",
    [(0,), (0, 1, 2), (0, float("nan")), (float("inf"), 1)],
)
def test_world_annotation_rejects_invalid_offset(offset):
    with pytest.raises(ValueError, match="two finite coordinates"):
        hkw.WorldAnnotation("test", position=(0, 0, 0), offset=offset)


def test_world_annotation_offset_and_arrow_properties():
    wa = hkw.WorldAnnotation(
        "label",
        position=(1.0, 2.0, 3.0),
        offset=(10.0, -20.0),
        arrow=False,
    )
    assert wa.offset == (10.0, -20.0)
    assert wa.arrow is False

    scene = hkw.compile(hkw.layer(_triangle()).annotate(wa))
    assert len(scene.world_annotations) == 1
    cwa = scene.world_annotations[0]
    assert cwa.offset == (10.0, -20.0)
    assert cwa.arrow is False


def test_browser_callout_offset_and_optional_arrow(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    layer = (
        hkw.layer(_triangle())
        .annotate(
            hkw.WorldAnnotation(
                "Above",
                position=(0.25, 0.25, 0.04),
                offset=(0.0, -10.0),
                arrow=False,
            )
        )
        .annotate(
            hkw.WorldAnnotation(
                "Right",
                position=(0.75, 0.2, 0.04),
                offset=(16.0, 0.0),
                arrow=True,
            )
        )
    )
    viewer = hkw.render(
        layer, filename=tmp_path / "callout_offset.html", backend="webgl", offline=True
    )
    compiled = {
        item.text: item.position for item in hkw.compile(layer).world_annotations
    }

    with playwright.sync_playwright() as runtime:
        try:
            browser = runtime.chromium.launch(
                headless=True, args=["--allow-file-access-from-files"]
            )
        except playwright.Error as exc:
            pytest.skip(f"Playwright Chromium is unavailable: {exc}")
        try:
            page = browser.new_page(viewport={"width": 800, "height": 600})
            page.goto(viewer.path.resolve().as_uri(), wait_until="domcontentloaded")
            page.wait_for_function("window.__hakowan_loaded === true")
            page.wait_for_function(
                "document.querySelectorAll('#world-callouts .world-callout').length === 2"
            )
            placed = page.evaluate(
                """(positions) => {
                  const h = window.__hakowan;
                  const T = h.THREE;
                  const cam = h.getCamera();
                  const rect = document.getElementById('world-callouts').getBoundingClientRect();
                  const paths = [...document.querySelectorAll('#world-callouts svg path')];
                  return [...document.querySelectorAll('#world-callouts .world-callout')].map((label, index) => {
                    const p = new T.Vector3(...positions[label.textContent]).project(cam);
                    return {
                      text: label.textContent,
                      left: parseFloat(label.style.left),
                      top: parseFloat(label.style.top),
                      width: label.offsetWidth,
                      height: label.offsetHeight,
                      ax: (p.x + 1) * rect.width / 2,
                      ay: (1 - p.y) * rect.height / 2,
                      centerX: rect.width / 2,
                      centerY: rect.height / 2,
                      line: getComputedStyle(paths[index * 2]).visibility,
                      tip: getComputedStyle(paths[index * 2 + 1]).visibility,
                      d: paths[index * 2].getAttribute('d'),
                    };
                  });
                }""",
                compiled,
            )
            by_text = {item["text"]: item for item in placed}
            above = by_text["Above"]
            right = by_text["Right"]
            assert abs(above["ax"] - above["centerX"]) > 15 or abs(above["ay"] - above["centerY"]) > 15
            assert abs(above["left"] - (above["ax"] - above["width"] / 2)) < 1
            assert abs(above["top"] - (above["ay"] - 10 - above["height"])) < 1
            assert above["line"] == "hidden"
            assert above["tip"] == "hidden"
            assert abs(right["left"] - (right["ax"] + 16)) < 1
            assert abs(right["top"] - (right["ay"] - right["height"] / 2)) < 1
            assert right["line"] == "visible"
            assert right["tip"] == "visible"
            assert right["d"] and "NaN" not in right["d"]
        finally:
            browser.close()


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


def test_browser_box_front_face_callout_finds_space_outside_geometry(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    layer = (
        hkw.layer(_box())
        .annotate(hkw.WorldAnnotation("Front", position=(0, 0, 0.5)))
        .annotate(hkw.WorldAnnotation("Back", position=(0, 0, -0.5)))
    )
    viewer = hkw.render(
        layer, filename=tmp_path / "box.html", backend="webgl", offline=True
    )
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            headless=True, args=["--allow-file-access-from-files"]
        )
        try:
            page = browser.new_page(viewport={"width": 1024, "height": 800})
            page.goto(viewer.path.resolve().as_uri(), wait_until="domcontentloaded")
            page.wait_for_function("window.__hakowan_loaded === true")
            front = page.locator(".world-callout", has_text="Front")
            back = page.locator(".world-callout", has_text="Back")
            assert front.is_visible()
            assert not back.is_visible()
            assert page.evaluate(
                """() => {
                  const h = window.__hakowan, T = h.THREE;
                  const rect = [...document.querySelectorAll('.world-callout')]
                    .find(label => label.textContent === 'Front').getBoundingClientRect();
                  const ray = new T.Raycaster();
                  ray.setFromCamera(new T.Vector2(
                    2 * (rect.x + rect.width / 2) / innerWidth - 1,
                    1 - 2 * (rect.y + rect.height / 2) / innerHeight
                  ), h.getCamera());
                  return ray.intersectObject(h.loadedRoot, true)
                    .every(hit => !hit.object.isMesh);
                }"""
            )
        finally:
            browser.close()


def test_static_box_label_uses_same_browser_and_raster_placement(tmp_path):
    import base64
    import io
    from PIL import Image
    from hakowan.common.overlay import _place_world_label

    playwright = pytest.importorskip("playwright.sync_api")
    layer = hkw.layer(_box()).annotate(hkw.WorldAnnotation("Front", (0, 0, 0.5)))
    figure = hkw.figure(layer).camera("orthographic", eye=(1, 1, 5))
    viewer = hkw.render(
        figure, filename=tmp_path / "placement.html", backend="webgl", offline=True
    )
    anchor = hkw.compile(layer).world_annotations[0].position
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            headless=True, args=["--allow-file-access-from-files"]
        )
        try:
            page = browser.new_page(viewport={"width": 1024, "height": 800})
            page.goto(viewer.path.resolve().as_uri(), wait_until="domcontentloaded")
            page.wait_for_function("window.__hakowan_loaded === true")
            page.evaluate("() => window.__hakowan.prepareCapture(1024, 800)")
            normal_url = page.evaluate("() => window.__hakowan.capturePng('normal')")
            page.evaluate("() => window.__hakowan.setPass('beauty')")
            layout = page.evaluate(
                """anchor => {
                  const h = window.__hakowan;
                  const p = new h.THREE.Vector3(...anchor).project(h.getCamera());
                  const box = document.querySelector('.world-callout').getBoundingClientRect();
                  return {anchor: [(p.x + 1) * 512, (1 - p.y) * 400],
                    size: [box.width, box.height], rect: [box.left, box.top, box.right, box.bottom]};
                }""",
                list(anchor),
            )
            with Image.open(io.BytesIO(base64.b64decode(normal_url.split(",", 1)[1]))) as image:
                mask = np.any(np.asarray(image.convert("RGB")) != 0, axis=2)

            def geometry_at(x, y):
                return bool(mask[min(799, int(y)), min(1023, int(x))])

            raster = _place_world_label(
                layout["anchor"], layout["size"], (1024, 800), [], geometry_at
            )
            assert raster is not None
            np.testing.assert_allclose(raster, layout["rect"], atol=1.0)
        finally:
            browser.close()


def test_box_snapshot_places_world_text_outside_mesh():
    pytest.importorskip("playwright.sync_api")
    layer = hkw.layer(_box()).annotate(
        hkw.WorldAnnotation("Front", (0, 0, 0.5), background="magenta", color="white")
    )
    figure = (
        hkw.figure(layer)
        .camera("orthographic", eye=(1, 1, 5))
        .output(width=320, height=240, background="black")
    )
    beauty = np.asarray(hkw.snapshot(figure).image.convert("RGB"))
    normal = np.asarray(hkw.snapshot(figure, pass_name="normal").image.convert("RGB"))
    panel = (beauty[..., 0] > 150) & (beauty[..., 1] < 25) & (beauty[..., 2] > 150)
    geometry = np.any(normal != 0, axis=2)
    assert panel.any()
    assert not np.any(panel & geometry)
