from __future__ import annotations

import json
from pathlib import Path

import lagrange
import numpy as np
import pytest

import hakowan as hkw


def _triangle() -> lagrange.SurfaceMesh:
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    mesh.add_triangle(0, 1, 2)
    mesh.create_attribute(
        "temperature",
        element=lagrange.AttributeElement.Facet,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.array([7.0]),
    )
    return mesh


@pytest.fixture(scope="module")
def playwright_browser():
    playwright = pytest.importorskip("playwright.sync_api")
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            browser.close()
    except playwright.Error as exc:
        pytest.skip(f"Playwright Chromium is unavailable: {exc}")


def test_snapshot_captures_deterministic_beauty_png(playwright_browser, tmp_path):
    layer = hkw.layer(_triangle()).material("Diffuse", "red")
    output = tmp_path / "snapshot.png"

    first = hkw.snapshot(layer, resolution=(64, 64), filename=output)
    second = hkw.snapshot(layer, resolution=(64, 64))

    assert first.image.size == (64, 64)
    assert first.path == output
    assert output.is_file()
    assert np.array_equal(np.asarray(first.image), np.asarray(second.image))
    assert first.world_to_camera.shape == (4, 4)
    assert first.projection.shape == (4, 4)
    assert first.bounds.shape == (2, 3)


def test_observe_captures_raw_passes_and_pick(playwright_browser, tmp_path):
    layer = hkw.layer(_triangle()).material("Diffuse", "red")
    output = tmp_path / "observation"

    observation = hkw.observe(
        layer,
        views=["front"],
        passes=["beauty", "albedo", "depth", "normal", "element_id", "layer_id"],
        resolution=(48, 48),
        output_dir=output,
    )

    assert len(observation.snapshots) == 6
    assert observation.snapshot("front", "depth").data.shape == (48, 48)
    assert observation.snapshot("front", "normal").data.shape == (48, 48, 3)
    assert observation.snapshot("front", "element_id").data.dtype == np.uint32
    assert observation.snapshot("front", "layer_id").data.dtype == np.uint32
    assert observation.contact_sheet is not None
    assert (output / "contact_sheet.png").is_file()
    assert (output / "manifest.json").is_file()
    assert all(
        (output / f"front_{name}.png").is_file()
        for name in ("beauty", "albedo", "depth", "normal", "element_id", "layer_id")
    )

    assert all(
        (output / f"front_{name}.npy").is_file()
        for name in ("albedo", "depth", "normal", "element_id", "layer_id")
    )
    hit = observation.pick("front", (24, 24))
    assert hit is not None
    assert hit.layer_id == 0
    assert hit.element_id == 0
    assert hit.attributes["temperature"] == pytest.approx(7.0)
    assert hit.normal == pytest.approx((0.0, 0.0, 1.0), abs=1e-5)
    assert hit.world_position[2] == pytest.approx(0.0, abs=1e-5)

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["version"] == "1.0"
    assert manifest["scene"]["layers"][0]["facet_count"] == 1
    assert all(item["path"] for item in manifest["snapshots"])

    assert all(
        item["data_path"] for item in manifest["snapshots"] if item["pass"] != "beauty"
    )


def test_depth_and_normal_raw_data_without_id_sampling(playwright_browser):
    observation = hkw.observe(
        hkw.layer(_triangle()),
        views=["front"],
        passes=["depth", "normal"],
        resolution=(48, 48),
    )

    depth = observation.snapshot("front", "depth").data
    normals = observation.snapshot("front", "normal").data
    assert np.isfinite(depth).any()
    finite_normals = normals[np.all(np.isfinite(normals), axis=2)]
    assert len(finite_normals) > 0
    np.testing.assert_allclose(np.median(finite_normals, axis=0), [0, 0, 1], atol=0.03)


def test_observe_builds_canonical_multi_view_contact_sheet(playwright_browser):
    layer = hkw.layer(_triangle()).material("Diffuse", "orange")

    observation = hkw.observe(
        layer,
        views=["front", "right", "top", "isometric"],
        passes=["beauty"],
        resolution=(32, 32),
    )

    assert set(observation.snapshots) == {
        ("front", "beauty"),
        ("right", "beauty"),
        ("top", "beauty"),
        ("isometric", "beauty"),
    }
    assert observation.contact_sheet is not None
    assert observation.contact_sheet.width > 4 * 32
    assert observation.contact_sheet.height > 32
    cameras = [item.camera.eye for item in observation.snapshots.values()]
    assert len(set(cameras)) == 4


def test_pick_requires_geometry_passes(playwright_browser):
    observation = hkw.observe(
        hkw.layer(_triangle()),
        views=["front"],
        passes=["beauty"],
        resolution=(32, 32),
    )
    with pytest.raises(hkw.ObservationError, match="requires depth"):
        observation.pick("front", (10, 10))


def test_snapshot_rejects_unknown_view_and_pass():
    layer = hkw.layer(_triangle())
    with pytest.raises(ValueError, match="Unknown view"):
        hkw.snapshot(layer, view="diagonal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown pass"):
        hkw.snapshot(layer, pass_name="wireframe")  # type: ignore[arg-type]


def test_webgl_offline_bundle_uses_local_modules(monkeypatch, tmp_path):
    import hakowan.backends.webgl.render as webgl_render

    def fake_copy(_version: str, output: str | Path) -> Path:
        root = Path(output)
        for relative in (
            "build/three.module.js",
            "examples/jsm/controls/OrbitControls.js",
            "examples/jsm/loaders/GLTFLoader.js",
            "examples/jsm/utils/BufferGeometryUtils.js",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("// asset", encoding="utf-8")
        return root

    monkeypatch.setattr(webgl_render, "copy_three_assets", fake_copy)
    output = tmp_path / "viewer.html"
    hkw.render(hkw.layer(_triangle()), filename=output, backend="webgl", offline=True)

    html = output.read_text(encoding="utf-8")
    assert "./viewer_assets/build/three.module.js" in html
    assert "./viewer_assets/examples/jsm/" in html
    assert "unpkg.com" not in html
    assert (tmp_path / "viewer_assets" / "build" / "three.module.js").is_file()


def test_webgl_offline_bundle_loads_without_network(playwright_browser, tmp_path):
    from playwright.sync_api import sync_playwright

    output = tmp_path / "viewer.html"
    hkw.render(hkw.layer(_triangle()), filename=output, backend="webgl", offline=True)

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            headless=True, args=["--allow-file-access-from-files"]
        )
        page = browser.new_page()
        page.route("http://**", lambda route: route.abort())
        page.route("https://**", lambda route: route.abort())
        page.goto(output.resolve().as_uri(), wait_until="domcontentloaded")
        page.wait_for_function("window.__hakowan_loaded === true")
        assert page.evaluate("window.__hakowan.getCameraMode()") == "perspective"
        browser.close()


def test_viewer_exposes_capture_automation(tmp_path):
    output = tmp_path / "viewer.html"
    hkw.render(hkw.layer(_triangle()), filename=output, backend="webgl")
    html = output.read_text(encoding="utf-8")

    assert "setCameraState" in html
    assert "capturePng" in html
    assert "sampleGeometry" in html
    assert "captureIdPng" in html
