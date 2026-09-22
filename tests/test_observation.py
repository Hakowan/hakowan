from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import sys
import urllib.error
import threading
from pathlib import Path

import lagrange
import numpy as np
import pytest
from PIL import Image

import hakowan.workflow.observation as observation_module
from hakowan.backends.webgl import assets as asset_module
from hakowan.workflow.validation import Diagnostic

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
    assert observation.snapshot("front", "albedo").data.dtype == np.uint8
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
    evidence = manifest["evidence"]["views"]["front"]
    assert 0.0 < evidence["occupancy"] < 1.0
    assert evidence["visible_bounds"] is not None
    assert evidence["layers"][0]["screen_fraction"] > 0.0
    assert evidence["layers"][0]["projection"]["clipped_fraction"] == 0.0
    assert evidence["layers"][0]["attributes"][0]["attribute"] == "temperature"
    assert evidence["contrast"]["luminance_contrast"] > 0.0
    assert isinstance(manifest["visual_diagnostics"], list)
    assert manifest["contact_sheet_metadata"]["mime_type"] == "image/png"
    assert len(manifest["contact_sheet_metadata"]["sha256"]) == 64
    assert all(item["mime_type"] == "image/png" for item in manifest["snapshots"])
    assert all(len(item["sha256"]) == 64 for item in manifest["snapshots"])


def test_observe_uses_figure_camera_and_reports_low_occupancy(playwright_browser):
    figure = hkw.figure(hkw.layer(_triangle())).camera(
        "perspective",
        eye=(0.5, 0.5, 100.0),
        target=(0.5, 0.5, 0.0),
        fov=35.0,
    )

    observation = hkw.observe(
        figure,
        passes=["beauty", "depth", "element_id", "layer_id"],
        resolution=(64, 64),
    )

    assert {view for view, _ in observation.snapshots} == {"figure"}
    assert observation.visual_evidence()["views"]["figure"]["occupancy"] < 0.02
    assert any(
        item.code in {"visual.frame_empty", "visual.occupancy_low"}
        for item in observation.visual_diagnostics()
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
    with pytest.raises(ValueError, match="Invalid view label"):
        hkw.observe(
            layer,
            views=["../escape"],
            cameras={"../escape": hkw.CameraState((0, 0, 4), (0, 0, 0), (0, 1, 0))},
        )


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
        assert page.evaluate("window.__hakowan.hasEnvironment()") is True
        browser.close()


def test_viewer_exposes_capture_automation(tmp_path):
    output = tmp_path / "viewer.html"
    hkw.render(hkw.layer(_triangle()), filename=output, backend="webgl")
    html = output.read_text(encoding="utf-8")

    assert "setCameraState" in html
    assert "capturePng" in html
    assert "sampleGeometry" in html
    assert "captureIdPng" in html


def test_result_dataclasses_are_json_safe():
    diagnostic = Diagnostic(
        code="example",
        severity="warning",
        path="layer",
        message="example warning",
    )
    camera = hkw.CameraState(
        eye=(1.0, 2.0, 3.0), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)
    )
    snapshot = hkw.Snapshot(
        image=Image.new("RGBA", (2, 2)),
        data=np.zeros((2, 2), dtype=np.float32),
        camera=camera,
        diagnostics=(diagnostic,),
    )
    summary = hkw.SceneSummary(
        bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        center=(0.0, 0.0, 0.0),
        radius=1.0,
        up_axis="y",
        layers=(observation_module.LayerSummary(0, "mesh", "surface", 3, 1),),
    )

    assert camera.to_dict()["eye"] == (1.0, 2.0, 3.0)
    assert summary.to_dict()["layers"][0]["name"] == "mesh"
    json.dumps(snapshot.to_manifest())


@pytest.mark.parametrize(
    ("up_axis", "view", "axis", "sign"),
    [
        ("y", "front", 2, 1),
        ("y", "back", 2, -1),
        ("y", "left", 0, -1),
        ("y", "right", 0, 1),
        ("y", "top", 1, 1),
        ("y", "bottom", 1, -1),
        ("z", "front", 1, -1),
        ("z", "back", 1, 1),
        ("z", "top", 2, 1),
        ("z", "bottom", 2, -1),
    ],
)
def test_camera_presets_follow_documented_axes(up_axis, view, axis, sign):
    summary = hkw.SceneSummary(
        bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        center=(0.0, 0.0, 0.0),
        radius=1.0,
        up_axis=up_axis,
        layers=(),
    )
    camera = observation_module._camera_for_view(view, summary, (320, 240))

    assert np.sign(camera.eye[axis]) == sign
    assert camera.near > 0
    assert camera.far > camera.near


def test_isometric_camera_and_portrait_framing():
    summary = hkw.SceneSummary(
        bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        center=(0.0, 0.0, 0.0),
        radius=1.0,
        up_axis="y",
        layers=(),
    )
    landscape = observation_module._camera_for_view("isometric", summary, (640, 320))
    portrait = observation_module._camera_for_view("isometric", summary, (320, 640))

    assert all(value > 0 for value in landscape.eye)
    assert np.linalg.norm(portrait.eye) > np.linalg.norm(landscape.eye)


def test_explicit_orthographic_camera_is_applied(playwright_browser):
    camera = hkw.CameraState(
        eye=(0.0, 0.0, 4.0),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        mode="orthographic",
        scale=2.0,
    )
    observation = hkw.observe(
        hkw.layer(_triangle()),
        views=["ortho"],
        passes=["depth", "element_id", "layer_id"],
        cameras={"ortho": camera},
        resolution=(40, 40),
    )
    result = observation.snapshot("ortho", "depth")

    assert result.camera == camera
    assert result.projection[3, 3] == pytest.approx(1.0)
    np.testing.assert_allclose(
        result.world_to_camera @ np.array([*camera.eye, 1.0]),
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-6,
    )
    ids = observation.snapshot("ortho", "layer_id").data
    assert ids is not None
    y, x = np.argwhere(ids != observation_module.BACKGROUND_ID)[0]
    hit = observation.pick("ortho", (int(x), int(y)))
    assert hit is not None
    assert hit.world_position[2] == pytest.approx(0.0, abs=1e-5)


def test_snapshot_writes_raw_sidecar(playwright_browser, tmp_path):
    output = tmp_path / "depth.png"
    result = hkw.snapshot(
        hkw.layer(_triangle()),
        view="front",
        pass_name="depth",
        resolution=(32, 32),
        filename=output,
    )

    assert result.data_path == tmp_path / "depth.npy"
    np.testing.assert_array_equal(
        np.load(result.data_path, allow_pickle=False), result.data
    )


def test_observe_camera_overrides_and_light_background(playwright_browser):
    custom = hkw.CameraState(
        eye=(0.0, 0.0, 4.0),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov=25.0,
    )
    observation = hkw.observe(
        hkw.layer(_triangle()),
        views=["front", "right"],
        passes=["beauty"],
        cameras={"right": custom},
        background="light",
        resolution=(32, 32),
    )

    assert observation.snapshot("right", "beauty").camera == custom
    assert observation.snapshot("front", "beauty").camera != custom
    corner = np.asarray(observation.snapshot("front", "beauty").image)[0, 0, :3]
    assert float(np.mean(corner)) > 100.0


def test_observation_save_after_capture_updates_manifest(playwright_browser, tmp_path):
    observation = hkw.observe(
        hkw.layer(_triangle()),
        views=["front"],
        passes=["depth"],
        resolution=(32, 32),
    )
    observation.save(tmp_path)

    assert observation.snapshot("front", "depth").path == tmp_path / "front_depth.png"
    assert (
        observation.snapshot("front", "depth").data_path == tmp_path / "front_depth.npy"
    )
    assert observation.manifest["contact_sheet"] == str(tmp_path / "contact_sheet.png")


def test_pick_background_and_bounds(playwright_browser):
    observation = hkw.observe(
        hkw.layer(_triangle()),
        views=["front"],
        passes=["depth", "element_id", "layer_id"],
        resolution=(40, 40),
    )

    assert observation.pick("front", (0, 0)) is None
    with pytest.raises(IndexError, match="outside"):
        observation.pick("front", (40, 0))


def test_single_point_scene_transform_is_finite():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertex([2.0, 3.0, 4.0])

    scene = hkw.compile(hkw.layer(mesh).mark("Point"))

    assert np.all(np.isfinite(scene[0].global_transform))
    transformed = scene[0].global_transform @ np.array([2.0, 3.0, 4.0, 1.0])
    np.testing.assert_allclose(transformed[:3], [0.0, 0.0, 0.0])


def test_point_pick_returns_vertex_attributes(playwright_browser):
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertex([0.0, 0.0, 0.0])
    mesh.create_attribute(
        "value",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.array([12.0]),
    )
    layer = hkw.layer(mesh).mark("Point").channel(size=0.2)
    observation = hkw.observe(
        layer,
        views=["front"],
        passes=["depth", "element_id", "layer_id"],
        resolution=(48, 48),
    )
    ids = observation.snapshot("front", "element_id").data
    y, x = np.argwhere(ids != int(observation_module.BACKGROUND_ID))[0]
    hit = observation.pick("front", (int(x), int(y)))

    assert hit is not None
    assert hit.element_id == 0
    assert hit.attributes["value"] == pytest.approx(12.0)


def test_layer_id_pass_distinguishes_composed_views(playwright_browser):
    mesh = _triangle()
    left = hkw.layer(mesh, name="left").material("Diffuse", "red")
    right = hkw.layer(mesh, name="right").material("Diffuse", "blue")
    observation = hkw.observe(
        left | right,
        views=["front"],
        passes=["layer_id"],
        resolution=(96, 48),
    )
    values = set(np.unique(observation.snapshot("front", "layer_id").data).tolist())
    values.discard(int(observation_module.BACKGROUND_ID))

    assert values == {0, 1}
    assert [layer.name for layer in observation.scene_summary.layers] == [
        "left",
        "right",
    ]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"views": []}, "At least one view"),
        ({"passes": []}, "At least one pass"),
        ({"resolution": (0, 32)}, "Resolution must be positive"),
    ],
)
def test_observe_rejects_invalid_capture_requests(kwargs, message):
    with pytest.raises(ValueError, match=message):
        hkw.observe(hkw.layer(_triangle()), **kwargs)


def test_snapshot_rejects_unsupported_backend():
    with pytest.raises(NotImplementedError, match="webgl"):
        hkw.snapshot(hkw.layer(_triangle()), backend="mitsuba")


def test_backend_capabilities_advertise_observation():
    features = hkw.backend_capabilities("webgl").features
    assert {
        "headless_snapshot",
        "multi_view_observation",
        "element_id_observation",
        "layer_id_observation",
    } <= features


def test_missing_playwright_has_actionable_error(monkeypatch):
    original_import = __import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("playwright"):
            raise ImportError("blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    with pytest.raises(hkw.ObservationError, match=r"hakowan\[observe\]"):
        observation_module._require_playwright()


def test_asset_cache_download_reuse_and_safe_copy(monkeypatch, tmp_path):
    monkeypatch.setenv("HAKOWAN_CACHE_DIR", str(tmp_path / "cache"))
    payload = b"// module"
    manifest = {
        "test": {
            "build/three.module.js": hashlib.sha256(payload).hexdigest(),
            "examples/jsm/loaders/EXRLoader.js": hashlib.sha256(payload).hexdigest(),
        }
    }
    monkeypatch.setattr(asset_module, "_ASSET_MANIFEST", manifest)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return payload

    calls = []

    def urlopen(url, timeout):
        calls.append((url, timeout))
        return Response()

    monkeypatch.setattr(asset_module.urllib.request, "urlopen", urlopen)
    cache = asset_module.ensure_three_assets("test")
    assert len(calls) == 2
    assert all((cache / relative).is_file() for relative in manifest["test"])

    calls.clear()
    assert asset_module.ensure_three_assets("test") == cache
    assert calls == []

    destination = tmp_path / "bundle"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    asset_module.copy_three_assets("test", destination)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_offline_asset_manifest_includes_environment_loaders():
    assets = asset_module._ASSET_MANIFEST["0.170.0"]
    assert "examples/jsm/loaders/EXRLoader.js" in assets
    assert "examples/jsm/loaders/RGBELoader.js" in assets
    assert "examples/jsm/libs/fflate.module.js" in assets


def test_asset_integrity_failure_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv("HAKOWAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        asset_module,
        "_ASSET_MANIFEST",
        {"test": {"build/three.module.js": "0" * 64}},
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b"tampered"

    monkeypatch.setattr(
        asset_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    with pytest.raises(asset_module.WebGLAssetError, match="integrity"):
        asset_module.ensure_three_assets("test")


def test_asset_download_failure_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv("HAKOWAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        asset_module,
        "_ASSET_MANIFEST",
        {"test": {"build/three.module.js": "0" * 64}},
    )
    monkeypatch.setattr(
        asset_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError("offline")
        ),
    )
    with pytest.raises(asset_module.WebGLAssetError, match="network access"):
        asset_module.ensure_three_assets("test")


def test_cli_missing_normal_is_actionable(monkeypatch, tmp_path):
    mesh_path = tmp_path / "mesh.ply"
    lagrange.io.save_mesh(mesh_path, _triangle())
    main_module = importlib.import_module("hakowan.__main__")
    monkeypatch.setattr(
        sys,
        "argv",
        ["hakowan", str(mesh_path), "--normal", "missing"],
    )

    with pytest.raises(SystemExit, match="Normal attribute 'missing' not found"):
        main_module.main()


def test_cli_webgl_turntable_uses_snapshot(monkeypatch, tmp_path):
    mesh_path = tmp_path / "mesh.ply"
    lagrange.io.save_mesh(mesh_path, _triangle())
    output = tmp_path / "turntable.png"
    calls = []

    def fake_snapshot(_layer, *, filename, resolution, pass_name, **_kwargs):
        calls.append((Path(filename), resolution, pass_name))
        Image.new("RGB", resolution, "white").save(filename)

    main_module = importlib.import_module("hakowan.__main__")
    monkeypatch.setattr(hkw, "snapshot", fake_snapshot)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hakowan",
            str(mesh_path),
            "--backend",
            "webgl",
            "--turn-table",
            "2",
            "--resolution",
            "24",
            "16",
            "--depth",
            "--output",
            str(output),
            "--no-open",
        ],
    )

    main_module.main()

    assert len(calls) == 2
    assert all(
        resolution == (24, 16) and pass_name == "depth"
        for _, resolution, pass_name in calls
    )
    assert output.with_suffix(".gif").is_file()


def test_snapshot_propagates_diagnostics_without_mutating_config(playwright_browser):
    config = hkw.config()
    original_size = (config.film.width, config.film.height)
    layer = hkw.layer(_triangle()).mark("Surface").channel(size=0.1)

    result = hkw.snapshot(layer, resolution=(32, 24), config=config)

    assert any(item.code == "channel.mark_incompatible" for item in result.diagnostics)
    assert (config.film.width, config.film.height) == original_size


def test_active_event_loop_uses_worker_thread(monkeypatch):
    main_thread = threading.get_ident()

    def fake_capture(*_args, **_kwargs):
        return threading.get_ident()

    monkeypatch.setattr(observation_module, "_capture_sync", fake_capture)

    async def invoke():
        return observation_module._capture()

    worker_thread = asyncio.run(invoke())
    assert worker_thread != main_thread
