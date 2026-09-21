from __future__ import annotations

import base64
import re

import lagrange
import numpy as np
import pytest

import hakowan as hkw


def _triangle():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    mesh.add_triangle(0, 1, 2)
    return mesh


def _glb_from_html(path):
    match = re.search(
        r'GLB_DATA_URI\s*=\s*"data:model/gltf-binary;base64,([A-Za-z0-9+/=]+)"',
        path.read_text(encoding="utf-8"),
    )
    assert match
    return base64.b64decode(match.group(1))

@pytest.fixture(scope="module")
def playwright_browser():
    playwright = pytest.importorskip("playwright.sync_api")
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            browser.close()
    except playwright.Error as exc:
        pytest.skip(f"Playwright Chromium is unavailable: {exc}")


def test_figure_fluent_scene_settings_are_immutable(tmp_path):
    layer = hkw.layer(_triangle())
    base = hkw.figure(layer)
    figure = (
        base.camera("perspective", eye=(2.0, 3.0, 4.0), fov=35.0)
        .light("point", position=(1.0, 2.0, 3.0), color="red", intensity=4.0)
        .light("directional", direction=(0.0, 0.0, -1.0), intensity=2.0)
        .environment(tmp_path / "studio.exr", scale=0.5, visible=True)
        .output(
            width=640,
            height=360,
            background="light",
            passes=("beauty", "depth", "normal"),
            sampler_seed=9,
        )
    )

    assert base.scene.camera is None
    assert figure.layer is layer
    assert isinstance(figure.scene.camera, hkw.PerspectiveCamera)
    assert len(figure.scene.lights) == 2
    assert figure.scene.output.width == 640

    config = figure.to_config()
    assert tuple(config.sensor.location) == (2.0, 3.0, 4.0)
    assert config.film.width == 640 and config.film.height == 360
    assert config.render_passes == {"depth", "normal"}
    assert config.sampler.seed == 9
    assert config.background == "light"
    assert config.environment_visible
    assert len(config.emitters) == 3  # environment + point + directional


def test_clear_lights_and_disabled_environment():
    figure = (
        hkw.figure(hkw.layer(_triangle()))
        .light("point")
        .clear_lights()
        .environment(enabled=False)
    )

    config = figure.to_config()

    assert figure.scene.lights == ()
    assert config.emitters == []
    assert not config.environment_visible


@pytest.mark.parametrize(
    "factory",
    [
        lambda: hkw.PerspectiveCamera(eye=(1.0, 2.0, 3.0)),
        lambda: hkw.OrthographicCamera(eye=(1.0, 2.0, 3.0)),
        lambda: hkw.ThinLensCamera(
            eye=(1.0, 2.0, 3.0), aperture_radius=0.2, focus_distance=4.0
        ),
    ],
)
def test_camera_variants_resolve_to_config(factory):
    figure = hkw.figure(hkw.layer(_triangle())).camera(factory())
    config = figure.to_config()
    assert tuple(config.sensor.location) == (1.0, 2.0, 3.0)


def test_scene_model_validation():
    with pytest.raises(ValueError, match="eye and target"):
        hkw.PerspectiveCamera(eye=(0, 0, 0), target=(0, 0, 0))
    with pytest.raises(ValueError, match="non-zero"):
        hkw.DirectionalLight(direction=(0, 0, 0))
    with pytest.raises(ValueError, match="positive"):
        hkw.OutputSettings(width=0)
    with pytest.raises(ValueError, match="duplicates"):
        hkw.OutputSettings(passes=("beauty", "beauty"))


def test_figure_render_uses_scene_settings_and_explicit_config_wins(tmp_path):
    figure = hkw.figure(hkw.layer(_triangle())).camera(
        "perspective", eye=(2.0, 3.0, 4.0)
    ).output(width=320, height=200, background="light")
    figure_path = tmp_path / "figure.html"
    hkw.render(figure, filename=figure_path, backend="webgl")
    figure_html = figure_path.read_text(encoding="utf-8")

    assert "const INITIAL_EYE    = [2.0, 3.0, 4.0]" in figure_html
    assert 'let bgMode = "light"' in figure_html

    config = hkw.config()
    config.sensor.location = [9.0, 8.0, 7.0]
    explicit_path = tmp_path / "explicit.html"
    hkw.render(figure, config=config, filename=explicit_path, backend="webgl")
    explicit_html = explicit_path.read_text(encoding="utf-8")

    assert "const INITIAL_EYE    = [9.0, 8.0, 7.0]" in explicit_html
    assert 'let bgMode = "dark"' in explicit_html


def test_figure_output_passes_drive_render_manifest(tmp_path):
    figure = hkw.figure(hkw.layer(_triangle())).output(
        width=80, height=60, passes=("beauty", "depth", "normal")
    )
    result = hkw.render(figure, filename=tmp_path / "viewer.html", backend="webgl")

    assert result.outputs == {
        "main": tmp_path / "viewer.html",
        "depth": "interactive",
        "normal": "interactive",
    }


def test_webgl_emits_point_and_directional_lights(tmp_path):
    pygltflib = pytest.importorskip("pygltflib")
    figure = (
        hkw.figure(hkw.layer(_triangle()))
        .environment(enabled=False)
        .light("point", position=(1, 2, 3), color="red", intensity=4)
        .light("directional", direction=(0, 0, -1), color="blue", intensity=2)
    )
    output = tmp_path / "lights.html"

    hkw.render(figure, filename=output, backend="webgl")
    gltf = pygltflib.GLTF2().load_from_bytes(_glb_from_html(output))
    lights = gltf.extensions["KHR_lights_punctual"]["lights"]

    assert [light["type"] for light in lights] == ["point", "directional"]
    assert lights[0]["color"] == pytest.approx([1.0, 0.0, 0.0])
    assert lights[0]["intensity"] == pytest.approx(4.0)
    assert lights[1]["color"] == pytest.approx([0.0, 0.0, 1.0])
    assert lights[1]["intensity"] == pytest.approx(2.0)

def test_webgl_uses_visible_declarative_environment(tmp_path):
    figure = hkw.figure(hkw.layer(_triangle())).environment(visible=True)
    output = tmp_path / "environment.html"

    hkw.render(figure, filename=output, backend="webgl")
    html = output.read_text(encoding="utf-8")

    assert '"background": true' in html


def test_figure_schema_1_1_round_trip_and_relative_environment(tmp_path):
    mesh = _triangle()
    figure = (
        hkw.figure(hkw.layer(mesh))
        .camera("thin_lens", eye=(2, 3, 4), aperture_radius=0.2)
        .light("directional", direction=(0, 0, -1), color="white", intensity=3)
        .environment("studio.exr", visible=True)
        .output(width=320, height=200, passes=("beauty", "depth"))
    )

    spec = hkw.to_spec(figure, data_ids={id(mesh): "mesh"})
    restored = hkw.from_spec(spec, data_resolver={"mesh": mesh}, base_dir=tmp_path)
    round_trip = hkw.to_spec(restored, data_ids={id(mesh): "mesh"})

    assert spec.version == "1.1"
    assert spec.scene.camera.kind == "thin_lens"
    assert isinstance(restored, hkw.Figure)
    assert restored.scene.environment.path == tmp_path / "studio.exr"
    assert round_trip.to_json(canonical=True) == spec.to_json(canonical=True)


def test_mitsuba_directional_light_translation():
    pytest.importorskip("mitsuba")
    from hakowan.backends.mitsuba.emitter import generate_emitter_config
    from hakowan.setup.emitter import Directional

    config = generate_emitter_config(
        Directional(direction=[0, 0, -1], color="red", intensity=3.0)
    )

    assert config["type"] == "directional"
    assert config["direction"] == [0, 0, -1]
    assert config["irradiance"]["type"] == "rgb"


def test_schema_1_0_remains_backward_compatible():
    payload = {
        "$schema": "https://hakowan.github.io/hakowan/schema/v1.json",
        "version": "1.0",
        "root": {"kind": "layer", "spec": {}},
    }

    restored = hkw.from_spec(payload)

    assert isinstance(restored, hkw.layer)


def test_scene_validation_reports_backend_degradation(tmp_path):
    figure = hkw.figure(hkw.layer(_triangle())).camera("thin_lens").output(
        passes=("beauty", "facet_id")
    )
    strict = hkw.validate(figure, backend="webgl", strict=True)
    permissive = hkw.validate(figure, backend="webgl", strict=False)

    assert {item.code for item in strict.errors} >= {
        "backend.camera.thin_lens",
        "backend.passes.unsupported",
    }
    assert permissive.valid

    missing_environment = figure.environment(tmp_path / "missing.exr")
    report = hkw.validate(missing_environment, backend="webgl")
    assert any(item.code == "environment.path.missing" for item in report.errors)


def test_figure_snapshot_uses_camera_and_output_defaults(playwright_browser):
    figure = hkw.figure(hkw.layer(_triangle())).camera(
        "perspective", eye=(0.0, 0.0, 4.0), fov=25.0
    ).output(width=48, height=32, background="light", passes=("beauty", "depth"))

    snapshot = hkw.snapshot(figure)
    observation = hkw.observe(figure, views=["front"])

    assert snapshot.view == "figure"
    assert snapshot.camera.eye == (0.0, 0.0, 4.0)
    assert snapshot.image.size == (48, 32)
    assert set(observation.snapshots) == {("front", "beauty"), ("front", "depth")}


def test_explicit_observation_options_override_figure(playwright_browser):
    figure = hkw.figure(hkw.layer(_triangle())).camera(
        "perspective", eye=(0.0, 0.0, 4.0)
    ).output(width=100, height=80, background="light")
    explicit = hkw.CameraState(
        eye=(4.0, 0.0, 0.0),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
    )

    snapshot = hkw.snapshot(
        figure,
        view="right",
        camera=explicit,
        resolution=(32, 24),
        background="dark",
    )

    assert snapshot.view == "right"
    assert snapshot.camera == explicit
    assert snapshot.image.size == (32, 24)
