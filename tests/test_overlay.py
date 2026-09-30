from __future__ import annotations

import json
from dataclasses import replace

import lagrange
import numpy as np
import pytest
from PIL import Image

import hakowan as hkw
from hakowan.common.overlay import composite_overlay_file, composite_overlays
from hakowan.common.colormap.named_colormaps import get_colormap
from hakowan.compiler.overlay import (
    CompiledAnnotation,
    CompiledLegend,
    CompiledWorldAnnotation,
)
from hakowan.setup.sensor import Orthographic, Perspective


def _scalar_mesh():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    mesh.add_triangle(0, 1, 2)
    mesh.create_attribute(
        "temperature",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.array([10.0, 20.0, 30.0]),
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


def test_compile_collects_continuous_legend_and_annotation():
    mesh = _scalar_mesh()
    layer = (
        hkw.layer(mesh)
        .material(
            "Diffuse",
            hkw.texture.ScalarField(
                hkw.attribute("temperature", unit="°C"),
                colormap="viridis",
                legend=hkw.Legend(title="Temperature", ticks=3, format=".1f"),
            ),
        )
        .annotate(
            "heated surface",
            position=(0.5, 0.05),
            anchor="center",
            background="black",
        )
    )

    scene = hkw.compile(layer)

    assert len(scene.legends) == 1
    legend = scene.legends[0]
    assert legend.title == "Temperature"
    assert legend.units == "°C"
    assert legend.domain == (10.0, 30.0)
    assert legend.values == (10.0, 20.0, 30.0)
    assert legend.labels == ("10.0", "20.0", "30.0")
    assert len(legend.colors) == 16
    assert len(scene.annotations) == 1
    assert scene.annotations[0].text == "heated surface"


def test_legend_domain_tracks_user_scale_pipeline():
    mesh = _scalar_mesh()
    field = hkw.attribute("temperature", unit="K", scale=hkw.scale.Uniform(factor=2.0))

    scene = hkw.compile(
        hkw.layer(mesh).material("Diffuse", hkw.texture.ScalarField(field))
    )

    assert scene.legends[0].domain == (20.0, 60.0)
    assert scene.legends[0].scale == ("uniform",)


def test_legend_domain_uses_declared_normalize_output_range():
    mesh = _scalar_mesh()
    field = hkw.attribute(
        "temperature",
        scale=hkw.scale.Normalize(
            domain_min=0.0,
            domain_max=100.0,
            range_min=0.0,
            range_max=1.0,
        ),
    )

    scene = hkw.compile(
        hkw.layer(mesh).material("Diffuse", hkw.texture.ScalarField(field))
    )

    assert scene.legends[0].domain == (0.0, 1.0)
    assert scene.legends[0].values == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_normalize_domain_controls_colormap_coordinates():
    mesh = _scalar_mesh()
    colormap = get_colormap("viridis")
    assert colormap is not None

    def compile_colors(domain_max):
        field = hkw.attribute(
            "temperature",
            scale=hkw.scale.Normalize(
                domain_min=0.0,
                domain_max=domain_max,
                range_min=0.0,
                range_max=1.0,
            ),
        )
        view = hkw.compile(
            hkw.layer(mesh).material("Diffuse", hkw.texture.ScalarField(field))
        )[0]
        color_id = view.data_frame.mesh.get_matching_attribute_id(
            usage=lagrange.AttributeUsage.Color
        )
        return np.asarray(view.data_frame.mesh.attribute(color_id).data).copy()

    wide = compile_colors(100.0)
    narrow = compile_colors(30.0)

    np.testing.assert_allclose(wide[0], colormap(0.1).data)
    np.testing.assert_allclose(narrow[0], colormap(1.0 / 3.0).data)
    assert not np.allclose(wide, narrow)


def test_legend_colors_follow_requested_colormap_range():
    mesh = _scalar_mesh()
    scene = hkw.compile(
        hkw.layer(mesh).material(
            "Diffuse",
            hkw.texture.ScalarField(
                "temperature", colormap="viridis", range=(0.25, 0.75)
            ),
        )
    )
    colormap = get_colormap("viridis")
    assert colormap is not None
    np.testing.assert_allclose(scene.legends[0].colors[0], colormap(0.25).data[:3])
    np.testing.assert_allclose(scene.legends[0].colors[-1], colormap(0.75).data[:3])


def test_unsupported_data_driven_hair_color_has_no_legend():
    scene = hkw.compile(
        hkw.layer(_scalar_mesh()).material(
            "Hair", color=hkw.texture.ScalarField("temperature")
        )
    )

    assert scene.legends == []


def test_compile_collects_categorical_labels_and_deduplicates_overlay():
    mesh = _scalar_mesh()
    legend = hkw.Legend(category_labels={"10": "cold", "20": "compact", "20.0": "warm"})
    base = (
        hkw.layer(mesh)
        .material(
            "Diffuse",
            hkw.texture.ScalarField(
                "temperature", colormap="set1", categories=True, legend=legend
            ),
        )
        .annotate("same")
    )

    scene = hkw.compile(base + base)

    assert len(scene.legends) == 1
    assert scene.legends[0].categories
    assert scene.legends[0].values == (10.0, 20.0, 30.0)
    assert scene.legends[0].labels == ("cold", "warm", "30")
    assert len(scene.legends[0].colors) == 3
    assert len(scene.annotations) == 1


@pytest.mark.parametrize(
    "texture",
    [
        hkw.texture.ScalarField("temperature", legend=False),
        hkw.texture.ScalarField("temperature", colormap="identity"),
    ],
)
def test_legend_can_be_suppressed(texture):
    scene = hkw.compile(hkw.layer(_scalar_mesh()).material("Diffuse", texture))
    assert scene.legends == []


def test_annotation_validation_and_shorthand():
    mesh = _scalar_mesh()
    layer = hkw.layer(mesh).annotate(
        "center", position=(0.5, 0.5), anchor="center", color="red"
    )

    assert layer._spec.annotations == [
        hkw.Annotation("center", position=(0.5, 0.5), anchor="center", color="red")
    ]
    with pytest.raises(ValueError, match="position"):
        hkw.Annotation("outside", position=(2.0, 0.0))
    with pytest.raises(ValueError, match="exactly two"):
        hkw.Annotation("short", position=(0.5,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly two"):
        hkw.Annotation("long", position=(0.2, 0.3, 0.4))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ticks"):
        hkw.Legend(ticks=1)


def test_raster_overlay_compositor_adds_panel_and_annotation(tmp_path):
    legend = CompiledLegend(
        title="Temperature",
        units="°C",
        categories=False,
        domain=(0.0, 1.0),
        values=(0.0, 0.5, 1.0),
        labels=("0", "0.5", "1"),
        colors=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        position="right",
        width=120,
        scale=("normalize",),
    )
    annotation = CompiledAnnotation(
        text="sample",
        position=(0.5, 0.1),
        color=(1.0, 1.0, 1.0),
        font_size=14,
        anchor="center",
        background=(0.0, 0.0, 0.0),
        padding=3,
    )
    source = Image.new("RGB", (200, 80), "navy")

    result = composite_overlays(source, [legend], [annotation])

    assert result.size == (200, 80)
    assert np.asarray(result)[:, 80:].std() > 0
    path = tmp_path / "overlay.png"
    source.save(path)
    assert composite_overlay_file(path, [legend], [annotation])
    assert Image.open(path).size == (200, 80)
    pcx_path = tmp_path / "overlay.pcx"
    source.save(pcx_path)
    assert composite_overlay_file(pcx_path, [legend], [annotation])
    with Image.open(pcx_path) as encoded:
        assert encoded.mode == "RGB"
        assert encoded.size == (200, 80)
    assert not composite_overlay_file(tmp_path / "image.exr", [legend], [])


def _world_callout(position=(0.0, 0.0, 0.0), text="anchor"):
    return CompiledWorldAnnotation(
        text=text,
        position=position,
        layer=0,
        cell=None,
        color=(1.0, 0.0, 0.0),
        font_size=14,
        background=(0.0, 0.0, 0.0),
        padding=3,
    )


def test_world_callouts_project_with_sensor_and_avoid_anchor_overlap(tmp_path):
    source = Image.new("RGB", (240, 160), "white")
    camera = Perspective(
        location=(0, 0, 5), target=(0, 0, 0), up=(0, 1, 0),
        fov=90, fov_axis="y",
    )
    annotations = [_world_callout(), _world_callout(text="second")]
    result = composite_overlays(
        source, [], [], world_annotations=annotations, sensor=camera
    )
    assert result.getpixel((120, 80)) == (255, 0, 0, 255)
    # Distinct labels fit in different locations, neither occluding the anchor.
    dark = np.all(np.asarray(result)[:, :, :3] < 100, axis=2)
    assert dark.sum() > 100
    assert not dark[80, 120]
    assert np.any(dark[:70, 130:])
    assert np.any(dark[80:, 130:])

    pale = composite_overlays(
        source, [], [], world_annotations=[
            replace(annotations[0], color=(1.0, 1.0, 1.0), background=None)
        ], sensor=camera,
    )
    assert np.any(np.asarray(pale)[:, :, :3] < 50)  # contrast stroke on white

    path = tmp_path / "world.png"
    source.save(path)
    assert composite_overlay_file(
        path, [], [], world_annotations=annotations[:1], sensor=camera
    )
    with Image.open(path) as encoded:
        assert encoded.getpixel((120, 80))[:3] == (255, 0, 0)
    assert not composite_overlay_file(
        tmp_path / "world.exr", [], [],
        world_annotations=annotations[:1], sensor=camera,
    )


def test_world_callouts_follow_camera_and_reject_clipped_anchors():
    source = Image.new("RGB", (240, 160), "white")
    camera = Orthographic(
        location=(0, 0, 5), target=(0, 0, 0), up=(0, 1, 0), scale=2,
    )
    annotations = [
        _world_callout((0.5, 0, 0)),
        _world_callout((0, 0.25, 0), "height"),
        _world_callout((0, 0, 6), "behind"),
        _world_callout((8, 0, 0), "outside"),
    ]
    result = composite_overlays(
        source, [], [], world_annotations=annotations, sensor=camera
    )
    assert result.getpixel((180, 80)) == (255, 0, 0, 255)
    # Mitsuba's orthographic scale spans the horizontal film, even at 240x160.
    assert result.getpixel((120, 50)) == (255, 0, 0, 255)
    facing_right = replace(camera, target=(0.5, 0, 0))
    turned = composite_overlays(
        source, [], [], world_annotations=annotations[:1], sensor=facing_right
    )
    assert turned.getpixel((120, 80)) == (255, 0, 0, 255)
    assert turned.getpixel((180, 80)) != result.getpixel((180, 80))
    hidden = composite_overlays(
        source, [], [], world_annotations=annotations[2:], sensor=camera
    )
    assert hidden.tobytes() == source.convert("RGBA").tobytes()


def test_world_callouts_use_capture_camera_matrices():
    source = Image.new("RGB", (240, 160), "white")
    view = np.eye(4)
    view[2, 3] = -5
    projection = np.eye(4)
    projection[0, 0] = 0.5
    projection[2, 2] = -0.1
    camera = Orthographic(location=(0, 0, 5), target=(0, 0, 0), scale=2)
    result = composite_overlays(
        source, [], [],
        world_annotations=[_world_callout((1, 0, 0))],
        sensor=camera, world_to_camera=view, projection=projection,
    )
    assert result.getpixel((180, 80)) == (255, 0, 0, 255)
    assert result.getpixel((120, 80)) == (255, 255, 255, 255)
    view[0, 3] = -1
    shifted = composite_overlays(
        source, [], [], world_annotations=[_world_callout((1, 0, 0))],
        world_to_camera=view, projection=projection,
    )
    assert shifted.getpixel((120, 80)) == (255, 0, 0, 255)


def test_raster_overlay_preserves_transparent_render_background(tmp_path):
    legend = CompiledLegend(
        title="Value",
        units=None,
        categories=False,
        domain=(0.0, 1.0),
        values=(0.0, 1.0),
        labels=("0", "1"),
        colors=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        position="right",
        width=80,
        scale=(),
    )
    source = Image.new("RGBA", (200, 120), (255, 0, 0, 0))
    source.putpixel((5, 5), (255, 0, 0, 255))

    result = composite_overlays(source, [legend], [])

    assert result.size == source.size
    assert result.getpixel((0, 0))[3] == 0
    assert 0 < result.getpixel((113, 8))[3] < 148
    assert result.getpixel((5, 5)) == (255, 0, 0, 255)
    assert result.getpixel((112, 8))[3] == 0
    assert result.getpixel((120, 16))[3] == 148

    path = tmp_path / "transparent.png"
    source.save(path)
    assert composite_overlay_file(path, [legend], [])
    with Image.open(path) as encoded:
        assert encoded.mode == "RGBA"
        assert encoded.size == source.size
        assert encoded.getpixel((0, 0))[3] == 0


def test_webgl_embeds_semantic_overlay_metadata(tmp_path):
    layer = (
        hkw.layer(_scalar_mesh())
        .material(
            "Diffuse",
            hkw.texture.ScalarField(
                hkw.attribute("temperature", unit="K"),
                legend=hkw.Legend(title="Heat"),
            ),
        )
        .annotate("simulation A", position=(0.5, 0.05), anchor="center")
    )
    output = tmp_path / "viewer.html"

    hkw.render(layer, filename=output, backend="webgl")
    html = output.read_text(encoding="utf-8")

    assert "const LEGENDS" in html
    assert "const ANNOTATIONS" in html
    assert '"title": "Heat"' in html
    assert '"units": "K"' in html
    assert '"text": "simulation A"' in html
    assert "buildSemanticOverlays()" in html
    assert "{{LEGENDS_JSON}}" not in html


def test_webgl_escapes_html_and_inline_script_payloads(tmp_path):
    script_payload = "</script><script>globalThis.hakowanInjected=true</script>"
    title_payload = "</title><script>globalThis.hakowanTitleInjected=true</script>"
    layer = hkw.layer(_scalar_mesh()).annotate(script_payload)
    output = tmp_path / "viewer.html"

    hkw.render(layer, filename=output, backend="webgl", title=title_payload)
    html = output.read_text(encoding="utf-8")

    assert script_payload not in html
    assert "\\u003c/script\\u003e" in html
    assert title_payload not in html
    assert "&lt;/title&gt;" in html


def test_overlay_schema_round_trip():
    mesh = _scalar_mesh()
    layer = (
        hkw.layer(mesh)
        .material(
            "Diffuse",
            hkw.texture.ScalarField(
                hkw.attribute("temperature", unit="Pa"),
                legend=hkw.Legend(title="Pressure", position="left"),
            ),
        )
        .annotate("peak", position=(0.8, 0.2), anchor="right")
    )

    spec = hkw.to_spec(layer, data_ids={id(mesh): "mesh"})
    restored = hkw.from_spec(spec, data_resolver={"mesh": mesh})
    round_trip = hkw.to_spec(restored, data_ids={id(mesh): "mesh"})

    assert round_trip.to_json(canonical=True) == spec.to_json(canonical=True)
    payload = spec.to_dict()
    json.dumps(payload)
    assert "annotations" in payload["root"]["spec"]


def test_observation_manifest_contains_semantic_metadata(playwright_browser):
    layer = (
        hkw.layer(_scalar_mesh())
        .material("Diffuse", hkw.texture.ScalarField("temperature"))
        .annotate("note")
        .annotate(hkw.WorldAnnotation("world", (0.0, 0.0, 0.0)))
    )

    observation = hkw.observe(
        layer, views=["front"], passes=["beauty"], resolution=(48, 48)
    )

    assert observation.manifest["legends"][0]["title"] == "temperature"
    assert observation.manifest["annotations"][0]["text"] == "note"
    assert observation.manifest["world_annotations"][0]["text"] == "world"
    assert observation.manifest["world_annotations"][0]["layer"] == 0

def test_observation_projects_world_callout_only_in_beauty(playwright_browser):
    layer = hkw.layer(_scalar_mesh()).annotate(
        hkw.WorldAnnotation("vertex", (0.25, 0.25, 0.0), color="red", background="black")
    )
    observation = hkw.observe(
        layer, views=["front"], passes=["beauty", "normal"],
        resolution=(240, 180),
    )
    beauty = observation.snapshot("front", "beauty")
    normal = observation.snapshot("front", "normal")
    position = observation.manifest["world_annotations"][0]["position"]
    clip = beauty.projection @ beauty.world_to_camera @ np.array([*position, 1.0])
    px = round((clip[0] / clip[3] + 1) * beauty.image.width / 2)
    py = round((1 - clip[1] / clip[3]) * beauty.image.height / 2)

    assert beauty.image.getpixel((px, py))[:3] == (255, 0, 0)
    assert normal.image.getpixel((px, py))[:3] != (255, 0, 0)
    assert observation.manifest["world_annotations"][0]["text"] == "vertex"
