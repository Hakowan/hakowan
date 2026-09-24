from __future__ import annotations

import json

import lagrange
import numpy as np
import pytest
from PIL import Image

import hakowan as hkw
from hakowan.common.overlay import composite_overlay_file, composite_overlays
from hakowan.common.colormap.named_colormaps import get_colormap
from hakowan.compiler.overlay import CompiledAnnotation, CompiledLegend


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
        return np.asarray(view.data_frame.mesh.attribute(color_id).data)

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
    )

    observation = hkw.observe(
        layer, views=["front"], passes=["beauty"], resolution=(48, 48)
    )

    assert observation.manifest["legends"][0]["title"] == "temperature"
    assert observation.manifest["annotations"][0]["text"] == "note"
