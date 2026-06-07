"""End-to-end tests for the WebGL backend (``hakowan.render(..., backend='webgl')``)."""

from __future__ import annotations

import base64
import math
import re

import numpy as np
import pytest

pygltflib = pytest.importorskip("pygltflib")
import lagrange

import hakowan as hkw
from hakowan.setup.emitter import Envmap, Point as PointEmitter


def _decode_glb_from_html(html: str) -> bytes:
    m = re.search(
        r'GLB_DATA_URI\s*=\s*"data:model/gltf-binary;base64,([A-Za-z0-9+/=]+)"',
        html,
    )
    assert m, "no GLB data URI found in HTML"
    return base64.b64decode(m.group(1))


def _make_icosphere() -> lagrange.SurfaceMesh:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ],
        dtype=np.float64,
    )
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    tris = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [2, 11, 10],
            [4, 5, 11],
            [9, 1, 5],
            [8, 7, 1],
            [6, 10, 7],
            [4, 9, 5],
            [9, 8, 1],
            [8, 6, 7],
            [6, 2, 10],
            [2, 4, 11],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
        ],
        dtype=np.uint32,
    )
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(verts)
    mesh.add_triangles(tris)
    return mesh


class TestEndToEnd:
    def test_register_webgl_backend(self):
        assert "webgl" in hkw.list_backends()

    def test_simple_render(self, tmp_path):
        layer = (
            hkw.layer(_make_icosphere())
            .mark(hkw.mark.Surface)
            .channel(material=hkw.material.Diffuse(reflectance="red"))
        )
        out_path = tmp_path / "out.html"
        result = hkw.render(layer, filename=str(out_path), backend="webgl")
        assert out_path.exists()
        # render() may return a Path or None depending on backend.
        if result is not None:
            assert str(result) == str(out_path.resolve())

    def test_filename_suffix_normalised(self, tmp_path):
        layer = (
            hkw.layer(_make_icosphere())
            .mark(hkw.mark.Surface)
            .channel(material=hkw.material.Diffuse(reflectance="red"))
        )
        png_path = tmp_path / "out.png"
        hkw.render(layer, filename=str(png_path), backend="webgl")
        # Non-html suffix should produce an .html file alongside.
        assert (tmp_path / "out.html").exists()

    def test_default_output_is_html(self, tmp_path):
        layer = (
            hkw.layer(_make_icosphere())
            .mark(hkw.mark.Surface)
            .channel(material=hkw.material.Diffuse(reflectance="blue"))
        )
        out_path = tmp_path / "scene.html"
        hkw.render(layer, filename=str(out_path), backend="webgl")
        text = out_path.read_text(encoding="utf-8")
        assert text.lstrip().startswith("<!DOCTYPE html>")
        assert "three" in text  # CDN URL present
        assert "GLB_DATA_URI" in text

    def test_glb_round_trips_with_basic_scene(self, tmp_path):
        layer = (
            hkw.layer(_make_icosphere())
            .mark(hkw.mark.Surface)
            .channel(material=hkw.material.Diffuse(reflectance="purple"))
        )
        out_path = tmp_path / "scene.html"
        hkw.render(layer, filename=str(out_path), backend="webgl")
        html = out_path.read_text(encoding="utf-8")
        glb = _decode_glb_from_html(html)
        gltf = pygltflib.GLTF2().load_from_bytes(glb)
        assert len(gltf.meshes) == 1
        assert len(gltf.materials) == 1
        assert len(gltf.cameras) == 1

    def test_point_view_emits_mesh(self, tmp_path):
        mesh = lagrange.SurfaceMesh()
        mesh.add_vertices(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64))
        layer = (
            hkw.layer(mesh)
            .mark(hkw.mark.Point)
            .channel(
                material=hkw.material.Diffuse(reflectance="red"),
                size=0.1,
            )
        )
        out_path = tmp_path / "pts.html"
        hkw.render(layer, filename=str(out_path), backend="webgl")
        glb = _decode_glb_from_html(out_path.read_text())
        gltf = pygltflib.GLTF2().load_from_bytes(glb)
        # 3 points × 12 icosphere verts = 36 verts; the position accessor's
        # count should reflect this.
        positions_acc = gltf.accessors[gltf.meshes[0].primitives[0].attributes.POSITION]
        assert positions_acc.count == 36

    def test_curve_view_emits_lines_when_no_size(self, tmp_path):
        layer = (
            hkw.layer(_make_icosphere())
            .mark(hkw.mark.Curve)
            .channel(material=hkw.material.Diffuse(reflectance="white"))
        )
        out_path = tmp_path / "curve.html"
        hkw.render(layer, filename=str(out_path), backend="webgl")
        glb = _decode_glb_from_html(out_path.read_text())
        gltf = pygltflib.GLTF2().load_from_bytes(glb)
        # LINES mode = 1
        assert gltf.meshes[0].primitives[0].mode == 1

    def test_curve_view_emits_triangles_when_size_set(self, tmp_path):
        layer = (
            hkw.layer(_make_icosphere())
            .mark(hkw.mark.Curve)
            .channel(
                material=hkw.material.Diffuse(reflectance="white"),
                size=0.02,
            )
        )
        out_path = tmp_path / "tube.html"
        hkw.render(layer, filename=str(out_path), backend="webgl")
        glb = _decode_glb_from_html(out_path.read_text())
        gltf = pygltflib.GLTF2().load_from_bytes(glb)
        # TRIANGLES mode = 4
        assert gltf.meshes[0].primitives[0].mode == 4

    def test_point_light_registered(self, tmp_path):
        layer = (
            hkw.layer(_make_icosphere())
            .mark(hkw.mark.Surface)
            .channel(material=hkw.material.Diffuse(reflectance="red"))
        )
        cfg = hkw.config()
        cfg.emitters = [Envmap(), PointEmitter(position=[3, 4, 5], intensity=10.0)]
        out_path = tmp_path / "light.html"
        hkw.render(layer, cfg, filename=str(out_path), backend="webgl")
        glb = _decode_glb_from_html(out_path.read_text())
        gltf = pygltflib.GLTF2().load_from_bytes(glb)
        assert "KHR_lights_punctual" in (gltf.extensionsUsed or [])

    def test_envmap_descriptor_embedded(self, tmp_path):
        # Default config has Envmap → should embed HDR data URI.
        layer = (
            hkw.layer(_make_icosphere())
            .mark(hkw.mark.Surface)
            .channel(material=hkw.material.Diffuse(reflectance="red"))
        )
        out_path = tmp_path / "env.html"
        hkw.render(layer, filename=str(out_path), backend="webgl")
        html = out_path.read_text()
        assert "data:application/octet-stream;base64," in html
        assert '"format":' in html  # envmap descriptor JSON

    def test_pass_ui_always_present(self, tmp_path):
        layer = (
            hkw.layer(_make_icosphere())
            .mark(hkw.mark.Surface)
            .channel(material=hkw.material.Diffuse(reflectance="red"))
        )
        out_path = tmp_path / "passes.html"
        hkw.render(layer, filename=str(out_path), backend="webgl")
        html = out_path.read_text()
        # Pass UI shipped in every render — no Config opt-in.
        assert 'RENDER_PASSES = ["albedo", "depth", "normal"]' in html
        assert "function setPass(pass)" in html
        assert 'id="passes"' in html

    def test_checkerboard_uses_non_mipmap_nearest_sampler(self, tmp_path):
        mesh = lagrange.SurfaceMesh()
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64
        )
        facets = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        mesh.add_vertices(vertices)
        mesh.add_triangles(facets)
        mesh.create_attribute(
            "uv",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.UV,
            initial_values=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        )
        layer = (
            hkw.layer(mesh)
            .mark(hkw.mark.Surface)
            .channel(
                material=hkw.material.Diffuse(
                    reflectance=hkw.texture.Checkerboard(
                        uv="uv",
                        texture1=hkw.texture.Uniform(color="white"),
                        texture2=hkw.texture.Uniform(color="black"),
                        size=8,
                    )
                )
            )
        )
        out_path = tmp_path / "checker.html"
        hkw.render(layer, filename=str(out_path), backend="webgl")
        page_html = out_path.read_text(encoding="utf-8")
        glb = _decode_glb_from_html(page_html)
        gltf = pygltflib.GLTF2().load_from_bytes(glb)
        attrs_json = gltf.meshes[0].primitives[0].attributes.to_json()
        assert "TEXCOORD_0" in attrs_json
        tex_idx = gltf.materials[0].pbrMetallicRoughness.baseColorTexture.index
        sampler_idx = gltf.textures[tex_idx].sampler
        sampler = gltf.samplers[sampler_idx]
        assert sampler.magFilter == 9728
        assert sampler.minFilter == 9728
        assert gltf.materials[0].extras["hakowan"]["checkerboard"] is True
        base_tex = gltf.materials[0].pbrMetallicRoughness.baseColorTexture
        assert not base_tex.extensions
        assert "applyCrispCheckerTextures" in page_html

    def test_isocontour_exports_scalar_attribute_and_shader(self, tmp_path):
        mesh = _make_icosphere()
        h = np.asarray(mesh.vertices[:, 1], dtype=np.float64)
        mesh.create_attribute(
            "h",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=h,
        )
        layer = (
            hkw.layer(mesh)
            .mark(hkw.mark.Surface)
            .channel(
                material=hkw.material.Diffuse(
                    reflectance=hkw.texture.Isocontour(
                        data=hkw.attribute(name="h"),
                        num_contours=6,
                        ratio=0.3,
                        texture1=hkw.texture.Uniform(color="black"),
                        texture2=hkw.texture.Uniform(color="white"),
                    )
                )
            )
        )
        out_path = tmp_path / "iso.html"
        hkw.render(layer, filename=str(out_path), backend="webgl")
        html = out_path.read_text(encoding="utf-8")
        glb = _decode_glb_from_html(html)
        gltf = pygltflib.GLTF2().load_from_bytes(glb)
        attrs_json = gltf.meshes[0].primitives[0].attributes.to_json()
        assert "_scalar_0" in attrs_json
        iso = gltf.materials[0].extras["hakowan"]["isocontour"]
        assert iso["num_contours"] == 6
        # Viewer shader must reference the lowercased glTF attribute name.
        assert "attribute float _scalar_0" in html

    def test_multiview_layer_chain(self, tmp_path):
        sphere = _make_icosphere()
        a = (
            hkw.layer(sphere)
            .mark(hkw.mark.Surface)
            .channel(material=hkw.material.Diffuse(reflectance="red"))
            .translate(offset=(-1, 0, 0))
        )
        b = (
            hkw.layer(sphere)
            .mark(hkw.mark.Surface)
            .channel(material=hkw.material.Diffuse(reflectance="blue"))
            .translate(offset=(1, 0, 0))
        )
        out_path = tmp_path / "multi.html"
        hkw.render(a + b, filename=str(out_path), backend="webgl")
        glb = _decode_glb_from_html(out_path.read_text())
        gltf = pygltflib.GLTF2().load_from_bytes(glb)
        assert len(gltf.meshes) == 2
        assert len(gltf.materials) == 2
