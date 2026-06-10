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
        # render() returns a RenderResult; .path is the main output as given.
        assert isinstance(result, hkw.RenderResult)
        assert result.path == out_path
        assert result.backend == "webgl"

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

    def test_point_view_is_instanced(self, tmp_path):
        # A uniform-size point cloud is translation·scale per point, so it
        # GPU-instances one prototype sphere instead of baking 3 copies.
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
        assert "EXT_mesh_gpu_instancing" in (gltf.extensionsUsed or [])
        # One prototype icosphere (12 verts), not 3 × 12 baked.
        assert len(gltf.meshes) == 1
        positions_acc = gltf.accessors[gltf.meshes[0].primitives[0].attributes.POSITION]
        assert positions_acc.count == 12
        inst_nodes = [
            n
            for n in gltf.nodes
            if n.extensions and "EXT_mesh_gpu_instancing" in n.extensions
        ]
        assert len(inst_nodes) == 1
        attrs = inst_nodes[0].extensions["EXT_mesh_gpu_instancing"]["attributes"]
        # 3 points → 3 instances, each with translation/rotation/scale.
        assert gltf.accessors[attrs["TRANSLATION"]].count == 3
        assert gltf.accessors[attrs["ROTATION"]].count == 3
        assert gltf.accessors[attrs["SCALE"]].count == 3

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

    def _vector_field_mesh(self):
        mesh = _make_icosphere()
        v = np.asarray(mesh.vertices, dtype=np.float64)
        mesh.create_attribute(
            "vec",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Vector,
            initial_values=(v * 0.3),
        )
        mesh.create_attribute(
            "speed",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=np.linalg.norm(v, axis=1),
        )
        return mesh

    def test_constant_radius_curve_is_instanced(self, tmp_path):
        # A constant-size vector-field curve must collapse to one prototype
        # cylinder + per-segment instance transforms, not a baked tube soup.
        mesh = self._vector_field_mesh()
        layer = (
            hkw.layer(mesh)
            .mark(hkw.mark.Curve)
            .channel(
                vector_field=hkw.channel.VectorField(data="vec"),
                material=hkw.material.Diffuse(
                    reflectance=hkw.texture.ScalarField(data="speed")
                ),
                size=0.01,
            )
        )
        out_path = tmp_path / "stream.html"
        hkw.render(layer, filename=str(out_path), backend="webgl")
        glb = _decode_glb_from_html(out_path.read_text())
        gltf = pygltflib.GLTF2().load_from_bytes(glb)
        assert "EXT_mesh_gpu_instancing" in (gltf.extensionsUsed or [])
        # Exactly one prototype mesh; its vertex count is the cylinder, not the
        # 12 segments × 16 verts a baked tube would carry.
        assert len(gltf.meshes) == 1
        pos = gltf.accessors[gltf.meshes[0].primitives[0].attributes.POSITION]
        # One capped 8-sided unit cylinder prototype (lagrange.primitive), far
        # smaller than the 12 segments a baked tube would carry.
        assert pos.count < 100
        inst_nodes = [
            n
            for n in gltf.nodes
            if n.extensions and "EXT_mesh_gpu_instancing" in n.extensions
        ]
        assert len(inst_nodes) == 1
        attrs = inst_nodes[0].extensions["EXT_mesh_gpu_instancing"]["attributes"]
        # 12 vertices → 12 instances, and a per-instance colour from the field.
        assert gltf.accessors[attrs["TRANSLATION"]].count == 12
        assert "_COLOR_0" in attrs

    def test_arrow_curve_is_instanced(self, tmp_path):
        # The arrow glyph (shaft + tapered cone) is a fixed prototype scaled
        # (r, length, r) per vector, so the whole field GPU-instances one mesh.
        mesh = self._vector_field_mesh()
        layer = (
            hkw.layer(mesh)
            .mark(hkw.mark.Curve)
            .channel(
                vector_field=hkw.channel.VectorField(data="vec", end_type="arrow"),
                material=hkw.material.Diffuse(reflectance="white"),
                size=0.01,
            )
        )
        out_path = tmp_path / "arrows.html"
        hkw.render(layer, filename=str(out_path), backend="webgl")
        glb = _decode_glb_from_html(out_path.read_text())
        gltf = pygltflib.GLTF2().load_from_bytes(glb)
        assert "EXT_mesh_gpu_instancing" in (gltf.extensionsUsed or [])
        # One prototype arrow mesh: a capped shaft cylinder + cone head built
        # from lagrange.primitive — a fixed small glyph, not baked per vector.
        assert len(gltf.meshes) == 1
        pos = gltf.accessors[gltf.meshes[0].primitives[0].attributes.POSITION]
        assert pos.count < 150
        inst_nodes = [
            n
            for n in gltf.nodes
            if n.extensions and "EXT_mesh_gpu_instancing" in n.extensions
        ]
        assert len(inst_nodes) == 1
        attrs = inst_nodes[0].extensions["EXT_mesh_gpu_instancing"]["attributes"]
        # 12 vectors → 12 instances.
        assert gltf.accessors[attrs["TRANSLATION"]].count == 12

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

    def test_juxtapose_renders_side_by_side(self, tmp_path):
        sphere = _make_icosphere()
        # No manual `.translate`: `|` lays the two layers out side by side.
        a = (
            hkw.layer(sphere)
            .mark(hkw.mark.Surface)
            .channel(material=hkw.material.Diffuse(reflectance="red"))
        )
        b = (
            hkw.layer(sphere)
            .mark(hkw.mark.Surface)
            .channel(material=hkw.material.Diffuse(reflectance="blue"))
        )
        out_path = tmp_path / "juxtapose.html"
        hkw.render(a | b, filename=str(out_path), backend="webgl")
        glb = _decode_glb_from_html(out_path.read_text())
        gltf = pygltflib.GLTF2().load_from_bytes(glb)
        assert len(gltf.meshes) == 2
        assert len(gltf.materials) == 2
        # The layout transform rides on each mesh node's column-major matrix; the
        # X translation (index 12) must differ between the two side-by-side cells.
        x_translations = [
            node.matrix[12] for node in gltf.nodes if node.mesh is not None and node.matrix
        ]
        assert len(x_translations) == 2
        assert abs(x_translations[0] - x_translations[1]) > 0.5

    def test_juxtapose_tags_nodes_with_distinct_cells(self, tmp_path):
        # Each cell's nodes carry a `hakowan_cell` extra so the interactive viewer
        # can rotate each comparison cell about its own centre.
        sphere = _make_icosphere()
        a = hkw.layer(sphere).mark(hkw.mark.Surface)
        b = hkw.layer(sphere).mark(hkw.mark.Surface)
        out_path = tmp_path / "cells.html"
        hkw.render(a | b, filename=str(out_path), backend="webgl")
        gltf = pygltflib.GLTF2().load_from_bytes(_decode_glb_from_html(out_path.read_text()))
        cells = [
            n.extras["hakowan_cell"]
            for n in gltf.nodes
            if n.mesh is not None and n.extras and "hakowan_cell" in n.extras
        ]
        assert len(cells) == 2
        assert cells[0] != cells[1]

    def test_overlay_nodes_have_no_cell_tag(self, tmp_path):
        # Without `|`, nodes are untagged (the viewer treats them as one group).
        sphere = _make_icosphere()
        a = hkw.layer(sphere).mark(hkw.mark.Surface)
        b = hkw.layer(sphere).mark(hkw.mark.Surface)
        out_path = tmp_path / "overlay.html"
        hkw.render(a + b, filename=str(out_path), backend="webgl")
        gltf = pygltflib.GLTF2().load_from_bytes(_decode_glb_from_html(out_path.read_text()))
        for n in gltf.nodes:
            if n.mesh is not None:
                assert not (n.extras and n.extras.get("hakowan_cell"))

    def test_viewer_has_object_rotation(self, tmp_path):
        # The interactive viewer ships the object-rotation toggle and machinery.
        out_path = tmp_path / "viewer.html"
        hkw.render(
            hkw.layer(_make_icosphere()).mark(hkw.mark.Surface),
            filename=str(out_path),
            backend="webgl",
        )
        html = out_path.read_text()
        assert "btn-objrotate" in html
        assert "buildCellGroups" in html
        assert "hakowan_cell" in html
        # Juxtaposition scenes default to per-object rotation (>= 2 cells).
        assert "cellGroups.length >= 2" in html
        assert "setObjectRotateEnabled(true)" in html
