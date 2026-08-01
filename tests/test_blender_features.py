"""Tests for newly added Blender backend features.

Covers the camera fixes (up vector, orthographic, ThinLens DOF, fov_axis) and
the image-based texture channels (Image base color, normal map, bump map) that
were previously missing/ignored.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

if sys.platform == "win32" and os.environ.get("CI") == "true":
    pytest.skip("bpy crashes on Windows CI runners", allow_module_level=True)

bpy = pytest.importorskip("bpy")
import lagrange
from PIL import Image as PILImage

import hakowan as hkw
import hakowan.setup.sensor as S
from hakowan.backends.blender.render import BlenderBackend


# --------------------------------------------------------------------------- #
# Camera                                                                       #
# --------------------------------------------------------------------------- #


def _setup_camera(sensor, width=400, height=300):
    config = hkw.config()
    config.sensor = sensor
    config.film.width = width
    config.film.height = height
    backend = BlenderBackend()
    backend._clear_scene()
    backend._setup_camera(config)
    cam = bpy.context.scene.camera
    bpy.context.view_layer.update()
    return cam, cam.data


class TestCamera:
    def test_up_vector_honored(self):
        # Camera on -Y looking at origin with up=+Z (a z-up configuration).
        cam, _ = _setup_camera(
            S.Perspective(location=[0, -5, 0], target=[0, 0, 0], up=[0, 0, 1])
        )
        R = cam.matrix_world.to_3x3()
        world_up = np.array(R @ __import__("mathutils").Vector((0, 1, 0)))
        world_fwd = np.array(R @ __import__("mathutils").Vector((0, 0, -1)))
        assert np.allclose(world_up, [0, 0, 1], atol=1e-4)
        assert np.allclose(world_fwd, [0, 1, 0], atol=1e-4)  # toward target

    def test_clip_planes(self):
        _, cd = _setup_camera(S.Perspective(near_clip=0.5, far_clip=42.0))
        assert cd.clip_start == pytest.approx(0.5)
        assert cd.clip_end == pytest.approx(42.0)

    def test_orthographic(self):
        _, cd = _setup_camera(S.Orthographic())
        assert cd.type == "ORTHO"
        assert cd.ortho_scale == pytest.approx(2.0)

    def test_thinlens_enables_dof(self):
        _, cd = _setup_camera(S.ThinLens(aperture_radius=0.1, focus_distance=3.0))
        assert cd.type == "PERSP"
        assert cd.dof.use_dof is True
        assert cd.dof.focus_distance == pytest.approx(3.0)
        # radius 0.1 -> f/2.8 per the documented mapping.
        assert cd.dof.aperture_fstop == pytest.approx(2.8, rel=1e-3)

    def test_fov_axis_x_and_y(self):
        # 400x300 (width > height).
        _, cd = _setup_camera(S.Perspective(fov_axis="x"))
        assert cd.sensor_fit == "HORIZONTAL"
        _, cd = _setup_camera(S.Perspective(fov_axis="y"))
        assert cd.sensor_fit == "VERTICAL"
        # "smaller" with width>height -> the shorter axis is vertical.
        _, cd = _setup_camera(S.Perspective(fov_axis="smaller"))
        assert cd.sensor_fit == "VERTICAL"
        _, cd = _setup_camera(S.Perspective(fov_axis="larger"))
        assert cd.sensor_fit == "HORIZONTAL"


# --------------------------------------------------------------------------- #
# Image / normal / bump textures                                              #
# --------------------------------------------------------------------------- #


def _quad_with_uv() -> lagrange.SurfaceMesh:
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    )
    mesh.add_triangles(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32))
    mesh.create_attribute(
        "uv",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.UV,
        initial_values=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
    )
    return mesh


def _write_png(mode="RGB", color=(200, 100, 50)) -> str:
    path = tempfile.mktemp(suffix=".png")
    PILImage.new(mode, (8, 8), color).save(path)
    return path


def _build_material(layer):
    view = list(hkw.compiler.compile(layer))[0]
    backend = BlenderBackend()
    backend._clear_scene()
    backend._create_surface_object(view, 0)
    mesh = bpy.data.meshes["mesh_000"]
    nodes = bpy.data.materials["material_000"].node_tree.nodes
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    return mesh, nodes, bsdf


def _source_type(socket):
    return socket.links[0].from_node.type if socket.links else None


class TestImageTexture:
    def test_image_base_color_wired(self):
        png = _write_png()
        mesh, nodes, bsdf = _build_material(
            hkw.layer()
            .data(_quad_with_uv())
            .mark(hkw.mark.Surface)
            .channel(
                material=hkw.material.Diffuse(
                    reflectance=hkw.texture.Image(filename=png)
                )
            )
        )
        assert any(n.type == "TEX_IMAGE" for n in nodes)
        assert _source_type(bsdf.inputs["Base Color"]) == "TEX_IMAGE"
        assert "UVMap" in mesh.uv_layers.keys()


class TestFur:
    def _fur_view(self, n=8, segments=5):
        mesh = lagrange.SurfaceMesh()
        mesh.add_vertex([0.0, 0.0, 0.0])
        mesh.add_vertex([1.0, 0.0, 0.0])
        mesh.add_vertex([1.0, 1.0, 0.0])
        mesh.add_vertex([0.0, 1.0, 0.0])
        mesh.add_triangle(0, 1, 2)
        mesh.add_triangle(0, 2, 3)
        mesh.create_attribute(
            "flow",
            element=lagrange.AttributeElement.Facet,
            usage=lagrange.AttributeUsage.Vector,
            initial_values=np.array(
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
            ),
        )
        layer = (
            hkw.layer(mesh)
            .transform(
                hkw.transform.Fur(vec_field="flow", n=n, segments=segments, seed=1)
            )
            .mark("curve")
            .material("Hair")
        )
        return list(hkw.compiler.compile(layer))[0]

    def test_fur_creates_native_hair_curves(self):
        n, segments = 8, 5
        view = self._fur_view(n=n, segments=segments)
        backend = BlenderBackend()
        backend._clear_scene()
        backend._create_view_object(view, 0)

        # Fur is rendered as a native hair-Curves datablock (Cycles hair
        # primitives), not a beveled mesh-tube CURVE.
        assert "fur_000" in bpy.data.hair_curves
        curves = bpy.data.hair_curves["fur_000"]
        assert len(curves.curves) == n
        assert len(curves.points) == n * (segments + 1)
        assert curves.attributes.get("radius") is not None

        # The Hair BSDF material is assigned to the object.
        obj = bpy.data.objects["fur_000"]
        assert len(obj.data.materials) == 1
        mat_nodes = obj.data.materials[0].node_tree.nodes
        assert any(node.type == "BSDF_HAIR_PRINCIPLED" for node in mat_nodes)

    def test_fur_children_geometry_nodes_expand_strands(self):
        n, segments, children = 6, 5, 8
        mesh = lagrange.SurfaceMesh()
        for v in ([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]):
            mesh.add_vertex([float(c) for c in v])
        mesh.add_triangle(0, 1, 2)
        mesh.add_triangle(0, 2, 3)
        mesh.create_attribute(
            "flow",
            element=lagrange.AttributeElement.Facet,
            usage=lagrange.AttributeUsage.Vector,
            initial_values=np.array(
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
            ),
        )
        layer = (
            hkw.layer(mesh)
            .transform(
                hkw.transform.Fur(
                    vec_field="flow",
                    n=n,
                    segments=segments,
                    children=children,
                    clump=0.6,
                    seed=1,
                )
            )
            .mark("curve")
            .material("Hair")
        )
        view = list(hkw.compiler.compile(layer))[0]
        backend = BlenderBackend()
        backend._clear_scene()
        backend._create_view_object(view, 0)

        obj = bpy.data.objects["fur_000"]
        # A Geometry Nodes modifier is attached ...
        assert any(m.type == "NODES" for m in obj.modifiers)
        # ... the base data still holds only the guide strands ...
        assert len(bpy.data.hair_curves["fur_000"].curves) == n
        # ... but the evaluated geometry expands to n * children hairs.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        assert len(evaluated.data.curves) == n * children

    def _fur_hair_nodes(self, **mat_kwargs):
        mesh = lagrange.SurfaceMesh()
        for v in ([0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]):
            mesh.add_vertex([float(c) for c in v])
        mesh.add_triangle(0, 1, 2)
        mesh.add_triangle(0, 2, 3)
        mesh.create_attribute(
            "flow",
            element=lagrange.AttributeElement.Facet,
            usage=lagrange.AttributeUsage.Vector,
            initial_values=np.array(
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
            ),
        )
        mesh.create_attribute(
            "s",
            element=lagrange.AttributeElement.Facet,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=np.array([0.2, 0.8], dtype=np.float64),
        )
        layer = (
            hkw.layer(mesh)
            .transform(hkw.transform.Fur(vec_field="flow", n=6, seed=1))
            .mark("curve")
            .material("Hair", **mat_kwargs)
        )
        view = list(hkw.compiler.compile(layer))[0]
        backend = BlenderBackend()
        backend._clear_scene()
        backend._create_view_object(view, 0)
        return bpy.data.objects["fur_000"].data.materials[0].node_tree.nodes

    def _hair_bsdf(self, nodes):
        return next(n for n in nodes if n.type == "BSDF_HAIR_PRINCIPLED")

    def test_fur_hair_constant_rgb_color(self):
        # A constant RGB color uses the COLOR parametrization with the (linear)
        # color; blue channel dominates for a blue input.
        bsdf = self._hair_bsdf(self._fur_hair_nodes(color=[0.15, 0.35, 0.95]))
        assert bsdf.parametrization == "COLOR"
        rgba = list(bsdf.inputs["Color"].default_value)
        assert rgba[2] > rgba[0]  # blue > red

    def test_fur_hair_data_color_falls_back_to_melanin(self):
        # A data-driven color isn't supported by the hair renderer -> melanin.
        bsdf = self._hair_bsdf(
            self._fur_hair_nodes(
                color=hkw.texture.ScalarField(data="s", colormap="viridis")
            )
        )
        assert bsdf.parametrization == "MELANIN"

    def test_fur_hair_root_tip_gradient(self):
        # A root/tip gradient builds a Hair Info -> ColorRamp -> Color graph.
        nodes = self._fur_hair_nodes(
            root_color=[0.05, 0.02, 0.01], tip_color=[0.9, 0.65, 0.3]
        )
        bsdf = self._hair_bsdf(nodes)
        assert bsdf.parametrization == "COLOR"
        assert bsdf.inputs["Color"].is_linked
        assert any(n.type == "HAIR_INFO" for n in nodes)
        ramp = next(n for n in nodes if n.type == "VALTORGB")
        # Ramp endpoints hold the (linear) root and tip colors: dark -> light.
        root_v = sum(ramp.color_ramp.elements[0].color[:3])
        tip_v = sum(ramp.color_ramp.elements[1].color[:3])
        assert tip_v > root_v

    def test_fur_hair_color_variation_adds_huesaturation(self):
        # Per-strand variation on a constant color inserts a Hue/Sat/Value node
        # driven by the Hair Info random output.
        nodes = self._fur_hair_nodes(color=[0.4, 0.25, 0.13], color_variation=0.5)
        assert any(n.type == "HUE_SAT" for n in nodes)
        assert any(n.type == "HAIR_INFO" for n in nodes)

    def test_fur_hair_melanin_variation_sets_random_color(self):
        # Variation with the melanin parametrization uses the BSDF Random Color.
        bsdf = self._hair_bsdf(self._fur_hair_nodes(color_variation=0.3))
        assert bsdf.parametrization == "MELANIN"
        assert bsdf.inputs["Random Color"].default_value == pytest.approx(0.3)


class TestSmoke:
    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="headless Blender render not supported in CI",
    )
    def test_blender_render_produces_image(self, triangle, tmp_path):
        """End-to-end smoke: exercises the full bpy render path (EEVEE, 16×16)."""
        config = hkw.config()
        config.film.width = 16
        config.film.height = 16
        config.sampler.sample_count = 1
        layer = hkw.layer().data(triangle).mark(hkw.mark.Surface)
        out = tmp_path / "smoke.png"
        hkw.render(
            layer,
            config,
            filename=out,
            backend="blender",
            blender_engine="BLENDER_EEVEE",
        )
        assert out.exists() and out.stat().st_size > 0

    def _smoke_layer(self, triangle):
        config = hkw.config()
        config.film.width = config.film.height = 16
        config.sampler.sample_count = 1
        return hkw.layer().data(triangle).mark(hkw.mark.Surface), config

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="headless Blender render not supported in CI",
    )
    @pytest.mark.parametrize("ext", [".png", ".webp", ".jpg", ".tif"])
    def test_blender_writes_pillow_formats(self, triangle, tmp_path, ext):
        """Non-EXR LDR output is re-encoded from PNG via Pillow, so any
        Pillow-writable format is produced and no PNG intermediate is left."""
        layer, config = self._smoke_layer(triangle)
        out = tmp_path / f"img{ext}"
        result = hkw.render(
            layer,
            config,
            filename=out,
            backend="blender",
            blender_engine="BLENDER_EEVEE",
        )
        assert result.path == out
        assert out.exists() and out.stat().st_size > 0
        with PILImage.open(out) as im:
            assert im.size == (16, 16)
        # The PNG intermediate must not leak into the output directory.
        if ext != ".png":
            assert not (tmp_path / "img.png").exists()

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="headless Blender render not supported in CI",
    )
    def test_blender_webp_does_not_clobber_existing_png(self, triangle, tmp_path):
        """Rendering ``scene.webp`` must not touch a pre-existing ``scene.png``."""
        sentinel = tmp_path / "scene.png"
        sentinel.write_bytes(b"SENTINEL")
        layer, config = self._smoke_layer(triangle)
        hkw.render(
            layer,
            config,
            filename=tmp_path / "scene.webp",
            backend="blender",
            blender_engine="BLENDER_EEVEE",
        )
        assert (tmp_path / "scene.webp").exists()
        assert sentinel.read_bytes() == b"SENTINEL"

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="headless Blender render not supported in CI",
    )
    def test_blender_unsupported_format_raises(self, triangle, tmp_path):
        layer, config = self._smoke_layer(triangle)
        with pytest.raises(ValueError, match="Unsupported output image format"):
            hkw.render(
                layer,
                config,
                filename=tmp_path / "img.xyz",
                backend="blender",
                blender_engine="BLENDER_EEVEE",
            )

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="headless Blender render not supported in CI",
    )
    def test_blender_facet_id_stays_lossless_for_lossy_output(self, triangle, tmp_path):
        """A lossy main format (.jpg) must not corrupt the discrete facet-id
        sidecar: it stays PNG (FACET_ID.discrete lossless contract), not .jpg."""
        layer, config = self._smoke_layer(triangle)
        config.facet_id = True
        out = tmp_path / "img.jpg"
        hkw.render(
            layer,
            config,
            filename=out,
            backend="blender",
            blender_engine="BLENDER_EEVEE",
        )
        assert out.exists()
        # Sidecar is kept lossless as PNG; the lossy .jpg sidecar must not exist.
        assert (tmp_path / "img_facet_id.png").exists()
        assert not (tmp_path / "img_facet_id.jpg").exists()


class TestNormalAndBumpMap:
    def test_normal_map_wired(self):
        nrm = _write_png(color=(128, 128, 255))
        _, nodes, bsdf = _build_material(
            hkw.layer()
            .data(_quad_with_uv())
            .mark(hkw.mark.Surface)
            .channel(
                material=hkw.material.Diffuse(reflectance="ivory"),
                normal_map=hkw.channel.NormalMap(
                    texture=hkw.texture.Image(filename=nrm, raw=True)
                ),
            )
        )
        assert any(n.type == "NORMAL_MAP" for n in nodes)
        assert _source_type(bsdf.inputs["Normal"]) == "NORMAL_MAP"

    def test_bump_map_wired(self):
        bmp = _write_png(mode="L", color=128)
        _, nodes, bsdf = _build_material(
            hkw.layer()
            .data(_quad_with_uv())
            .mark(hkw.mark.Surface)
            .channel(
                material=hkw.material.Diffuse(reflectance="ivory"),
                bump_map=hkw.channel.BumpMap(
                    texture=hkw.texture.Image(filename=bmp, raw=True), scale=0.5
                ),
            )
        )
        bump = next(n for n in nodes if n.type == "BUMP")
        assert _source_type(bsdf.inputs["Normal"]) == "BUMP"
        assert bump.inputs["Distance"].default_value == pytest.approx(0.5)

    def test_normal_and_bump_compose(self):
        nrm = _write_png(color=(128, 128, 255))
        bmp = _write_png(mode="L", color=128)
        _, nodes, bsdf = _build_material(
            hkw.layer()
            .data(_quad_with_uv())
            .mark(hkw.mark.Surface)
            .channel(
                material=hkw.material.Diffuse(reflectance="ivory"),
                normal_map=hkw.channel.NormalMap(
                    texture=hkw.texture.Image(filename=nrm, raw=True)
                ),
                bump_map=hkw.channel.BumpMap(
                    texture=hkw.texture.Image(filename=bmp, raw=True)
                ),
            )
        )
        bump = next(n for n in nodes if n.type == "BUMP")
        # Bump drives the BSDF normal; the normal map feeds the bump node.
        assert _source_type(bsdf.inputs["Normal"]) == "BUMP"
        assert _source_type(bump.inputs["Normal"]) == "NORMAL_MAP"


class TestBackSide:
    def test_back_side_builds_mix_shader(self):
        _, nodes, _ = _build_material(
            hkw.layer()
            .data(_quad_with_uv())
            .mark(hkw.mark.Surface)
            .channel(
                material=hkw.material.Diffuse(
                    "red",
                    back_side=hkw.material.Principled(color="blue", metallic=1.0),
                )
            )
        )
        # A back_side mixes a second BSDF in via the Geometry "Backfacing" output.
        assert any(n.type == "MIX_SHADER" for n in nodes)
        assert any(n.type == "NEW_GEOMETRY" for n in nodes)
        assert sum(1 for n in nodes if n.type == "BSDF_PRINCIPLED") == 2
        assert bpy.data.materials["material_000"].use_backface_culling is False

    def test_back_side_textured_back_falls_back(self):
        mesh = _quad_with_uv()
        mesh.create_attribute(
            "scalar",
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=np.array([0.0, 0.3, 0.6, 1.0], dtype=np.float64),
        )
        _, nodes, _ = _build_material(
            hkw.layer()
            .data(mesh)
            .mark(hkw.mark.Surface)
            .channel(
                material=hkw.material.Diffuse(
                    "red",
                    back_side=hkw.material.Diffuse(
                        reflectance=hkw.texture.ScalarField(
                            data=hkw.attribute(name="scalar")
                        )
                    ),
                )
            )
        )
        # The textured back color is unsupported; it falls back to a flat BSDF
        # but the mix structure is still built.
        assert any(n.type == "MIX_SHADER" for n in nodes)
        assert sum(1 for n in nodes if n.type == "BSDF_PRINCIPLED") == 2
