"""Tests for newly added Blender backend features.

Covers the camera fixes (up vector, orthographic, ThinLens DOF, fov_axis) and
the image-based texture channels (Image base color, normal map, bump map) that
were previously missing/ignored.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

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
            .channel(material=hkw.material.Diffuse(reflectance=hkw.texture.Image(filename=png)))
        )
        assert any(n.type == "TEX_IMAGE" for n in nodes)
        assert _source_type(bsdf.inputs["Base Color"]) == "TEX_IMAGE"
        assert "UVMap" in mesh.uv_layers.keys()


class TestSmoke:
    def test_blender_render_produces_image(self, triangle, tmp_path):
        """End-to-end smoke: exercises the full bpy render path (EEVEE, 16×16)."""
        config = hkw.config()
        config.film.width = 16
        config.film.height = 16
        config.sampler.sample_count = 1
        layer = hkw.layer().data(triangle).mark(hkw.mark.Surface)
        out = tmp_path / "smoke.png"
        hkw.render(layer, config, filename=out, backend="blender", engine="BLENDER_EEVEE")
        assert out.exists() and out.stat().st_size > 0


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
