"""Tests for hakowan.backends.webgl.builder."""

from __future__ import annotations

import json

import numpy as np
import pytest

pygltflib = pytest.importorskip("pygltflib")

from hakowan.backends.webgl.builder import (
    MODE_LINES,
    MODE_TRIANGLES,
    GLTFBuilder,
)


def _roundtrip(glb_bytes: bytes) -> pygltflib.GLTF2:
    return pygltflib.GLTF2().load_from_bytes(glb_bytes)


def _basic_triangle_positions_indices():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    return positions, indices


class TestGLBRoundtrip:
    def test_minimal_mesh(self):
        b = GLTFBuilder()
        mat = b.add_material({"baseColorFactor": [1, 0, 0, 1]})
        positions, indices = _basic_triangle_positions_indices()
        b.add_mesh_node(positions, indices, material_idx=mat)
        glb = b.finalize()
        gltf = _roundtrip(glb)
        assert len(gltf.meshes) == 1
        assert len(gltf.materials) == 1
        assert len(gltf.nodes) == 1
        assert gltf.scenes[0].nodes == [0]

    def test_indices_uint16_for_small_meshes(self):
        b = GLTFBuilder()
        mat = b.add_material({})
        positions = np.random.rand(100, 3).astype(np.float32)
        indices = np.random.randint(0, 100, size=(60,)).astype(np.uint32)
        b.add_mesh_node(positions, indices, material_idx=mat)
        gltf = _roundtrip(b.finalize())
        idx_acc = gltf.accessors[gltf.meshes[0].primitives[0].indices]
        assert idx_acc.componentType == 5123  # UNSIGNED_SHORT

    def test_indices_uint32_for_large_meshes(self):
        b = GLTFBuilder()
        mat = b.add_material({})
        n = 70_000
        positions = np.random.rand(n, 3).astype(np.float32)
        indices = np.random.randint(0, n, size=(6,)).astype(np.uint32)
        b.add_mesh_node(positions, indices, material_idx=mat)
        gltf = _roundtrip(b.finalize())
        idx_acc = gltf.accessors[gltf.meshes[0].primitives[0].indices]
        assert idx_acc.componentType == 5125  # UNSIGNED_INT

    def test_normals_optional(self):
        b = GLTFBuilder()
        mat = b.add_material({})
        positions, indices = _basic_triangle_positions_indices()
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        b.add_mesh_node(positions, indices, normals=normals, material_idx=mat)
        gltf = _roundtrip(b.finalize())
        attrs = gltf.meshes[0].primitives[0].attributes
        assert attrs.POSITION is not None
        assert attrs.NORMAL is not None

    def test_colors_and_uvs(self):
        b = GLTFBuilder()
        mat = b.add_material({})
        positions, indices = _basic_triangle_positions_indices()
        colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
        uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        b.add_mesh_node(positions, indices, colors=colors, uvs=uvs, material_idx=mat)
        gltf = _roundtrip(b.finalize())
        attrs = gltf.meshes[0].primitives[0].attributes
        assert attrs.COLOR_0 is not None
        assert attrs.TEXCOORD_0 is not None

    def test_lines_mode(self):
        b = GLTFBuilder()
        mat = b.add_material({})
        endpoints = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32
        )
        indices = np.arange(4, dtype=np.uint32)
        b.add_mesh_node(endpoints, indices, material_idx=mat, mode=MODE_LINES)
        gltf = _roundtrip(b.finalize())
        assert gltf.meshes[0].primitives[0].mode == MODE_LINES

    def test_custom_attribute_underscore_required(self):
        b = GLTFBuilder()
        mat = b.add_material({})
        positions, indices = _basic_triangle_positions_indices()
        with pytest.raises(ValueError, match="must start with '_'"):
            b.add_mesh_node(
                positions,
                indices,
                custom_attributes={"BAD_NAME": np.zeros(3, dtype=np.float32)},
                material_idx=mat,
            )

    def test_custom_attribute_in_output(self):
        b = GLTFBuilder()
        mat = b.add_material({})
        positions, indices = _basic_triangle_positions_indices()
        b.add_mesh_node(
            positions,
            indices,
            custom_attributes={
                "_scalar_0": np.array([0.1, 0.5, 0.9], dtype=np.float32)
            },
            material_idx=mat,
        )
        glb = b.finalize()
        # pygltflib's standard from_bytes doesn't surface unknown attribute
        # fields as object members; verify they appear in the serialized JSON.
        gltf = _roundtrip(glb)
        attrs_json = gltf.meshes[0].primitives[0].attributes.to_json()
        assert "_scalar_0" in attrs_json


class TestMaterials:
    def test_base_color_factor_passthrough(self):
        b = GLTFBuilder()
        idx = b.add_material({"baseColorFactor": [0.2, 0.4, 0.6, 1.0]})
        assert idx == 0
        gltf = _roundtrip(_minimal_with_material(b))
        assert gltf.materials[0].pbrMetallicRoughness.baseColorFactor == [
            0.2,
            0.4,
            0.6,
            1.0,
        ]

    def test_metallic_roughness(self):
        b = GLTFBuilder()
        b.add_material({"metallicFactor": 0.8, "roughnessFactor": 0.3})
        gltf = _roundtrip(_minimal_with_material(b))
        pbr = gltf.materials[0].pbrMetallicRoughness
        assert pbr.metallicFactor == pytest.approx(0.8)
        assert pbr.roughnessFactor == pytest.approx(0.3)

    def test_double_sided_flag(self):
        b = GLTFBuilder()
        b.add_material({}, double_sided=True)
        gltf = _roundtrip(_minimal_with_material(b))
        assert gltf.materials[0].doubleSided is True

    def test_base_color_texture_with_scale_registers_extension(self):
        b = GLTFBuilder()
        # Need a texture index that exists.
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50  # dummy; not parsed.
        tex_idx = b.add_image_texture(png)
        b.add_material(
            {
                "baseColorTextureIndex": tex_idx,
                "baseColorTextureScale": (4.0, 4.0),
            }
        )
        gltf = _roundtrip(_minimal_with_material(b))
        assert "KHR_texture_transform" in gltf.extensionsUsed
        tex_info = gltf.materials[0].pbrMetallicRoughness.baseColorTexture
        assert tex_info.extensions["KHR_texture_transform"]["scale"] == [4.0, 4.0]

    def test_normal_texture(self):
        b = GLTFBuilder()
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        tex_idx = b.add_image_texture(png)
        b.add_material({"normalTextureIndex": tex_idx, "normalScale": 0.5})
        gltf = _roundtrip(_minimal_with_material(b))
        nt = gltf.materials[0].normalTexture
        assert nt is not None
        assert nt.index == tex_idx
        assert nt.scale == pytest.approx(0.5)

    def test_material_extras_pass_through(self):
        b = GLTFBuilder()
        extras = {"hakowan": {"isocontour": {"num_contours": 8}}}
        b.add_material({"extras": extras})
        gltf = _roundtrip(_minimal_with_material(b))
        assert gltf.materials[0].extras == extras


def _minimal_with_material(b: GLTFBuilder) -> bytes:
    """Add a placeholder triangle so the GLB is non-empty."""
    if not b._gltf.meshes:
        positions, indices = _basic_triangle_positions_indices()
        b.add_mesh_node(positions, indices, material_idx=0)
    return b.finalize()


class TestCameras:
    def test_perspective_camera_round_trip(self):
        b = GLTFBuilder()
        b.add_material({})
        positions, indices = _basic_triangle_positions_indices()
        b.add_mesh_node(positions, indices, material_idx=0)
        b.add_perspective_camera(
            yfov=0.5,
            aspect_ratio=16 / 9,
            znear=0.01,
            zfar=100,
            world_transform_4x4=np.eye(4),
        )
        gltf = _roundtrip(b.finalize())
        assert len(gltf.cameras) == 1
        p = gltf.cameras[0].perspective
        assert p.yfov == pytest.approx(0.5)
        assert p.aspectRatio == pytest.approx(16 / 9)
        assert p.znear == pytest.approx(0.01)
        assert p.zfar == pytest.approx(100)

    def test_orthographic_camera(self):
        b = GLTFBuilder()
        b.add_material({})
        positions, indices = _basic_triangle_positions_indices()
        b.add_mesh_node(positions, indices, material_idx=0)
        b.add_orthographic_camera(
            xmag=1.5,
            ymag=1.0,
            znear=0.1,
            zfar=50,
            world_transform_4x4=np.eye(4),
        )
        gltf = _roundtrip(b.finalize())
        o = gltf.cameras[0].orthographic
        assert o.xmag == pytest.approx(1.5)
        assert o.ymag == pytest.approx(1.0)


class TestPointLights:
    def test_point_light_registers_extension(self):
        b = GLTFBuilder()
        b.add_material({})
        positions, indices = _basic_triangle_positions_indices()
        b.add_mesh_node(positions, indices, material_idx=0)
        b.add_point_light(position=(1, 2, 3), color=(1, 1, 1), intensity=5.0)
        glb = b.finalize()
        # Roundtripping via from_bytes drops custom extensions, so inspect
        # the embedded JSON directly.
        gltf = _roundtrip(glb)
        assert "KHR_lights_punctual" in gltf.extensionsUsed
        # The extension lives on the top-level glTF.
        ext = gltf.extensions.get("KHR_lights_punctual")
        assert ext is not None
        lights = ext["lights"]
        assert lights[0]["type"] == "point"
        assert lights[0]["intensity"] == pytest.approx(5.0)
