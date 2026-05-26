"""Thin glTF assembly layer around ``pygltflib``.

``GLTFBuilder`` accumulates a single binary buffer plus the list of
bufferViews / accessors / meshes / materials / cameras / nodes that reference
it, then serialises everything as a GLB byte string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pygltflib

from .utils import gltf_matrix, np_to_bytes


# glTF component-type constants (subset)
_COMPONENT_UINT16 = 5123
_COMPONENT_UINT32 = 5125
_COMPONENT_FLOAT = 5126

# bufferView targets
_TARGET_ARRAY_BUFFER = 34962
_TARGET_ELEMENT_ARRAY_BUFFER = 34963


@dataclass
class GLTFBuilder:
    """Accumulates glTF nodes/meshes/materials and produces a final GLB."""

    _gltf: pygltflib.GLTF2 = field(default_factory=pygltflib.GLTF2)
    _bin: bytearray = field(default_factory=bytearray)
    _scene_node_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Ensure a buffer slot exists; its length is patched at finalize time.
        self._gltf.buffers.append(pygltflib.Buffer(byteLength=0))
        # Default scene.
        self._gltf.scenes.append(pygltflib.Scene(nodes=[]))
        self._gltf.scene = 0
        self._gltf.asset = pygltflib.Asset(version="2.0", generator="hakowan-webgl")

    # ------------------------------------------------------------------ #
    # Low-level buffer helpers                                            #
    # ------------------------------------------------------------------ #

    def _align(self, alignment: int = 4) -> None:
        pad = (-len(self._bin)) % alignment
        if pad:
            self._bin.extend(b"\x00" * pad)

    def _add_buffer_view(self, data: bytes, target: int) -> int:
        self._align(4)
        offset = len(self._bin)
        self._bin.extend(data)
        view = pygltflib.BufferView(
            buffer=0, byteOffset=offset, byteLength=len(data), target=target
        )
        self._gltf.bufferViews.append(view)
        return len(self._gltf.bufferViews) - 1

    def _add_accessor(
        self,
        arr: np.ndarray,
        component_type: int,
        gltf_type: str,
        target: int,
        compute_minmax: bool,
    ) -> int:
        data = np_to_bytes(arr)
        view_idx = self._add_buffer_view(data, target)

        kwargs: dict[str, Any] = {
            "bufferView": view_idx,
            "componentType": component_type,
            "count": arr.shape[0],
            "type": gltf_type,
        }
        if compute_minmax:
            # glTF requires min/max for POSITION accessors; provide for all
            # vector attributes since spec-validators expect them.
            if arr.ndim == 1:
                kwargs["min"] = [float(arr.min())]
                kwargs["max"] = [float(arr.max())]
            else:
                kwargs["min"] = arr.min(axis=0).astype(float).tolist()
                kwargs["max"] = arr.max(axis=0).astype(float).tolist()
        self._gltf.accessors.append(pygltflib.Accessor(**kwargs))
        return len(self._gltf.accessors) - 1

    # ------------------------------------------------------------------ #
    # High-level adders                                                   #
    # ------------------------------------------------------------------ #

    def add_material(self, pbr: dict[str, Any], double_sided: bool = False) -> int:
        pbr_material = pygltflib.PbrMetallicRoughness(
            baseColorFactor=pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1.0]),
            metallicFactor=pbr.get("metallicFactor", 0.0),
            roughnessFactor=pbr.get("roughnessFactor", 1.0),
        )
        mat = pygltflib.Material(
            pbrMetallicRoughness=pbr_material, doubleSided=double_sided
        )
        self._gltf.materials.append(mat)
        return len(self._gltf.materials) - 1

    def add_mesh_node(
        self,
        positions: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray | None,
        material_idx: int,
        transform_4x4: np.ndarray | None = None,
    ) -> int:
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must be (N, 3)")

        # Choose index width based on vertex count.
        index_component = (
            _COMPONENT_UINT16 if positions.shape[0] <= 65535 else _COMPONENT_UINT32
        )
        index_dtype = np.uint16 if index_component == _COMPONENT_UINT16 else np.uint32
        indices_arr = np.ascontiguousarray(indices, dtype=index_dtype).reshape(-1)

        pos_acc = self._add_accessor(
            positions, _COMPONENT_FLOAT, "VEC3", _TARGET_ARRAY_BUFFER, True
        )
        idx_acc = self._add_accessor(
            indices_arr,
            index_component,
            "SCALAR",
            _TARGET_ELEMENT_ARRAY_BUFFER,
            False,
        )

        attributes: dict[str, int] = {"POSITION": pos_acc}
        if normals is not None:
            normals = np.ascontiguousarray(normals, dtype=np.float32)
            assert normals.shape == positions.shape, "normals must match positions"
            nor_acc = self._add_accessor(
                normals, _COMPONENT_FLOAT, "VEC3", _TARGET_ARRAY_BUFFER, False
            )
            attributes["NORMAL"] = nor_acc

        primitive = pygltflib.Primitive(
            attributes=pygltflib.Attributes(**attributes),
            indices=idx_acc,
            material=material_idx,
            mode=4,  # TRIANGLES
        )
        mesh = pygltflib.Mesh(primitives=[primitive])
        self._gltf.meshes.append(mesh)
        mesh_idx = len(self._gltf.meshes) - 1

        node_kwargs: dict[str, Any] = {"mesh": mesh_idx}
        if transform_4x4 is not None and not np.allclose(transform_4x4, np.eye(4)):
            node_kwargs["matrix"] = gltf_matrix(transform_4x4)
        node = pygltflib.Node(**node_kwargs)
        self._gltf.nodes.append(node)
        node_idx = len(self._gltf.nodes) - 1
        self._gltf.scenes[0].nodes.append(node_idx)
        self._scene_node_indices.append(node_idx)
        return node_idx

    def add_perspective_camera(
        self,
        yfov: float,
        aspect_ratio: float,
        znear: float,
        zfar: float | None,
        world_transform_4x4: np.ndarray,
    ) -> int:
        persp_kwargs: dict[str, Any] = {
            "yfov": yfov,
            "aspectRatio": aspect_ratio,
            "znear": znear,
        }
        if zfar is not None and zfar > 0 and np.isfinite(zfar):
            persp_kwargs["zfar"] = zfar
        persp = pygltflib.Perspective(**persp_kwargs)
        cam = pygltflib.Camera(type="perspective", perspective=persp)
        self._gltf.cameras.append(cam)
        cam_idx = len(self._gltf.cameras) - 1

        node = pygltflib.Node(
            camera=cam_idx, matrix=gltf_matrix(world_transform_4x4)
        )
        self._gltf.nodes.append(node)
        node_idx = len(self._gltf.nodes) - 1
        self._gltf.scenes[0].nodes.append(node_idx)
        return node_idx

    def add_orthographic_camera(
        self,
        xmag: float,
        ymag: float,
        znear: float,
        zfar: float,
        world_transform_4x4: np.ndarray,
    ) -> int:
        ortho = pygltflib.Orthographic(xmag=xmag, ymag=ymag, znear=znear, zfar=zfar)
        cam = pygltflib.Camera(type="orthographic", orthographic=ortho)
        self._gltf.cameras.append(cam)
        cam_idx = len(self._gltf.cameras) - 1

        node = pygltflib.Node(
            camera=cam_idx, matrix=gltf_matrix(world_transform_4x4)
        )
        self._gltf.nodes.append(node)
        node_idx = len(self._gltf.nodes) - 1
        self._gltf.scenes[0].nodes.append(node_idx)
        return node_idx

    # ------------------------------------------------------------------ #
    # Finalisation                                                        #
    # ------------------------------------------------------------------ #

    def finalize(self) -> bytes:
        """Patch buffer length and return GLB bytes."""
        self._align(4)
        blob = bytes(self._bin)
        self._gltf.buffers[0].byteLength = len(blob)
        self._gltf.set_binary_blob(blob)
        return b"".join(self._gltf.save_to_bytes())
