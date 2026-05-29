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

# glTF primitive modes
MODE_TRIANGLES = 4
MODE_LINES = 1
MODE_POINTS = 0


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

    def _add_buffer_view(
        self, data: bytes, target: int | None = None
    ) -> int:
        self._align(4)
        offset = len(self._bin)
        self._bin.extend(data)
        view_kwargs: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(data),
        }
        if target is not None:
            view_kwargs["target"] = target
        self._gltf.bufferViews.append(pygltflib.BufferView(**view_kwargs))
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
            if arr.ndim == 1:
                kwargs["min"] = [float(arr.min())]
                kwargs["max"] = [float(arr.max())]
            else:
                kwargs["min"] = arr.min(axis=0).astype(float).tolist()
                kwargs["max"] = arr.max(axis=0).astype(float).tolist()
        self._gltf.accessors.append(pygltflib.Accessor(**kwargs))
        return len(self._gltf.accessors) - 1

    # ------------------------------------------------------------------ #
    # Image / texture                                                     #
    # ------------------------------------------------------------------ #

    def add_image_texture(self, png_bytes: bytes) -> int:
        """Embed a PNG image as a glTF texture and return the texture index."""
        view_idx = self._add_buffer_view(png_bytes, target=None)
        image = pygltflib.Image(mimeType="image/png", bufferView=view_idx)
        self._gltf.images.append(image)
        image_idx = len(self._gltf.images) - 1

        if not self._gltf.samplers:
            self._gltf.samplers.append(
                pygltflib.Sampler(
                    magFilter=9729,  # LINEAR
                    minFilter=9987,  # LINEAR_MIPMAP_LINEAR
                    wrapS=10497,     # REPEAT
                    wrapT=10497,     # REPEAT
                )
            )
        sampler_idx = 0

        texture = pygltflib.Texture(source=image_idx, sampler=sampler_idx)
        self._gltf.textures.append(texture)
        return len(self._gltf.textures) - 1

    # ------------------------------------------------------------------ #
    # Material                                                            #
    # ------------------------------------------------------------------ #

    def add_material(self, pbr: dict[str, Any], double_sided: bool = False) -> int:
        pbr_kwargs: dict[str, Any] = {
            "baseColorFactor": pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1.0]),
            "metallicFactor": pbr.get("metallicFactor", 0.0),
            "roughnessFactor": pbr.get("roughnessFactor", 1.0),
        }
        if "baseColorTextureIndex" in pbr:
            tex_info = pygltflib.TextureInfo(index=pbr["baseColorTextureIndex"])
            scale = pbr.get("baseColorTextureScale")
            if scale is not None:
                # KHR_texture_transform: scale UVs at sample time so a 2x2
                # checker tiles `scale` times across the existing UV range.
                tex_info.extensions = {
                    "KHR_texture_transform": {
                        "scale": [float(scale[0]), float(scale[1])],
                    }
                }
                self._register_extension("KHR_texture_transform")
            pbr_kwargs["baseColorTexture"] = tex_info
        pbr_material = pygltflib.PbrMetallicRoughness(**pbr_kwargs)
        mat_kwargs: dict[str, Any] = {
            "pbrMetallicRoughness": pbr_material,
            "doubleSided": double_sided,
        }
        if "normalTextureIndex" in pbr:
            normal_kwargs: dict[str, Any] = {"index": pbr["normalTextureIndex"]}
            if "normalScale" in pbr:
                normal_kwargs["scale"] = float(pbr["normalScale"])
            mat_kwargs["normalTexture"] = pygltflib.NormalMaterialTexture(
                **normal_kwargs
            )
        if "extras" in pbr:
            mat_kwargs["extras"] = pbr["extras"]

        # KHR_materials_transmission / _ior / _volume: glass / dielectric.
        # three.js GLTFLoader maps these onto MeshPhysicalMaterial.transmission,
        # .ior, .thickness, .attenuationDistance, .attenuationColor — the
        # renderer then handles backside-rendering for refraction automatically.
        material_extensions: dict[str, Any] = {}
        if "transmissionFactor" in pbr:
            material_extensions["KHR_materials_transmission"] = {
                "transmissionFactor": float(pbr["transmissionFactor"]),
            }
            self._register_extension("KHR_materials_transmission")
        if "ior" in pbr:
            material_extensions["KHR_materials_ior"] = {"ior": float(pbr["ior"])}
            self._register_extension("KHR_materials_ior")
        if any(k in pbr for k in ("thicknessFactor", "attenuationDistance", "attenuationColor")):
            vol: dict[str, Any] = {}
            if "thicknessFactor" in pbr:
                vol["thicknessFactor"] = float(pbr["thicknessFactor"])
            if "attenuationDistance" in pbr:
                vol["attenuationDistance"] = float(pbr["attenuationDistance"])
            if "attenuationColor" in pbr:
                vol["attenuationColor"] = [float(c) for c in pbr["attenuationColor"]]
            material_extensions["KHR_materials_volume"] = vol
            self._register_extension("KHR_materials_volume")
        if material_extensions:
            mat_kwargs["extensions"] = material_extensions

        mat = pygltflib.Material(**mat_kwargs)
        self._gltf.materials.append(mat)
        return len(self._gltf.materials) - 1

    def _register_extension(self, name: str) -> None:
        if self._gltf.extensionsUsed is None:
            self._gltf.extensionsUsed = []
        if name not in self._gltf.extensionsUsed:
            self._gltf.extensionsUsed.append(name)

    # ------------------------------------------------------------------ #
    # Geometry                                                            #
    # ------------------------------------------------------------------ #

    def add_mesh_node(
        self,
        positions: np.ndarray,
        indices: np.ndarray,
        *,
        normals: np.ndarray | None = None,
        colors: np.ndarray | None = None,
        uvs: np.ndarray | None = None,
        custom_attributes: dict[str, np.ndarray] | None = None,
        material_idx: int,
        mode: int = MODE_TRIANGLES,
        transform_4x4: np.ndarray | None = None,
    ) -> int:
        """Add a single-primitive mesh node referencing the given attributes.

        ``colors`` may be Nx3 (RGB) or Nx4 (RGBA) float32 in linear [0, 1].
        ``uvs`` is Nx2 float32. ``mode`` is one of the ``MODE_*`` constants.
        ``custom_attributes`` is a name → ndarray map for additional generic
        vertex attributes (e.g. ``"_SCALAR_0"`` for shader-injection paths);
        names must follow the glTF underscore-prefix convention.
        """
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must be (N, 3)")
        n_vertices = positions.shape[0]

        index_component = (
            _COMPONENT_UINT16 if n_vertices <= 65535 else _COMPONENT_UINT32
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
            if normals.shape != positions.shape:
                raise ValueError("normals must match positions shape")
            attributes["NORMAL"] = self._add_accessor(
                normals, _COMPONENT_FLOAT, "VEC3", _TARGET_ARRAY_BUFFER, False
            )
        if colors is not None:
            colors = np.ascontiguousarray(colors, dtype=np.float32)
            if colors.ndim != 2 or colors.shape[0] != n_vertices:
                raise ValueError("colors must be (N, 3) or (N, 4)")
            gltf_type = "VEC3" if colors.shape[1] == 3 else "VEC4"
            attributes["COLOR_0"] = self._add_accessor(
                colors, _COMPONENT_FLOAT, gltf_type, _TARGET_ARRAY_BUFFER, False
            )
        if uvs is not None:
            uvs = np.ascontiguousarray(uvs, dtype=np.float32)
            if uvs.ndim != 2 or uvs.shape != (n_vertices, 2):
                raise ValueError("uvs must be (N, 2)")
            attributes["TEXCOORD_0"] = self._add_accessor(
                uvs, _COMPONENT_FLOAT, "VEC2", _TARGET_ARRAY_BUFFER, False
            )
        if custom_attributes:
            for name, arr in custom_attributes.items():
                if not name.startswith("_"):
                    raise ValueError(
                        f"custom attribute name '{name}' must start with '_' "
                        "per glTF convention"
                    )
                arr = np.ascontiguousarray(arr, dtype=np.float32)
                if arr.shape[0] != n_vertices:
                    raise ValueError(
                        f"custom attribute '{name}' length {arr.shape[0]} != "
                        f"vertex count {n_vertices}"
                    )
                if arr.ndim == 1:
                    gltf_type = "SCALAR"
                elif arr.ndim == 2 and arr.shape[1] == 2:
                    gltf_type = "VEC2"
                elif arr.ndim == 2 and arr.shape[1] == 3:
                    gltf_type = "VEC3"
                elif arr.ndim == 2 and arr.shape[1] == 4:
                    gltf_type = "VEC4"
                else:
                    raise ValueError(
                        f"custom attribute '{name}' must be SCALAR / VEC2 / "
                        "VEC3 / VEC4 shape"
                    )
                attributes[name] = self._add_accessor(
                    arr, _COMPONENT_FLOAT, gltf_type, _TARGET_ARRAY_BUFFER, False
                )

        # Separate standard glTF attributes from custom underscore-prefixed
        # ones — pygltflib.Attributes is a dataclass for the standard set but
        # accepts custom attributes assigned via setattr (it serialises them
        # in to_json).
        std_attrs = {k: v for k, v in attributes.items() if not k.startswith("_")}
        attrs_obj = pygltflib.Attributes(**std_attrs)
        for k, v in attributes.items():
            if k.startswith("_"):
                setattr(attrs_obj, k, v)
        primitive = pygltflib.Primitive(
            attributes=attrs_obj,
            indices=idx_acc,
            material=material_idx,
            mode=mode,
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

    # ------------------------------------------------------------------ #
    # Cameras                                                              #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Lights (KHR_lights_punctual)                                         #
    # ------------------------------------------------------------------ #

    def add_point_light(
        self,
        position: tuple[float, float, float] | list[float],
        color: tuple[float, float, float] | list[float],
        intensity: float,
    ) -> int:
        """Register a KHR_lights_punctual point light and parent it to a node.

        glTF places lights via node transforms; we emit a node at the given
        world-space position carrying the light.
        """
        ext_obj = self._gltf.extensions or {}
        khr = ext_obj.setdefault("KHR_lights_punctual", {"lights": []})
        light = {
            "type": "point",
            "color": [float(color[0]), float(color[1]), float(color[2])],
            "intensity": float(intensity),
        }
        khr["lights"].append(light)
        self._gltf.extensions = ext_obj
        used = self._gltf.extensionsUsed or []
        if "KHR_lights_punctual" not in used:
            used.append("KHR_lights_punctual")
            self._gltf.extensionsUsed = used

        light_idx = len(khr["lights"]) - 1
        translation = [float(position[0]), float(position[1]), float(position[2])]
        node = pygltflib.Node(
            translation=translation,
            extensions={"KHR_lights_punctual": {"light": light_idx}},
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
