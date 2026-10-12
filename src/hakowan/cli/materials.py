"""Import materials and image textures from Lagrange scenes."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any

import hakowan as hkw
import lagrange
from PIL import Image

from ..grammar.channel.material import Material
from ..grammar.layer import Layer


def _combine_layers(layers: list[Layer]) -> Layer:
    combined = layers[0]
    for child_layer in layers[1:]:
        combined = combined + child_layer
    return combined


def node_to_layer(
    scene: lagrange.scene.Scene,
    node: lagrange.scene.Node,
    mats: list[Material],
) -> Layer | None:
    """
    Recursively converts a scene node and its children into a composited Layer.

    Parameters:
        scene (lagrange.scene.Scene): The scene containing the nodes and meshes.
        node (lagrange.scene.Node): The current node to process.
        mats (list[hkw.material.Material]): List of materials used for mesh instances.

    Returns:
        hkw.layer or None: The composited Layer for this node and its children,
        or None if no layers are created.
    """
    layers: list[Layer] = []
    for child_id in node.children:
        child_layer = node_to_layer(scene, scene.nodes[child_id], mats)
        if child_layer is not None:
            layers.append(child_layer)

    for mesh_instance in node.meshes:
        mesh_id = mesh_instance.mesh
        if mesh_id is None:
            raise ValueError("Scene mesh instance does not reference a mesh.")
        mesh = scene.meshes[mesh_id]
        layer = hkw.layer(mesh)

        if len(mesh_instance.materials) == 1:
            mat = mats[mesh_instance.materials[0]]
            layer = layer.channel(material=mat)
        else:
            print(f"Zero, two or more materials: {len(mesh_instance.materials)}")

        layers.append(layer)

    if not layers:
        return None
    return _combine_layers(layers).transform(
        hkw.transform.Affine(matrix=node.transform)
    )


def get_tmp_image_name() -> Path:
    """
    Generate a temporary file path for an image in the system's temporary directory.

    Returns:
        Path: A Path object pointing to a unique PNG file in the temp directory.
    """
    tmp_dir = Path(tempfile.gettempdir())
    return tmp_dir / f"{uuid.uuid4()}.png"


def _base_color_diffuse(material: Any) -> Material:
    """Fallback material using the material's constant base color factor (RGB)."""
    return hkw.material.Diffuse(
        hkw.texture.Uniform(list(material.base_color_value[:3])),
        two_sided=True,
    )


def extract_material(
    scene: lagrange.scene.Scene,
    saturation: float = 1.0,
    whiteness: float = 0.0,
) -> list[Material]:
    """Convert Lagrange scene materials to Hakowan materials.

    Args:
        scene: Scene containing materials, textures, and embedded images.
        saturation: Saturation multiplier applied to extracted image textures.
        whiteness: Fraction used to blend extracted image textures toward white.

    Returns:
        Hakowan material objects in source-scene order.

    """
    mats = []

    for mat_idx, material in enumerate(scene.materials):
        mat: hkw.material.Material
        tex_info = material.base_color_texture
        if tex_info.index is not None:
            tex = scene.textures[tex_info.index]
            if tex.image is not None:
                tex_img = scene.images[tex.image]
                buf = tex_img.image
                if buf.width == 0 or buf.height == 0:
                    # Texture is referenced but its image data was not loaded
                    # (common with FBX assets whose textures live in external
                    # files Lagrange could not resolve). Fall back to the
                    # material's constant base color instead of saving an empty
                    # image (which crashes PIL with "tile cannot extend outside
                    # image").
                    print(
                        f"Warning: Material[{mat_idx}] '{material.name}': base "
                        f"color texture {tex_info.index} has empty image data; "
                        f"falling back to constant base color."
                    )
                    mat = _base_color_diffuse(material)
                else:
                    tex_file = get_tmp_image_name()
                    im = Image.fromarray(buf.data).convert("RGBA")
                    im.save(str(tex_file))
                    mat = hkw.material.Principled(
                        color=hkw.texture.Image(
                            Path(tex_file), saturation=saturation, whiteness=whiteness
                        ),
                        roughness=0.5,
                        metallic=0.0,
                        two_sided=True,
                    )
            else:
                raise ValueError(
                    f"Material[{mat_idx}] '{material.name}': texture index "
                    f"{tex_info.index} has no associated image data."
                )
        elif "KHR_materials_pbrSpecularGlossiness" in material.extensions.data:
            pbr = material.extensions.data["KHR_materials_pbrSpecularGlossiness"]
            diffuse_texture = (
                pbr.get("diffuseTexture") if isinstance(pbr, dict) else None
            )
            diffuse_tex_idx = None
            if isinstance(diffuse_texture, dict):
                raw_index = diffuse_texture.get("index")
                if isinstance(raw_index, int) and not isinstance(raw_index, bool):
                    diffuse_tex_idx = raw_index

            if diffuse_tex_idx is not None and diffuse_tex_idx >= 0:
                diffuse_tex = scene.textures[diffuse_tex_idx]
                if diffuse_tex.image is None:
                    raise ValueError("KHR texture has no associated image data.")
                diffuse_img = scene.images[diffuse_tex.image]
                buf = diffuse_img.image
                if buf.width == 0 or buf.height == 0:
                    raise ValueError("KHR texture image data is empty.")
                diffuse_file = get_tmp_image_name()
                im = Image.fromarray(buf.data).convert("RGBA")
                im.save(str(diffuse_file))
                mat = hkw.material.Principled(
                    color=hkw.texture.Image(
                        diffuse_file, saturation=saturation, whiteness=whiteness
                    ),
                    roughness=0.5,
                    metallic=0.0,
                    two_sided=True,
                )
            else:
                raise ValueError("KHR texture not found")
        else:
            mat = _base_color_diffuse(material)

        mats.append(mat)

    return mats


def embed_texture(
    scene_file: str | Path,
    saturation: float = 1.0,
    whiteness: float = 0.0,
) -> Layer:
    """
    Loads a scene file, extracts its materials and textures, and constructs a composite layer representation.

    Supports any format accepted by ``lagrange.io.load_scene`` that carries
    material/texture data (e.g. GLB, GLTF, OBJ+MTL).

    Parameters:
        scene_file (str or Path): Path to the scene file to load.
        saturation (float): Saturation multiplier applied to all image textures.
        whiteness (float): Blend toward white applied to all image textures.

    Returns:
        hkw.layer: A composite layer object representing the scene with embedded textures.
    """
    scene = lagrange.io.load_scene(scene_file, stitch_vertices=True)
    mats = extract_material(scene, saturation=saturation, whiteness=whiteness)
    layers: list[Layer] = []
    for node_id in scene.root_nodes:
        layer = node_to_layer(scene, scene.nodes[node_id], mats)
        if layer is not None:
            layers.append(layer)
    if not layers:
        raise SystemExit("No valid layers found in scene")
    return _combine_layers(layers)
