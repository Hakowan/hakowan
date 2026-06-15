#!/usr/bin/env python

import argparse
import copy
import hakowan as hkw
from pathlib import Path
import math
import lagrange
import numpy as np
from PIL import Image
import uuid
import tempfile
import webbrowser
from tqdm import tqdm


_CLIP_AXES = {"x", "-x", "y", "-y", "z", "-z"}


def _clip_arg(value: str) -> tuple[str, float]:
    """Parse AXIS or AXIS:VALUE (e.g. 'x', '-y:0.3')."""
    if ":" in value:
        axis, val = value.rsplit(":", 1)
        try:
            fval = float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"clip value must be a number, got {val!r}"
            )
        if not (0.0 <= fval <= 1.0):
            raise argparse.ArgumentTypeError(
                f"clip value must be in [0, 1], got {fval}"
            )
    else:
        axis, fval = value, 0.5
    if axis not in _CLIP_AXES:
        raise argparse.ArgumentTypeError(
            f"clip axis must be one of {sorted(_CLIP_AXES)}, got {axis!r}"
        )
    return axis, fval


def _saturation_arg(value: str) -> float:
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"saturation must be a number, got {value!r}")
    if not math.isfinite(v) or v < 0:
        raise argparse.ArgumentTypeError(
            f"saturation must be non-negative and finite, got {v}"
        )
    return v


def _whiteness_arg(value: str) -> float:
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"whiteness must be a number, got {value!r}")
    if not math.isfinite(v) or not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"whiteness must be in [0, 1], got {v}")
    return v


def parse_args():
    """
    Parse command-line arguments for the mesh renderer.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Render a mesh.")
    parser.add_argument("--point-cloud", help="Render point cloud", action="store_true")
    parser.add_argument(
        "--point-size",
        help="Point size (absolute)",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--z-up", help="Use Z-up coordinate system.", action="store_true"
    )
    parser.add_argument("--comp", help="Visualize components", action="store_true")
    parser.add_argument(
        "--normal", help="Normal field", choices=["facet", "vertex"], default=None
    )
    parser.add_argument(
        "input_mesh",
        nargs="+",
        help="Input mesh file(s), up to 6, arranged in a grid (3 columns max).",
    )
    parser.add_argument("--camera", help="Camera location", nargs=3, type=float)
    parser.add_argument("-o", "--output", help="Output image file.")
    parser.add_argument(
        "-w", "--wireframe", help="Render wireframe", action="store_true"
    )
    parser.add_argument(
        "--wire-thickness",
        help="Wireframe/seam thickness relative to bbox diagonal",
        type=float,
        default=0.0005,
    )
    parser.add_argument(
        "--resolution", help="Resolution", nargs=2, type=int, default=(1024, 800)
    )
    parser.add_argument(
        "--material",
        default="plastic",
        help=(
            "Material types: plastic, roughplastic, glass, diffuse, uv, normal, texture, vertex_color. "
            "If none of these, it is treated as a scalar field attribute."
        ),
    )
    parser.add_argument("--color", help="Material color", type=str, default="ivory")
    parser.add_argument(
        "--back-color",
        help="Back-face color (enables two-sided rendering with a distinct back material).",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--orient-pca",
        action="store_true",
        help=(
            "Rotate and translate the mesh so PCA principal axes align with world axes: "
            "largest variance along +Y (default up), or +Z when --z-up is set; "
            "second/third variance along the remaining axes (right-handed). Applied before --rotate."
        ),
    )
    parser.add_argument(
        "--rotate", help="Rotate the mesh (degrees)", type=float, default=None
    )
    parser.add_argument(
        "--turn-table", help="Turn table animation (num samples)", type=int, default=0
    )
    parser.add_argument(
        "--two-sided", help="Render both sides of the mesh", action="store_true"
    )
    parser.add_argument("--seams", help="Render uv seams", action="store_true")
    parser.add_argument("--uv", help="Render uv map", action="store_true")
    parser.add_argument("--albedo", help="Render albedo", action="store_true")
    parser.add_argument("--depth", help="Render depth", action="store_true")
    parser.add_argument("--shading-normal", help="Render normal", action="store_true")
    parser.add_argument(
        "--facet-id",
        help=(
            "Render each facet colored by its index (RGB-encoded). "
            "No gamma correction, no blending, no anti-aliasing. "
            "Blender backend only."
        ),
        action="store_true",
    )
    parser.add_argument("--isoline", help="Render isoline", action="store_true")
    parser.add_argument(
        "--clip",
        help="Clip the mesh: AXIS or AXIS:VALUE where AXIS is x/y/z/-x/-y/-z and VALUE is a fraction of the bbox size in [0,1] (default 0.5). E.g. '--clip x' or '--clip -y:0.3'.",
        type=_clip_arg,
        default=None,
        metavar="AXIS[:VALUE]",
    )
    parser.add_argument("--singularity", help="Show singularity", action="store_true")
    streamline_group = parser.add_mutually_exclusive_group()
    streamline_group.add_argument(
        "--vector-field",
        metavar="ATTR",
        help="Overlay a plain vector field attribute (streamlines or arrows, see --field-style).",
        default=None,
    )
    streamline_group.add_argument(
        "--cross-field",
        metavar="ATTR",
        help="Overlay a 4-RoSy cross-field attribute (streamlines or arrows, see --field-style).",
        default=None,
    )
    parser.add_argument(
        "--field-style",
        choices=["streamline", "arrow"],
        default="streamline",
        help="How to visualize --vector-field / --cross-field: 'streamline' (default) or 'arrow'.",
    )

    parser.add_argument(
        "--num-streamlines",
        help="Number of streamlines to render (default: 500).",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--streamline-length",
        help="Max streamline half-length as a fraction of bbox diagonal (default: 0.5).",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--streamline-color",
        help="Streamline/arrow color (default black).",
        type=str,
        default="black",
    )
    parser.add_argument(
        "--quad-only", help="Only render quad facets", action="store_true"
    )
    parser.add_argument("--uv-scale", help="UV scale factor", type=float, default=1.0)
    parser.add_argument(
        "--categorical",
        help="Treat scalar attribute field as categorical (uses discrete colormap).",
        action="store_true",
    )
    parser.add_argument(
        "--saturation",
        help="Texture image saturation (1.0=full color, 0.0=grayscale, must be non-negative). Only applies when --material is a texture image.",
        type=_saturation_arg,
        default=1.0,
    )
    parser.add_argument(
        "--whiteness",
        help="Blend texture image toward pure white (0.0=original, 1.0=white, must be in [0,1]). Only applies when --material is a texture image.",
        type=_whiteness_arg,
        default=0.0,
    )
    parser.add_argument(
        "--backend",
        choices=hkw.list_backends(),
        default=None,
        help="Rendering backend to use (default: webgl)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="warning",
        help="Logging level",
    )
    parser.add_argument("--serialize", help="Serialize the config", action="store_true")
    parser.add_argument(
        "--no-open",
        help="Do not auto-open the output file in a browser (webgl backend only).",
        action="store_true",
    )
    parser.add_argument(
        "--camera-matrix",
        help=(
            "Save the world-to-camera transform and intrinsic matrix to this file "
            "(.npz or .json) for use in back-projecting pixels onto the mesh."
        ),
        default=None,
    )
    return parser.parse_args()


def node_to_layer(scene, node: lagrange.scene.Node, mats: list[hkw.material.Material]):
    """
    Recursively converts a scene node and its children into a composited Layer.

    Parameters:
        scene (lagrange.scene.Scene): The scene containing the nodes and meshes.
        node (lagrange.scene.Node): The current node to process.
        mats (list[hkw.material.Material]): List of materials used for mesh instances.

    Returns:
        hkw.layer.Layer or None: The composited Layer for this node and its children,
        or None if no layers are created.
    """
    layers = []
    if len(node.children) > 0:
        layers = [
            node_to_layer(scene, scene.nodes[child], mats) for child in node.children
        ]

    for mesh_instance in node.meshes:
        mesh = scene.meshes[mesh_instance.mesh]
        layer = hkw.layer(mesh)

        if len(mesh_instance.materials) == 1:
            mat = mats[mesh_instance.materials[0]]
            layer = layer.channel(material=mat)
        else:
            print(f"Zero, two or more materials: {len(mesh_instance.materials)}")

        layers.append(layer)

    if len(layers) > 0:
        return np.sum(layers).transform(hkw.transform.Affine(matrix=node.transform))
    else:
        return None


def get_tmp_image_name():
    """
    Generate a temporary file path for an image in the system's temporary directory.

    Returns:
        Path: A Path object pointing to a unique PNG file in the temp directory.
    """
    tmp_dir = Path(tempfile.gettempdir())
    return tmp_dir / f"{uuid.uuid4()}.png"


def extract_material(
    scene: lagrange.scene.Scene, saturation: float = 1.0, whiteness: float = 0.0
):
    """
    Extracts materials from a Lagrange scene and converts them to hakowan material objects.

    Parameters:
        scene (lagrange.scene.Scene): The scene object containing materials, textures, and images.

    Returns:
        list[hkw.material.Material]: A list of hakowan material objects corresponding to the scene's materials.
    """
    mats = []

    for mat_idx, material in enumerate(scene.materials):
        mat: hkw.material.Material
        tex_info = material.base_color_texture
        if tex_info.index is not None:
            tex = scene.textures[tex_info.index]
            if tex.image is not None:
                tex_img = scene.images[tex.image]
                tex_file = get_tmp_image_name()
                im = Image.fromarray(tex_img.image.data).convert("RGBA")
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
            if "diffuseTexture" in pbr and "index" in pbr["diffuseTexture"]:
                diffuse_tex_idx = pbr["diffuseTexture"]["index"]
            else:
                diffuse_tex_idx = -1
            if diffuse_tex_idx >= 0:
                diffuse_tex = scene.textures[diffuse_tex_idx]
                diffuse_img = scene.images[diffuse_tex.image]
                diffuse_file = diffuse_img.uri
                if diffuse_file is None:
                    diffuse_file = get_tmp_image_name()
                im = Image.fromarray(diffuse_img.image.data).convert("RGBA")
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
            mat = hkw.material.Diffuse(
                hkw.texture.Uniform(list(material.base_color_value[:3])),
                two_sided=True,
            )

        mats.append(mat)

    return mats


def embed_texture(scene_file, saturation: float = 1.0, whiteness: float = 0.0):
    """
    Loads a scene file, extracts its materials and textures, and constructs a composite layer representation.

    Supports any format accepted by ``lagrange.io.load_scene`` that carries
    material/texture data (e.g. GLB, GLTF, OBJ+MTL).

    Parameters:
        scene_file (str or Path): Path to the scene file to load.
        saturation (float): Saturation multiplier applied to all image textures.
        whiteness (float): Blend toward white applied to all image textures.

    Returns:
        hkw.layer.Layer: A composite layer object representing the scene with embedded textures.
    """
    scene = lagrange.io.load_scene(scene_file, stitch_vertices=True)
    mats = extract_material(scene, saturation=saturation, whiteness=whiteness)
    layers = [node_to_layer(scene, scene.nodes[nid], mats) for nid in scene.root_nodes]
    layers = [layer for layer in layers if layer is not None]
    assert len(layers) > 0, "No valid layers found in scene"
    return np.sum(layers)


def compute_camera_matrix(config) -> dict:
    """Compute the world-to-camera view matrix and camera intrinsics.

    Convention:
      - Camera space: right-handed, camera looks down **-Z**, Y is up.
      - Pixel space: (0, 0) at top-left, u increases right, v increases down.
      - Back-projection of pixel (u, v) to a camera-space ray direction:
            d_cam = normalize([(u - cx)/f, -(v - cy)/f, -1])
        then rotate to world space with the 3x3 upper-left block of view_matrix^-1
        (= view_matrix[:3,:3].T).

    Args:
        config: hakowan Config whose sensor/film are already fully configured.

    Returns:
        dict with keys:
          view_matrix      – 4x4 float64 world-to-camera transform
          intrinsic_matrix – 3x3 float64 K matrix (fx=fy, cx, cy)
          camera_location  – (3,) world-space eye position
          camera_target    – (3,) world-space look-at point
          camera_up        – (3,) orthogonalised up vector in world space
          fov_deg          – scalar FOV applied to the shorter image dimension
          width, height    – image dimensions (int32)
    """
    sensor = config.sensor
    eye = np.asarray(sensor.location, dtype=np.float64)
    target = np.asarray(getattr(sensor, "target", [0.0, 0.0, 0.0]), dtype=np.float64)
    up_hint = np.asarray(getattr(sensor, "up", [0.0, 1.0, 0.0]), dtype=np.float64)

    # Orthonormal camera basis
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)  # guaranteed orthogonal to both

    # 4x4 view matrix  (world → camera, camera looks down -Z)
    # [ right  | -dot(right, eye) ]
    # [ up     | -dot(up, eye)    ]
    # [ -fwd   |  dot(fwd, eye)   ]
    # [ 0 0 0  |  1               ]
    view = np.array(
        [
            [right[0], right[1], right[2], -np.dot(right, eye)],
            [up[0], up[1], up[2], -np.dot(up, eye)],
            [-fwd[0], -fwd[1], -fwd[2], np.dot(fwd, eye)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Intrinsic matrix K  (FOV applies to the shorter image dimension)
    width = config.film.width
    height = config.film.height
    fov_deg = float(getattr(sensor, "fov", 28.8415))
    f = (min(width, height) / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    return {
        "view_matrix": view,
        "intrinsic_matrix": K,
        "camera_location": eye,
        "camera_target": target,
        "camera_up": up,
        "fov_deg": np.float64(fov_deg),
        "width": np.int32(width),
        "height": np.int32(height),
    }


def _orient_pca_target_frame(z_up: bool) -> np.ndarray:
    """Columns: world directions for (major, mid, minor) PCA axes (right-handed)."""
    if z_up:
        # major -> +Z, mid -> +X, minor -> +Y
        return np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64
        )
    # major -> +Y, mid -> +Z, minor -> +X (Y-up)
    return np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )


def save_camera_matrix(data: dict, output_path: Path):
    """Save camera matrix data produced by compute_camera_matrix.

    Supported formats (determined by file extension):
      .npz  – NumPy compressed archive (default / recommended)
      .json – JSON for interoperability with non-Python tools
    """
    suffix = output_path.suffix.lower()
    if suffix == ".json":
        import json

        def _to_python(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            return v

        output_path.write_text(
            json.dumps({k: _to_python(v) for k, v in data.items()}, indent=2)
        )
    else:
        # Ensure the file has the .npz extension that numpy requires.
        out = (
            output_path
            if output_path.suffix.lower() == ".npz"
            else output_path.with_suffix(".npz")
        )
        np.savez(str(out), **data)


def _back_side_material(material_type: str, back_color: str) -> "hkw.material.Material":
    """Build a back-face material matching the front material type but with back_color."""
    match material_type:
        case "diffuse":
            return hkw.material.Diffuse(back_color)
        case "roughplastic":
            return hkw.material.RoughPlastic(back_color)
        case "plastic":
            return hkw.material.Plastic(back_color)
        case _:
            hkw.logger.warning(
                "--back-color is not supported for material type %r; "
                "back-face will use Plastic material instead.",
                material_type,
            )
            return hkw.material.Plastic(back_color)


def build_layer(args, mesh_path: str, normalize: bool = False) -> "hkw.layer.Layer":
    """Build the styled layer for a single input mesh.

    Holds all per-mesh logic (geometry load, material, overlays, transforms) so
    that multiple meshes can be built independently and laid out side by side.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.
        mesh_path (str): Path to the input mesh file.
        normalize (bool): Scale the result to unit bounding-sphere radius so
            meshes from unrelated coordinate systems show at the same on-screen
            size. Used when laying out multiple meshes in a grid.

    Returns:
        hkw.layer.Layer: The fully styled layer for this mesh.
    """
    mesh = lagrange.io.load_mesh(mesh_path, quiet=True, stitch_vertices=True)

    if args.quad_only:
        # Filter to only keep quad facets
        facets_to_keep = []
        for fid in range(mesh.num_facets):
            if mesh.get_facet_size(fid) == 4:
                facets_to_keep.append(fid)

        if len(facets_to_keep) == 0:
            raise ValueError("No quad facets found in mesh")

        mesh = lagrange.extract_submesh(mesh, np.array(facets_to_keep))

    bbox_min = np.amin(mesh.vertices, axis=0)
    bbox_max = np.amax(mesh.vertices, axis=0)
    bbox_size = bbox_max - bbox_min
    bbox_diag = np.linalg.norm(bbox_size)

    layer: hkw.layer.Layer = hkw.layer(mesh)

    back_side = (
        _back_side_material(args.material, args.back_color)
        if args.back_color is not None
        else None
    )
    two_sided = args.two_sided or (args.back_color is not None)

    match args.material:
        case "diffuse":
            layer = layer.material(
                "Diffuse", args.color, two_sided=two_sided, back_side=back_side
            )
        case "plastic":
            layer = layer.material(
                "Plastic", args.color, two_sided=two_sided, back_side=back_side
            )
        case "roughplastic":
            layer = layer.material(
                "RoughPlastic", args.color, two_sided=two_sided, back_side=back_side
            )
        case "glass":
            layer = layer.material("ThinDielectric", specular_reflectance=0.5)
        case "texture":
            layer = embed_texture(
                mesh_path, saturation=args.saturation, whiteness=args.whiteness
            )
        case "vertex_color":
            color_attr_ids = mesh.get_matching_attribute_ids(
                usage=lagrange.AttributeUsage.Color
            )
            assert len(color_attr_ids) > 0, (
                "No color attributes found in mesh for vertex_color material"
            )
            color_attr_id = color_attr_ids[0]
            color_attr_name = mesh.get_attribute_name(color_attr_id)
            layer = layer.material(
                "Principled",
                hkw.texture.ScalarField(color_attr_name, colormap="identity"),
                two_sided=two_sided,
            )
        case "normal":
            normal_attr_ids = mesh.get_matching_attribute_ids(
                usage=lagrange.AttributeUsage.Normal
            )
            if len(normal_attr_ids) == 0:
                normal_attr_id = lagrange.compute_vertex_normal(mesh)
            else:
                normal_attr_id = normal_attr_ids[0]
                if mesh.is_attribute_indexed(normal_attr_id):
                    normal_attr_id = lagrange.map_attribute(
                        mesh,
                        normal_attr_id,
                        "_vertex_normal",
                        lagrange.AttributeElement.Vertex,
                    )
            normal_attr = mesh.attribute(normal_attr_id)
            if normal_attr.element_type != lagrange.AttributeElement.Vertex:
                normal_attr_id = lagrange.map_attribute(
                    mesh,
                    normal_attr_id,
                    "_vertex_normal",
                    lagrange.AttributeElement.Vertex,
                )
                normal_attr = mesh.attribute(normal_attr_id)

            normal_attr_name = mesh.get_attribute_name(normal_attr_id)

            def normal_to_color(n):
                return (np.array([-n[0], n[2], n[1]]) + 1) / 2

            scale = hkw.scale.Custom(normal_to_color)
            layer = layer.material(
                "Diffuse",
                hkw.texture.ScalarField(
                    hkw.attribute(normal_attr_name, scale=scale), colormap="identity"
                ),
                two_sided=two_sided,
            )
        case "uv":
            uv_ids = mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV)
            assert len(uv_ids) > 0, "No UV attributes found in mesh for uv material"
            uv_id = uv_ids[0]
            uv_name = mesh.get_attribute_name(uv_id)
            base = hkw.layer(mesh)
            surface = base.material(
                "Principled",
                hkw.texture.Checkerboard(
                    uv=hkw.attribute(uv_name, scale=args.uv_scale), size=64
                ),
            )
            layer = surface
        case _:
            if mesh.has_attribute(args.material):
                scalar_texture = hkw.texture.ScalarField(
                    args.material,
                    categories=args.categorical,
                    colormap="set1" if args.categorical else "viridis",
                )

                if args.isoline:
                    solid_color = hkw.texture.Uniform(0)
                    isoline_texture = hkw.texture.Isocontour(
                        data=hkw.attribute(
                            args.material,
                            scale=hkw.scale.Normalize(range_min=0, range_max=1),
                        ),
                        ratio=0.9,
                        texture1=scalar_texture,
                        texture2=solid_color,
                        num_contours=32,
                    )
                    layer = layer.material(
                        "Principled",
                        color=isoline_texture,
                        two_sided=two_sided,
                    )
                else:
                    layer = layer.material(
                        "Principled",
                        color=scalar_texture,
                        two_sided=two_sided,
                    )
            else:
                texture_file = Path(args.material)
                assert texture_file.is_file(), f"Texture file {texture_file} not found"
                layer = layer.material(
                    "Principled",
                    hkw.texture.Image(
                        texture_file,
                        saturation=args.saturation,
                        whiteness=args.whiteness,
                    ),
                )

    if args.point_cloud:
        assert not args.comp, "--point-cloud and --comp options are mutually exclusive"
        layer = layer.mark("Point").channel(size=args.point_size)

    if args.comp:
        layer = layer.transform(hkw.transform.Compute(component="comp"))
        layer = layer.material(
            "Principled",
            hkw.texture.ScalarField("comp", colormap="set1", categories=True),
        )

    if args.normal == "vertex":
        layer = layer.transform(hkw.transform.Compute(vertex_normal="vertex_normal"))
        layer = layer.channel(normal="vertex_normal")
    elif args.normal == "facet":
        layer = layer.transform(hkw.transform.Compute(facet_normal="face_normal"))
        layer = layer.channel(normal="face_normal")

    if args.uv:
        layer = layer.transform(hkw.transform.UVMesh())

    if args.wireframe:
        w = layer.mark("Curve").material("Diffuse", "black")
        w = w.channel(size=args.wire_thickness * bbox_diag)
        layer = layer + w

    if args.seams:
        uv_ids = mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV)
        assert len(uv_ids) > 0, "No UV attributes found in mesh for seams rendering"
        uv_id = uv_ids[0]
        uv_name = mesh.get_attribute_name(uv_id)
        base = hkw.layer(mesh)
        seams = (
            base.transform(hkw.transform.Boundary(attributes=[uv_name]))
            .mark("Curve")
            .material("Diffuse", "black")
            .channel(size=args.wire_thickness * bbox_diag)
        )
        layer = layer + seams

    vec_field_attr = args.vector_field or args.cross_field
    if vec_field_attr is not None:
        is_cross = args.cross_field is not None
        vf_mesh = copy.deepcopy(mesh)
        lagrange.triangulate_polygonal_facets(vf_mesh)
        if args.field_style == "arrow":
            vf_layer = (
                hkw.layer(vf_mesh)
                .mark("Curve")
                .material("Diffuse", args.streamline_color)
                .channel(
                    vector_field=hkw.channel.VectorField(
                        data=vec_field_attr,
                        end_type="arrow",
                        normalize=True,
                    ),
                    size=args.wire_thickness * bbox_diag,
                )
            )
        else:
            vf_layer = (
                hkw.layer(vf_mesh)
                .transform(
                    hkw.transform.Streamline(
                        vec_field=vec_field_attr,
                        cross_field=is_cross,
                        n=args.num_streamlines,
                        length=args.streamline_length * bbox_diag
                        if args.streamline_length is not None
                        else None,
                    )
                )
                .mark("Curve")
                .material("Diffuse", args.streamline_color)
                .channel(size=args.wire_thickness * bbox_diag)
            )
        layer = layer + vf_layer

    if args.singularity:
        lagrange.compute_vertex_valence(mesh, output_attribute_name="valence")
        num_triangles = 0
        num_quads = 0
        for fid in range(mesh.num_facets):
            fsize = mesh.get_facet_size(fid)
            if fsize == 3:
                num_triangles += 1
            elif fsize == 4:
                num_quads += 1

        if mesh.is_triangle_mesh or num_triangles > 0.9 * mesh.num_facets:
            triangle_dominant = True
            quad_dominant = False
        elif mesh.is_quad_mesh or num_quads > 0.9 * mesh.num_facets:
            triangle_dominant = False
            quad_dominant = True
        else:
            # Mixed mesh - default to triangle mode
            triangle_dominant = True
            quad_dominant = False

        base = hkw.layer(mesh)
        valence_view = base.mark("Point").channel(size=0.005 * bbox_diag)

        if triangle_dominant:
            valence_view = valence_view.transform(
                hkw.transform.Filter("valence", lambda d: d != 6)
            )
            colormap = ["#C77DDB", "#E68445", "#27A6DE", "#FFC24F", "#FF0046"]
        elif quad_dominant:
            valence_view = valence_view.transform(
                hkw.transform.Filter("valence", lambda d: d != 4)
            )
            colormap = ["#27A6DE", "#FFC24F", "#FF0046", "#C77DDB", "#E68445"]
        else:
            raise NotImplementedError(
                "Singularity visualization for mixed meshes is not implemented"
            )

        valence_view = valence_view.material(
            "Principled",
            hkw.texture.ScalarField(
                "valence",
                colormap=colormap,
                domain=[3, 7],
                categories=True,
            ),
            roughness=0.0,
            metallic=0.2,
        )

        layer = layer + valence_view

    if args.orient_pca:
        layer = layer.transform(
            hkw.transform.PrincipalAxes(frame=_orient_pca_target_frame(args.z_up))
        )

    if args.rotate:
        if args.z_up:
            axis = [0, 0, 1]
        else:
            axis = [0, 1, 0]
        layer = layer.rotate(axis=axis, angle=args.rotate * math.pi / 180)

    if args.clip is not None:
        clip_axis, clip_value = args.clip
        match clip_axis:
            case "x":

                def condition(x):
                    return x[0] - bbox_min[0] >= clip_value * bbox_size[0]

            case "-x":

                def condition(x):
                    return x[0] - bbox_min[0] <= clip_value * bbox_size[0]

            case "y":

                def condition(x):
                    return x[1] - bbox_min[1] >= clip_value * bbox_size[1]

            case "-y":

                def condition(x):
                    return x[1] - bbox_min[1] <= clip_value * bbox_size[1]

            case "z":

                def condition(x):
                    return x[2] - bbox_min[2] >= clip_value * bbox_size[2]

            case "-z":

                def condition(x):
                    return x[2] - bbox_min[2] <= clip_value * bbox_size[2]

            case _:
                raise ValueError("Invalid clip axis")
        layer = layer.transform(hkw.transform.Filter(condition=condition))

    if normalize:
        # Outermost transform (applied after clip/rotate) so the styled result —
        # mesh and any overlays alike — fits a unit bounding sphere centered at the
        # origin. This is what makes every cell, including a lone mesh in a ragged
        # grid row, render at the same on-screen size regardless of source scale.
        layer = layer.transform(hkw.transform.Normalize())

    return layer


def grid_layout(layers: list["hkw.layer.Layer"], up_axis: str) -> "hkw.layer.Layer":
    """Arrange ``layers`` in a grid (at most 3 columns) facing the camera.

    Columns are laid out along X; rows are stacked along ``up_axis`` (the screen
    vertical, so the grid faces the camera). Column count is ``min(N, 3)``, so up
    to 3 meshes share a single row and 4-6 meshes fill a 2-row grid (6 meshes give
    a 2x3 grid). Rows are emitted top-to-bottom (the first mesh sits top-left). A
    ragged last row centers itself for free, courtesy of the recursive
    juxtaposition layout in the compiler.

    Parameters:
        layers (list[hkw.layer.Layer]): One styled layer per input mesh.
        up_axis (str): Screen-vertical axis to stack rows along ("y" or "z").

    Returns:
        hkw.layer.Layer: The composited grid layer (or the lone layer if N == 1).
    """
    if len(layers) == 1:
        return layers[0]

    # ``layers`` are pre-normalized to unit size by ``build_layer`` (see the
    # ``normalize`` path), so every cell — including a lone mesh in a ragged
    # row — is already the same size. The juxtaposition just packs them; no
    # per-cell ``normalize`` is needed (and using it would re-shrink whole rows
    # unevenly when row counts differ, e.g. a full row of 3 vs a ragged 2).
    cols = min(len(layers), 3)
    rows = [layers[i : i + cols] for i in range(0, len(layers), cols)]
    row_layers = [
        row[0] if len(row) == 1 else row[0].juxtapose(*row[1:], axis="x")
        for row in rows
    ]
    # Stack rows top-to-bottom: juxtapose packs in increasing-axis order, so
    # reverse the rows to place the first one at the top.
    row_layers.reverse()
    if len(row_layers) == 1:
        return row_layers[0]
    return row_layers[0].juxtapose(*row_layers[1:], axis=up_axis)


def main():
    """
    Entry point for the command-line interface for mesh rendering.
    Parses command-line arguments and orchestrates the mesh rendering process.
    """
    args = parse_args()

    if len(args.input_mesh) > 6:
        raise SystemExit(
            f"At most 6 input meshes are supported, got {len(args.input_mesh)}."
        )

    # Resolve the effective backend early so string comparisons below are safe.
    if args.backend is None:
        from hakowan.backends import resolve_backend_name

        args.backend = resolve_backend_name()

    hkw.logger.setLevel(args.log_level.upper())
    lagrange.logger.setLevel(args.log_level.upper())

    # Build one styled layer per input mesh; multiple meshes are arranged in a
    # near-square grid facing the camera (rows stacked along the up-axis). With
    # more than one mesh, normalize each to unit size so they show at the same
    # on-screen scale despite unrelated source coordinate systems.
    normalize = len(args.input_mesh) > 1
    layers = [
        build_layer(args, mesh_path, normalize=normalize)
        for mesh_path in args.input_mesh
    ]
    layer = grid_layout(layers, up_axis="z" if args.z_up else "y")

    config = hkw.config()
    [config.film.width, config.film.height] = args.resolution
    if args.z_up:
        config.z_up()
    if args.camera is not None:
        config.sensor.location = args.camera

    if args.albedo:
        config.albedo = True
    if args.depth:
        config.depth = True
    if args.shading_normal:
        config.normal = True
    if args.facet_id:
        config.facet_id = True

    if args.output:
        output_file = Path(args.output)
    else:
        default_suffix = ".html" if args.backend == "webgl" else ".png"
        output_file = Path(args.input_mesh[0]).with_suffix(default_suffix)

    if args.camera_matrix:
        cam_data = compute_camera_matrix(config)
        save_camera_matrix(cam_data, Path(args.camera_matrix))

    if args.turn_table == 0:
        kwargs = {}
        if args.serialize:
            kwargs["yaml_file"] = output_file.with_suffix(".yaml")
            if args.backend == "blender":
                kwargs["blend_file"] = output_file.with_suffix(".blend")
        result = hkw.render(
            layer,
            config,
            filename=output_file,
            backend=args.backend,
            **kwargs,
        )
        if args.backend == "webgl" and not args.no_open:
            # The webgl backend always lands at <stem>.html (even when the
            # user passed a .png filename, render() rewrites the suffix).
            # Trust the backend's returned path when available.
            opened_path = (
                Path(result)
                if isinstance(result, (str, Path))
                else (output_file.with_suffix(".html"))
            )
            webbrowser.open(opened_path.resolve().as_uri())
    else:
        if args.z_up:
            axis = np.array([0, 0, 1], dtype=float)
        else:
            axis = np.array([0, 1, 0], dtype=float)

        # Get initial camera position in normalized coordinates
        # (object is at origin with bounding radius 1)
        initial_camera = np.array(config.sensor.location, dtype=float)

        # Render frames to temporary files
        frames = []
        temp_files = []
        try:
            for i in tqdm(
                range(args.turn_table), desc="Rendering frames", unit="frame"
            ):
                # Rotate camera position around origin in normalized coordinates
                angle = i * 2 * math.pi / args.turn_table
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)

                # Rodrigues' rotation formula to rotate camera position around axis
                rotated_camera = (
                    initial_camera * cos_a
                    + np.cross(axis, initial_camera) * sin_a
                    + axis * np.dot(axis, initial_camera) * (1 - cos_a)
                )

                # Update camera position for this frame
                frame_config = hkw.config()
                frame_config.film.width = config.film.width
                frame_config.film.height = config.film.height
                if args.z_up:
                    frame_config.z_up()
                frame_config.sensor.location = list(rotated_camera)
                frame_config.sensor.target = [0.0, 0.0, 0.0]  # Look at origin

                if args.albedo:
                    frame_config.albedo = True
                if args.depth:
                    frame_config.depth = True
                if args.shading_normal:
                    frame_config.normal = True
                if args.facet_id:
                    frame_config.facet_id = True

                frame_file = get_tmp_image_name()
                temp_files.append(frame_file)
                hkw.render(
                    layer, frame_config, filename=frame_file, backend=args.backend
                )
                # Load frame and ensure solid white background
                with Image.open(frame_file) as img:
                    if img.mode == "RGBA":
                        # Create a white background
                        white_bg = Image.new("RGB", img.size, (255, 255, 255))
                        white_bg.paste(
                            img, mask=img.split()[3]
                        )  # Use alpha channel as mask
                        frames.append(white_bg)
                    else:
                        frames.append(img.convert("RGB"))

            # Save as animated GIF
            gif_file = output_file.with_suffix(".gif")
            frames[0].save(
                gif_file,
                save_all=True,
                append_images=frames[1:],
                duration=100,  # milliseconds per frame
                loop=0,
            )
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
