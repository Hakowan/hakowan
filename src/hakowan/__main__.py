#!/usr/bin/env python

import argparse
import hakowan as hkw
from pathlib import Path
import math
import lagrange
import numpy as np
from PIL import Image
import uuid
import tempfile


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
    parser.add_argument("input_mesh", help="Input mesh file.")
    parser.add_argument("--camera", help="Camera location", nargs=3, type=float)
    parser.add_argument("-o", "--output", help="Output image file.")
    parser.add_argument(
        "-w", "--wireframe", help="Render wireframe", action="store_true"
    )
    parser.add_argument(
        "--wire-thickness",
        help="Wireframe/seam thickness relative to bbox diagonal",
        type=float,
        default=0.001,
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
    parser.add_argument("--isoline", help="Render isoline", action="store_true")
    parser.add_argument(
        "--clip",
        help="Clip the mesh along a coordinate axis",
        type=str,
        default=None,
        choices=["x", "y", "z", "-x", "-y", "-z"],
    )
    parser.add_argument(
        "--clip-value",
        help="Clip value as percentage of bbox size",
        type=float,
        default=0.5,
    )
    parser.add_argument("--singularity", help="Show singularity", action="store_true")
    parser.add_argument("--uv-scale", help="UV scale factor", type=float, default=1.0)
    parser.add_argument(
        "--backend",
        choices=hkw.list_backends(),
        default="mitsuba",
        help="Rendering backend to use",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="warn",
        help="Logging level",
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


def extract_material(scene: lagrange.scene.Scene):
    """
    Extracts materials from a Lagrange scene and converts them to hakowan material objects.

    Parameters:
        scene (lagrange.scene.Scene): The scene object containing materials, textures, and images.

    Returns:
        list[hkw.material.Material]: A list of hakowan material objects corresponding to the scene's materials.
    """
    mats = []

    for material in scene.materials:
        mat: hkw.material.Material
        tex_info = material.base_color_texture
        if tex_info.index is not None:
            tex = scene.textures[tex_info.index]
            if tex.image is not None:
                tex_img = scene.images[tex.image]
                tex_file = get_tmp_image_name()
                im = Image.fromarray(tex_img.image.data, "RGBA")
                im.save(str(tex_file))
                mat = hkw.material.Principled(
                    color=hkw.texture.Image(Path(tex_file)),
                    roughness=0.5,
                    metallic=0.0,
                    two_sided=True,
                )
            else:
                raise ValueError("Texture image not found")
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
                im = Image.fromarray(diffuse_img.image.data, "RGBA")
                im.save(str(diffuse_file))
                mat = hkw.material.Principled(
                    color=hkw.texture.Image(diffuse_file),
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


def embed_texture(glb_file):
    """
    Loads a GLB file, extracts its materials and textures, and constructs a composite layer representation.

    Parameters:
        glb_file (str or Path): Path to the GLB file to load.

    Returns:
        hkw.layer.Layer: A composite layer object representing the scene with embedded textures.
    """
    scene = lagrange.io.load_scene(glb_file, stitch_vertices=True)
    mats = extract_material(scene)
    layers = [node_to_layer(scene, scene.nodes[nid], mats) for nid in scene.root_nodes]
    layers = [layer for layer in layers if layer is not None]
    assert len(layers) > 0, "No valid layers found in scene"
    return np.sum(layers)


def main():
    """
    Entry point for the command-line interface for mesh rendering.
    Parses command-line arguments and orchestrates the mesh rendering process.
    """
    args = parse_args()

    hkw.logger.setLevel(args.log_level.upper())
    lagrange.logger.setLevel(args.log_level.upper())

    mesh = lagrange.io.load_mesh(args.input_mesh, quiet=True, stitch_vertices=True)
    bbox_min = np.amin(mesh.vertices, axis=0)
    bbox_max = np.amax(mesh.vertices, axis=0)
    bbox_size = bbox_max - bbox_min
    bbox_diag = np.linalg.norm(bbox_size)

    layer: hkw.layer.Layer = hkw.layer(mesh)

    match args.material:
        case "diffuse":
            layer = layer.material("Diffuse", args.color, two_sided=args.two_sided)
        case "plastic":
            layer = layer.material("Plastic", args.color, two_sided=args.two_sided)
        case "roughplastic":
            layer = layer.material("RoughPlastic", args.color, two_sided=args.two_sided)
        case "glass":
            layer = layer.material("ThinDielectric", specular_reflectance=0.1)
        case "texture":
            assert Path(args.input_mesh).suffix in [".glb", ".gltf"], (
                f"Texture material requires .glb or .gltf file, got {Path(args.input_mesh).suffix}"
            )
            layer = embed_texture(args.input_mesh)
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
                two_sided=args.two_sided,
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
                two_sided=args.two_sided,
            )
        case "uv":
            mesh = lagrange.unify_index_buffer(mesh)
            diag_len = bbox_diag
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
            boundary = (
                base.transform(hkw.transform.Boundary())
                .mark("Curve")
                .material("Diffuse", "black")
                .channel(size=0.001 * diag_len)
            )
            layer = surface + boundary
        case _:
            if mesh.has_attribute(args.material):
                scalar_texture = hkw.texture.ScalarField(args.material)

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
                        two_sided=args.two_sided,
                    )
                else:
                    layer = layer.material(
                        "Principled",
                        color=scalar_texture,
                        two_sided=args.two_sided,
                    )
            else:
                texture_file = Path(args.material)
                assert texture_file.is_file(), f"Texture file {texture_file} not found"
                layer = layer.material("Principled", hkw.texture.Image(texture_file))

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
        mesh = lagrange.unify_index_buffer(mesh)
        uv_ids = mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV)
        assert len(uv_ids) > 0, "No UV attributes found in mesh for seams rendering"
        uv_id = uv_ids[0]
        base = hkw.layer(mesh)
        seams = (
            base.transform(hkw.transform.Boundary())
            .mark("Curve")
            .material("Diffuse", "black")
            .channel(size=args.wire_thickness * bbox_diag)
        )
        layer = layer + seams

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

    if args.rotate:
        if args.z_up:
            axis = [0, 0, 1]
        else:
            axis = [0, 1, 0]
        layer = layer.rotate(axis=axis, angle=args.rotate * math.pi / 180)

    if args.clip is not None:
        clip_value = args.clip_value
        match args.clip:
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

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(args.input_mesh).with_suffix(".png")

    if args.turn_table == 0:
        hkw.render(
            layer,
            config,
            filename=output_file,
            backend=args.backend,
            blend_file="debug.blend",
        )
    else:
        if args.z_up:
            axis = [0, 0, 1]
        else:
            axis = [0, 1, 0]

        for i in range(args.turn_table):
            frame = layer.rotate(axis=axis, angle=i * 2 * math.pi / args.turn_table)
            frame_file = output_file.with_stem(f"{output_file.stem}_{i:03d}")
            hkw.render(frame, config, filename=frame_file, backend=args.backend)


if __name__ == "__main__":
    main()
