"""Build styled Hakowan layers from parsed CLI options."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import hakowan as hkw
import lagrange
import numpy as np

from ..grammar.channel.material import Material
from ..grammar.layer import Layer
from .arguments import RenderArgs
from .camera import _orient_pca_target_frame
from .materials import embed_texture


def _back_side_material(material_type: str, back_color: str) -> Material:
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


def build_layer(
    args: RenderArgs,
    mesh_path: str | Path,
    normalize: bool = False,
    mesh: lagrange.SurfaceMesh | None = None,
) -> Layer:
    """Build the styled layer for a single input mesh.

    Holds all per-mesh logic (geometry load, material, overlays, transforms) so
    that multiple meshes can be built independently and laid out side by side.

    Parameters:
        args: Parsed command-line options.
        mesh_path: Path to the input mesh file.
        normalize: Scale the result to unit bounding-sphere radius so
            meshes from unrelated coordinate systems show at the same on-screen
            size. Used when laying out multiple meshes in a grid.
        mesh: Preloaded mesh. When omitted, load mesh_path.

    Returns:
        The fully styled layer for this mesh.
    """
    if mesh is None:
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

    layer: Layer = hkw.layer(mesh)

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
            if not color_attr_ids:
                raise SystemExit(
                    "No color attributes found in mesh for vertex_color material"
                )
            color_attr_id = color_attr_ids[0]
            if mesh.is_attribute_indexed(color_attr_id):
                color_attr_id = lagrange.map_attribute(
                    mesh,
                    color_attr_id,
                    "_vertex_color",
                    lagrange.AttributeElement.Vertex,
                )
            color_attr = mesh.attribute(color_attr_id)
            color_data = np.asarray(color_attr.data)
            color_attr_name = mesh.get_attribute_name(color_attr_id)
            # Integer color attributes (e.g. uint8 [0,255] from PCD/PLY) must be
            # normalized to the [0,1] float range the renderer expects; the
            # identity colormap passes values through unchanged, so 0-255 would
            # otherwise blow past 1.0 and clamp every point to white.
            if np.issubdtype(color_data.dtype, np.integer):
                color_attr_name = "_vertex_color_normalized"
                mesh.create_attribute(
                    color_attr_name,
                    element=color_attr.element_type,
                    usage=lagrange.AttributeUsage.Color,
                    initial_values=color_data.astype(np.float64)
                    / np.iinfo(color_data.dtype).max,
                )
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
            if not uv_ids:
                raise SystemExit("No UV attributes found in mesh for uv material")
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
                if not texture_file.is_file():
                    raise SystemExit(f"Texture file {texture_file} not found")
                layer = layer.material(
                    "Principled",
                    hkw.texture.Image(
                        texture_file,
                        saturation=args.saturation,
                        whiteness=args.whiteness,
                    ),
                )

    if args.point_cloud:
        if args.comp:
            raise SystemExit("--point-cloud and --comp options are mutually exclusive")
        layer = layer.mark("Point").channel(
            size=hkw.channel.Size(data=args.point_size, space="screen")
        )

    if args.comp:
        layer = layer.transform(hkw.transform.Compute(component="comp"))
        layer = layer.material(
            "Principled",
            hkw.texture.ScalarField("comp", colormap="set1", categories=True),
        )

    if args.explode is not None:
        # Displace each connected component outward from the mesh center by
        # `args.explode` times its offset. Explode groups facets by a facet
        # attribute, so reuse the "comp" component attribute when --comp already
        # computed it; otherwise compute a private one first. The extra Compute
        # is deeper in the transform chain, so it runs before Explode.
        pieces_attr = "comp"
        if not args.comp:
            pieces_attr = "_explode_comp"
            layer = layer.transform(hkw.transform.Compute(component=pieces_attr))
        layer = layer.transform(
            hkw.transform.Explode(pieces=pieces_attr, magnitude=args.explode)
        )

    normal_channel: str | None = None
    if args.normal == "vertex":
        layer = layer.transform(hkw.transform.Compute(vertex_normal="vertex_normal"))
        normal_channel = "vertex_normal"
    elif args.normal == "facet":
        layer = layer.transform(hkw.transform.Compute(facet_normal="face_normal"))
        normal_channel = "face_normal"
    elif args.normal is not None:
        if not mesh.has_attribute(args.normal):
            raise SystemExit(f"Normal attribute '{args.normal}' not found in mesh")
        normal_channel = args.normal

    if normal_channel is not None:
        if args.point_cloud:
            # Render each point as a disc oriented by the normal field.
            layer = layer.channel(
                shape=hkw.channel.Shape(base_shape="disk", orientation=normal_channel)
            )
        else:
            layer = layer.channel(normal=normal_channel)

    if args.uv:
        layer = layer.transform(hkw.transform.UVMesh())

    clip_transform = None
    if args.clip is not None:
        clip_axis, clip_value = args.clip
        axis_index = "xyz".index(clip_axis[-1])
        point = bbox_min.copy()
        point[axis_index] += clip_value * bbox_size[axis_index]
        normal = np.zeros(3)
        normal[axis_index] = -1.0 if clip_axis.startswith("-") else 1.0
        clip_transform = hkw.transform.Clip(point=point, normal=normal)
        layer = layer.transform(clip_transform)

    if args.wireframe:
        layer = layer.show_edges(
            width=args.wire_thickness, width_space="screen", name="Wireframe"
        )

    seam_base = None
    seam_uv_name = None
    if args.seams:
        uv_ids = mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV)
        if not uv_ids:
            raise SystemExit("No UV attributes found in mesh for seams rendering")
        seam_uv_name = mesh.get_attribute_name(uv_ids[0])
        seam_base = hkw.layer(mesh)

    vec_field_attr = args.vector_field or args.cross_field
    if vec_field_attr is not None:
        is_cross = args.cross_field is not None
        vf_mesh = copy.deepcopy(mesh)
        lagrange.triangulate_polygonal_facets(vf_mesh)
        if args.field_style == "fur":
            if is_cross:
                raise SystemExit(
                    "--field-style fur supports --vector-field only, not --cross-field."
                )
            fur_transform: hkw.transform.Transform = hkw.transform.Fur(
                vec_field=vec_field_attr,
                n=args.num_streamlines,
                follow_surface=True,
            )
            if clip_transform is not None:
                fur_transform = fur_transform * clip_transform
            vf_layer = (
                hkw.layer(vf_mesh)
                .transform(fur_transform)
                .mark("Curve")
                .material("Hair", color=args.streamline_color)
            )
        elif args.field_style == "arrow":
            vf_base = hkw.layer(vf_mesh)
            if clip_transform is not None:
                vf_base = vf_base.transform(clip_transform)
            vf_layer = (
                vf_base.mark("Curve")
                .material("Diffuse", args.streamline_color)
                .channel(
                    vector_field=hkw.channel.VectorField(
                        data=vec_field_attr,
                        end_type="arrow",
                        normalize=True,
                    ),
                    size=hkw.channel.Size(data=args.wire_thickness, space="screen"),
                )
            )
        else:
            streamline_transform: hkw.transform.Transform = hkw.transform.Streamline(
                vec_field=vec_field_attr,
                cross_field=is_cross,
                n=args.num_streamlines,
                length=(
                    float(args.streamline_length * bbox_diag)
                    if args.streamline_length is not None
                    else None
                ),
            )
            if clip_transform is not None:
                streamline_transform = streamline_transform * clip_transform
            vf_layer = (
                hkw.layer(vf_mesh)
                .transform(streamline_transform)
                .mark("Curve")
                .material("Diffuse", args.streamline_color)
                .channel(
                    size=hkw.channel.Size(data=args.wire_thickness, space="screen")
                )
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
        valence_view = base.mark("Point").channel(size=float(0.005 * bbox_diag))

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
                colormap=colormap,  # type: ignore[arg-type]
                domain=(3, 7),
                categories=True,
            ),
            roughness=0.0,
            metallic=0.2,
        )

        if clip_transform is not None:
            valence_view = valence_view.transform(clip_transform)
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

    if seam_base is not None:
        assert seam_uv_name is not None
        if args.orient_pca:
            seam_base = seam_base.transform(
                hkw.transform.PrincipalAxes(frame=_orient_pca_target_frame(args.z_up))
            )
        if args.rotate:
            seam_axis = [0, 0, 1] if args.z_up else [0, 1, 0]
            seam_base = seam_base.rotate(
                axis=seam_axis, angle=args.rotate * math.pi / 180
            )
        if args.clip is not None:
            seam_base = seam_base.transform(
                hkw.transform.Clip(point=point, normal=normal)
            )
        seams = (
            seam_base.transform(hkw.transform.Boundary(attributes=[seam_uv_name]))
            .mark("Curve")
            .material("Diffuse", "black")
            .channel(size=hkw.channel.Size(data=args.wire_thickness, space="screen"))
        )
        layer = layer + seams

    if normalize:
        # Outermost transform (applied after clip/rotate) so the styled result —
        # mesh and any overlays alike — fits a unit bounding sphere centered at the
        # origin. This is what makes every cell, including a lone mesh in a ragged
        # grid row, render at the same on-screen size regardless of source scale.
        layer = layer.transform(hkw.transform.Normalize())

    return layer
