from .bsdf import generate_bsdf_config
from .base_shapes import create_icosphere, create_disk
from .medium import generate_medium_config
from ...common import logger
from ...common.vector_field import filter_zero_length_vectors
from ...compiler import View
from ...compiler.fur import STRAND_RADIUS_ATTR
from ...grammar.scale import Attribute
from ...grammar.channel import DEFAULT_COVARIANCE_SIZE, DEFAULT_MARK_SIZE
from ...grammar.channel.curvestyle import Bend
from ...grammar.channel.material import Dielectric, Hair
from .utils import rotation
from ...common.color import srgb_to_linear_array

from dataclasses import fields, is_dataclass
from typing import Any, Iterator

import copy
import lagrange
import mitsuba as mi
import numpy as np
import numpy.typing as npt
import pathlib
import tempfile


def extract_size(view: View, default_size=DEFAULT_MARK_SIZE):
    """Extract the size attribute from a view.

    Args:
        view: The view to extract size from.
        default_size: The default size if size attribute is not specified.

    Returns:
        A list of size values of length n.
    """
    assert view.data_frame is not None
    mesh = view.data_frame.mesh

    if view.size_channel is not None:
        match view.size_channel.data:
            case float():
                return view.size_channel.data
            case Attribute():
                assert view.size_channel.data._internal_name is not None
                return mesh.attribute(
                    view.size_channel.data._internal_name
                ).data.tolist()
            case _:
                raise NotImplementedError(
                    f"Unsupported size channel type: {type(view.size_channel.data)}"
                )
    elif mesh.has_attribute(STRAND_RADIUS_ATTR):
        # Fur strands bake a per-vertex root-to-tip taper radius; use it as the
        # tube size so the strands taper when no explicit size channel is set.
        return mesh.attribute(STRAND_RADIUS_ATTR).data.tolist()
    else:
        return default_size


def extract_transform_from_covariances(view: View):
    """Extract the affine transform from covariance attribute from a view.

    Args:
        view: The view to extract covariance from.

    Returns:
        A list of n 3x3 affine transform matrices, M, where the covariance matrix is M @ M^T.
    """
    assert view.data_frame is not None
    mesh = view.data_frame.mesh

    assert view.covariance_channel is not None
    assert isinstance(view.covariance_channel.data, Attribute)
    attr_name = view.covariance_channel.data._internal_name
    assert attr_name is not None
    assert mesh.has_attribute(attr_name)

    attr = mesh.attribute(attr_name)
    assert attr.element_type == lagrange.AttributeElement.Vertex
    assert attr.data.shape[1] == 9
    if view.covariance_channel.full:
        sigma = attr.data.reshape(-1, 3, 3)
        U, S, Vh = np.linalg.svd(sigma)
        S_diag = np.apply_along_axis(lambda _s: np.diag(_s), 1, S)
        return U @ np.sqrt(S_diag)
    else:
        return attr.data.reshape(-1, 3, 3)


def generate_point_config(view: View, stamp: str, index: int) -> dict:
    """Generate point cloud shapes from a View.

    Args:
        view: The view to generate point cloud shapes from.
        stamp: The time stamp string used for creating a unique filename.
        index: The index of the view.

    Returns:
        The mitsuba config for the point cloud shapes.
    """
    assert view.data_frame is not None
    mesh = view.data_frame.mesh
    shapes: list[dict[str, Any]] = []

    # Extract shape
    base_shape = "sphere"
    if view.shape_channel is not None:
        base_shape = view.shape_channel.base_shape

    if view.covariance_channel is None:
        # Compute radii
        radii = extract_size(view)
        if np.isscalar(radii):
            radii = [radii] * mesh.num_vertices
        assert len(radii) == mesh.num_vertices

        # Generate spheres.
        global_transform = mi.ScalarTransform4f(view.global_transform)  # type: ignore
        match base_shape:
            case "sphere":
                # Ignore normal as sphere is invariant under rotation.
                shapes = list(
                    map(
                        lambda itr: {
                            "type": "sphere",
                            "center": itr[1].tolist(),
                            "radius": radii[itr[0]],
                            "to_world": global_transform,
                        },
                        enumerate(mesh.vertices),
                    )
                )
            case "cube" | "disk":
                local_transforms = [
                    np.array(
                        [
                            [radii[i], 0, 0, mesh.vertices[i][0]],
                            [0, radii[i], 0, mesh.vertices[i][1]],
                            [0, 0, radii[i], mesh.vertices[i][2]],
                            [0, 0, 0, 1],
                        ]
                    )
                    for i in range(mesh.num_vertices)
                ]

                # Apply normal rotation if necessary
                if (
                    view.shape_channel is not None
                    and view.shape_channel.orientation is not None
                ):
                    assert isinstance(view.shape_channel.orientation, Attribute)
                    normal_attr_name = view.shape_channel.orientation._internal_name
                    assert normal_attr_name is not None
                    assert mesh.has_attribute(normal_attr_name)

                    z = np.array([0, 0, 1])
                    normals = mesh.attribute(normal_attr_name).data  # type: ignore

                    for i, m in enumerate(local_transforms):
                        m[:, :] = m @ rotation(z, normals[i])

                # Compose with the global transform in Mitsuba's transform space;
                # ``global_transform @ np.ndarray`` is not a valid matmul.
                local_transforms = [
                    mi.ScalarTransform4f(m)
                    for m in local_transforms  # type: ignore
                ]

                if base_shape == "cube":
                    # Generate cubes.
                    shapes = list(
                        map(
                            lambda itr: {
                                "type": "cube",
                                "to_world": global_transform @ local_transforms[itr[0]],
                            },
                            enumerate(mesh.vertices),
                        )
                    )
                elif base_shape == "disk":
                    disk = create_disk(16)
                    tmp_dir = pathlib.Path(tempfile.gettempdir())
                    filename = tmp_dir / f"{stamp}-view-{index:03}.ply"
                    logger.debug(f"Saving point mark shape to '{str(filename)}'.")
                    lagrange.io.save_mesh(filename, disk)  # type: ignore
                    base_shape_config = {
                        "type": "ply",
                        "filename": str(filename.resolve()),
                        "face_normals": True,
                    }
                    shapes = [
                        base_shape_config | {"to_world": global_transform @ m}
                        for m in local_transforms
                    ]
    else:  # with covariance
        # Generate base shape config.
        match base_shape:
            case "sphere":
                # Generate point mark shape.
                sphere = create_icosphere(1)
                tmp_dir = pathlib.Path(tempfile.gettempdir())
                filename = tmp_dir / f"{stamp}-view-{index:03}.ply"
                logger.debug(f"Saving point mark shape to '{str(filename)}'.")
                lagrange.io.save_mesh(filename, sphere)  # type: ignore
                base_shape_config = {
                    "type": "ply",
                    "filename": str(filename.resolve()),
                    "face_normals": False,
                }
            case "cube":
                base_shape_config = {"type": "cube"}

        # Compute radii, defaulting to the covariance glyph size.
        radii = extract_size(view, DEFAULT_COVARIANCE_SIZE)
        if np.isscalar(radii):
            radii = [radii] * mesh.num_vertices
        assert len(radii) == mesh.num_vertices

        M = extract_transform_from_covariances(view)
        global_transform = mi.ScalarTransform4f(view.global_transform)  # type: ignore
        for i, v in enumerate(mesh.vertices):
            local_transform = np.eye(4)
            local_transform[:3, :3] = M[i] * radii[i]
            local_transform[:3, 3] = v
            local_transform = mi.ScalarTransform4f(local_transform)  # type: ignore
            shape = base_shape_config.copy()
            shape["to_world"] = global_transform @ local_transform
            shapes.append(shape)

    # Generate bsdf
    bsdfs = generate_bsdf_config(view, is_primitive=True)
    if "type" in bsdfs:
        # Single bsdf
        bsdf = bsdfs
        for shape in shapes:
            shape["bsdf"] = bsdf
    else:
        assert len(bsdfs) == len(shapes)
        for (bsdf_id, bsdf), shape in zip(bsdfs.items(), shapes):
            shape[bsdf_id] = bsdf

    mi_config: dict[str, Any] = {
        f"view_{index:03}_shape_{i:06}": shape for i, shape in enumerate(shapes)
    }
    return mi_config


def extract_vector_field(view: View):
    assert view.data_frame is not None
    mesh = view.data_frame.mesh

    assert view.vector_field_channel is not None
    assert isinstance(view.vector_field_channel.data, Attribute)
    attr_name = view.vector_field_channel.data._internal_name
    assert attr_name is not None
    assert mesh.has_attribute(attr_name)

    attr = mesh.attribute(attr_name)
    match attr.element_type:
        case lagrange.AttributeElement.Vertex:
            base = mesh.vertices
            size = extract_size(view)
            if np.isscalar(size):
                size = [size] * mesh.num_vertices
        case lagrange.AttributeElement.Facet:
            centroid_attr_id = lagrange.compute_facet_centroid(mesh)
            base = mesh.attribute(centroid_attr_id).data  # type: ignore
            size = extract_size(view)
            if np.isscalar(size):
                size = [size] * mesh.num_facets
        case _:
            raise NotImplementedError(
                f"Unsupported vector field element type: {attr.element_type}"
            )
    tip = attr.data + base
    ctrl_pts_1: npt.NDArray | None = None
    ctrl_pts_2: npt.NDArray | None = None

    match view.vector_field_channel.style:
        case Bend():
            direction = view.vector_field_channel.style.direction
            assert isinstance(direction, Attribute)
            assert direction._internal_name is not None
            assert mesh.has_attribute(direction._internal_name)
            dir_attr = mesh.attribute(direction._internal_name)
            assert dir_attr.element_type == attr.element_type, (
                "Direction attribute must have the same element type as vector field attribute."
            )
            dirs = dir_attr.data
            assert np.all(dirs.shape == base.shape)

            bend_type = view.vector_field_channel.style.bend_type
            if bend_type == "n":
                ctrl_pts_1 = base + dirs
                ctrl_pts_2 = tip + dirs
            elif bend_type == "r":
                ctrl_pts_1 = base + dirs
                ctrl_pts_2 = base + dirs + 0.5 * (tip - base)
                tip = tip + dirs
            elif bend_type == "s":
                ctrl_pts_1 = base + dirs
                ctrl_pts_2 = tip
                tip = tip + dirs
            else:
                raise NotImplementedError(f"Unsupported bend type: {bend_type}")

    def refine(mesh: lagrange.SurfaceMesh, data: npt.NDArray, level: int):
        assert mesh.is_triangle_mesh, "Only triangle mesh is supported."
        facets = mesh.facets
        n = level + 1
        refined_data = []
        B0, B1, B2 = np.mgrid[0 : n + 1, 0 : n + 1, 0 : n + 1]
        for b0, b1, b2 in zip(B0.ravel(), B1.ravel(), B2.ravel()):
            s = b0 + b1 + b2
            if s != n:
                continue

            d = (
                data[facets[:, 0]] * b0
                + data[facets[:, 1]] * b1
                + data[facets[:, 2]] * b2
            )
            d /= n
            refined_data.append(d)
        return np.vstack(refined_data)

    if view.vector_field_channel.refinement_level > 0:
        base = refine(mesh, base, view.vector_field_channel.refinement_level)
        tip = refine(mesh, tip, view.vector_field_channel.refinement_level)
        if ctrl_pts_1 is not None:
            assert ctrl_pts_2 is not None
            ctrl_pts_1 = refine(
                mesh, ctrl_pts_1, view.vector_field_channel.refinement_level
            )
            ctrl_pts_2 = refine(
                mesh, ctrl_pts_2, view.vector_field_channel.refinement_level
            )
        size = refine(
            mesh, np.array(size), view.vector_field_channel.refinement_level
        ).ravel()

    base_size = size
    if view.vector_field_channel.end_type == "point":
        tip_size = np.zeros_like(size)
    elif view.vector_field_channel.end_type == "arrow":
        assert ctrl_pts_1 is None
        assert ctrl_pts_2 is None
        stem_point = 0.25 * base + 0.75 * tip
        base = np.vstack([base, stem_point])
        tip = np.vstack([stem_point, tip])
        size = np.array(size)
        base_size = np.hstack([size, 2 * size])
        tip_size = np.hstack([size, np.zeros_like(size)])
    else:  # "flat": constant radius at both ends
        tip_size = size

    base, tip, base_size, tip_size, ctrl_pts_1, ctrl_pts_2 = filter_zero_length_vectors(
        base, tip, base_size, tip_size, ctrl_pts_1, ctrl_pts_2
    )

    return [base, ctrl_pts_1, ctrl_pts_2, tip, base_size, tip_size]


def extract_edges(view: View):
    assert view.data_frame is not None
    mesh = view.data_frame.mesh
    mesh.initialize_edges()

    base: npt.NDArray = np.ndarray((mesh.num_edges, mesh.dimension), dtype=np.float32)
    tip: npt.NDArray = np.ndarray((mesh.num_edges, mesh.dimension), dtype=np.float32)
    base_size: npt.NDArray = np.ndarray(mesh.num_edges, dtype=np.float32)
    tip_size: npt.NDArray = np.ndarray(mesh.num_edges, dtype=np.float32)

    sizes = extract_size(view)
    if np.isscalar(sizes):
        sizes = [sizes] * mesh.num_vertices

    vertices = mesh.vertices
    for i in range(mesh.num_edges):
        edge_vts = mesh.get_edge_vertices(i)
        base[i] = vertices[edge_vts[0]]
        tip[i] = vertices[edge_vts[1]]
        base_size[i] = sizes[edge_vts[0]]
        tip_size[i] = sizes[edge_vts[1]]

    return [base, tip, base_size, tip_size]


def _freeze_config(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze_config(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config(item) for item in value)
    if isinstance(value, np.ndarray):
        return tuple(_freeze_config(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _contains_mesh_attribute(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("type") == "mesh_attribute" or any(
            _contains_mesh_attribute(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_mesh_attribute(item) for item in value)
    return False


def _iter_attributes(value: Any) -> Iterator[Attribute]:
    if isinstance(value, Attribute):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_attributes(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_attributes(child)
    elif is_dataclass(value):
        for field in fields(value):
            yield from _iter_attributes(getattr(value, field.name))


def _material_attribute_domains(
    view: View, mesh: lagrange.SurfaceMesh
) -> tuple[set[lagrange.AttributeElement], tuple[str, ...]]:
    assert view.material_channel is not None
    domains: set[lagrange.AttributeElement] = set()
    facet_names: set[str] = set()
    for attribute in _iter_attributes(view.material_channel):
        for name in (attribute._internal_name, attribute._internal_color_field):
            if name is None or not mesh.has_attribute(name):
                continue
            element = mesh.attribute(name).element_type
            domains.add(element)
            if element == lagrange.AttributeElement.Facet:
                facet_names.add(attribute.name)
    return domains, tuple(sorted(facet_names))


def _expand_facet_bsdfs_to_curve_vertices(
    mesh: lagrange.SurfaceMesh,
    bsdf_config: dict[str, Any],
    num_edges: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Replicate each facet BSDF to the two per-edge curve endpoints.

    Endpoints are conceptually duplicated per curve edge, preserving facet
    discontinuities at source vertices shared by differently-colored edges.
    """
    mesh.initialize_edges()
    edge_vertex_bsdfs = []
    for edge_index in range(num_edges):
        incident_facets = mesh.facets_around_edge(edge_index)
        if len(incident_facets) == 0:
            raise RuntimeError(f"Curve edge {edge_index} has no incident facet.")
        facet_index = int(incident_facets[0])
        facet_bsdf = bsdf_config[f"bsdf_{facet_index:06}"]
        edge_vertex_bsdfs.append((facet_bsdf, facet_bsdf))
    return edge_vertex_bsdfs


def generate_curve_config(view: View, stamp: str, index: int) -> dict:
    assert view.data_frame is not None

    # The radius of the linearcurve shape in Mitsuba 3.4.0 will not be transformed using the
    # `to_world` transform. This seems to be a bug on Mitsuba's part. Thus, we use a temporary fix
    # to bypass this bug.
    # TODO: Update once Mitsuba fixed the bug.
    scale_correction_factor = np.absolute(
        np.cbrt(np.linalg.det(view.global_transform[:3, :3]))
    )

    if view.vector_field_channel is not None:
        base, ctrl_pts_1, ctrl_pts_2, tip, base_size, tip_size = extract_vector_field(
            view
        )
    else:
        base, tip, base_size, tip_size = extract_edges(view)
        ctrl_pts_1 = ctrl_pts_2 = None

    assert len(base) == len(tip)
    assert len(base) == len(base_size)
    assert len(tip) == len(tip_size)
    if len(base) == 0:
        logger.warning(f"View {index} has no curve segments — skipping.")
        return {}

    mesh = view.data_frame.mesh
    # Hair is a uniform strand BSDF, not a primitive-expanded material. Other
    # materials keep the primitive path so vertex/facet attributes are expanded.
    bsdf_config = generate_bsdf_config(
        view, is_primitive=not isinstance(view.material_channel, Hair)
    )
    if "type" in bsdf_config:
        if _contains_mesh_attribute(bsdf_config):
            raise NotImplementedError(
                "Mitsuba curve marks cannot bind mesh_attribute BSDF textures. "
                "Use an edge-derived curve with per-vertex BSDF expansion."
            )
        edge_groups = [
            (generate_bsdf_config(view, is_primitive=False), list(range(len(base))))
        ]
    else:
        if view.vector_field_channel is not None:
            raise NotImplementedError(
                "Mitsuba curve material textures currently require edge-derived "
                "curves, not vector-field curves."
            )

        attribute_domains, facet_attribute_names = _material_attribute_domains(
            view, mesh
        )
        facet_edge_bsdfs = None
        if lagrange.AttributeElement.Facet in attribute_domains:
            if lagrange.AttributeElement.Vertex in attribute_domains:
                raise NotImplementedError(
                    "Mitsuba curve material textures cannot mix vertex- and "
                    "facet-domain attributes."
                )
            if len(bsdf_config) != mesh.num_facets:
                raise NotImplementedError(
                    "Facet-domain curve materials must produce one BSDF per facet."
                )
            logger.warning(
                "Mitsuba curve view %d: expanding facet-domain material attribute(s) "
                "%s to per-edge vertex values.",
                index,
                ", ".join(facet_attribute_names),
            )
            facet_edge_bsdfs = _expand_facet_bsdfs_to_curve_vertices(
                mesh, bsdf_config, len(base)
            )
        else:
            if len(bsdf_config) != mesh.num_vertices:
                raise NotImplementedError(
                    "Mitsuba curve material textures currently require vertex-domain "
                    "attributes."
                )
            mesh.initialize_edges()

        groups: dict[Any, tuple[dict[str, Any], list[int]]] = {}
        for edge_index in range(len(base)):
            if facet_edge_bsdfs is None:
                vertex0, vertex1 = mesh.get_edge_vertices(edge_index)
                bsdf0 = bsdf_config[f"bsdf_{vertex0:06}"]
                bsdf1 = bsdf_config[f"bsdf_{vertex1:06}"]
            else:
                bsdf0, bsdf1 = facet_edge_bsdfs[edge_index]
            key0 = _freeze_config(bsdf0)
            if key0 != _freeze_config(bsdf1):
                raise NotImplementedError(
                    "Mitsuba curve material textures must be constant across each "
                    "mesh edge; use a tube mesh for smoothly varying values."
                )
            if key0 not in groups:
                groups[key0] = (bsdf0, [])
            groups[key0][1].append(edge_index)
        edge_groups = list(groups.values())

    curve_type = (
        "linearcurve" if ctrl_pts_1 is None or ctrl_pts_2 is None else "bsplinecurve"
    )
    mi_config: dict[str, Any] = {}
    for group_index, (bsdf, edge_indices) in enumerate(edge_groups):
        filename = (
            pathlib.Path(tempfile.gettempdir())
            / f"{stamp}-view-{index:03}-curve-{group_index:03}.txt"
        )
        logger.debug(f"Saving curves to '{str(filename)}'.")
        with open(filename, "w") as fout:
            if ctrl_pts_1 is None or ctrl_pts_2 is None:
                for curve_index in edge_indices:
                    p0, p1 = base[curve_index], tip[curve_index]
                    s0, s1 = base_size[curve_index], tip_size[curve_index]
                    fout.write(
                        f"{p0[0]} {p0[1]} {p0[2]} {s0 * scale_correction_factor}\n"
                    )
                    fout.write(
                        f"{p1[0]} {p1[1]} {p1[2]} {s1 * scale_correction_factor}\n\n"
                    )
            else:
                for curve_index in edge_indices:
                    p0 = base[curve_index]
                    p1 = ctrl_pts_1[curve_index]
                    p2 = ctrl_pts_2[curve_index]
                    p3 = tip[curve_index]
                    s0 = base_size[curve_index]
                    s3 = tip_size[curve_index]
                    s1 = 0.75 * s0 + 0.25 * s3
                    s2 = 0.25 * s0 + 0.75 * s3
                    for point, size in (
                        (p0, s0),
                        (p0, s0),
                        (p0, s0),
                        (p0, s0),
                        (p1, s1),
                        (p2, s2),
                        (p3, s3),
                        (p3, s3),
                        (p3, s3),
                        (p3, s3),
                    ):
                        fout.write(
                            f"{point[0]} {point[1]} {point[2]} "
                            f"{size * scale_correction_factor}\n"
                        )
                    fout.write("\n")

        mi_config[f"view_{index:03}_shape_{group_index:06}"] = {
            "type": curve_type,
            "filename": str(filename.resolve()),
            "bsdf": bsdf,
            "to_world": mi.ScalarTransform4f(view.global_transform),  # type: ignore
        }
    return mi_config


def _rename_attributes(mesh: lagrange.SurfaceMesh, active_attributes: list[Attribute]):
    """Rename generic scalar and vector attribute with suffix "_0". This is required by mitsuba to
    correct parse them from a ply file.

    Args:
        mesh: The mesh to rename attributes.
        active_attributes: The list of active attributes.
    """
    processed_names = set()
    for attr in active_attributes:
        name = attr._internal_name
        assert name is not None
        if lagrange.SurfaceMesh.attr_name_is_reserved(name):
            continue

        if name in processed_names:
            continue

        mesh_attr = mesh.attribute(name)
        if mesh_attr.usage not in [
            lagrange.AttributeUsage.Scalar,
            lagrange.AttributeUsage.Vector,
        ]:
            continue

        new_name = f"{name}_0"
        # It seems mitsuba requires a "_#" suffix to work propertly with scalar/vector
        # attributes. Color/position/normal/uv attributes all has their own representation in ply
        # format, so they do not need to be changed.
        mesh.rename_attribute(name, new_name)
        processed_names.add(name)

        # Note that we will keep attr._internal_name the same.


def _linearize_color_attributes(mesh: lagrange.SurfaceMesh):
    """Decode sRGB color attributes to linear RGB in place.

    hakowan bakes colormap / color-field colors in sRGB, but Mitsuba's
    ``mesh_attribute`` texture interprets vertex/face colors as linear RGB.
    Replace each color attribute with a decoded copy (same name) so the ply that
    Mitsuba loads carries linear colors, matching the uniform color path. Meant
    to run on the shallow export copy; the original mesh is left untouched.

    Args:
        mesh: The mesh (export copy) whose color attributes to linearize.
    """
    color_ids = mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.Color)
    for cid in color_ids:
        name = mesh.get_attribute_name(cid)
        if mesh.is_attribute_indexed(name):
            indexed_attr = mesh.indexed_attribute(name)
            element = indexed_attr.element_type
            values = np.array(indexed_attr.values.data, dtype=np.float64)  # copy
            indices = np.array(indexed_attr.indices.data)  # copy
            values[..., :3] = srgb_to_linear_array(values[..., :3])
            mesh.delete_attribute(name)
            mesh.create_attribute(
                name,
                element=element,
                usage=lagrange.AttributeUsage.Color,
                initial_values=values,
                initial_indices=indices,
            )
        else:
            attr = mesh.attribute(name)
            element = attr.element_type
            values = np.array(attr.data, dtype=np.float64)  # copy
            values[..., :3] = srgb_to_linear_array(values[..., :3])
            mesh.delete_attribute(name)
            mesh.create_attribute(
                name,
                element=element,
                usage=lagrange.AttributeUsage.Color,
                initial_values=values,
            )


def _generate_bsdf_with_mesh(
    view: View, mesh: lagrange.SurfaceMesh, is_primitive: bool
) -> dict:
    """Call generate_bsdf_config with a specific mesh instead of view.data_frame.mesh.

    Temporarily swaps the mesh on the view so that texture config generators
    see the export-ready mesh (with secondary Color attributes expanded to
    scalar triplets) rather than the original compiled mesh.
    """
    original_mesh = view.data_frame.mesh  # type: ignore[union-attr]
    view.data_frame.mesh = mesh  # type: ignore[union-attr]
    try:
        return generate_bsdf_config(view, is_primitive=is_primitive)
    finally:
        view.data_frame.mesh = original_mesh  # type: ignore[union-attr]


def _expand_secondary_color_attributes(mesh: lagrange.SurfaceMesh):
    """Convert Color attributes beyond the first into scalar triplets for Mitsuba.

    Mitsuba's PLY reader only recognises the first Color attribute set
    (``red``/``green``/``blue`` → ``vertex_color``).  Subsequent Color
    attributes land in the PLY as ``red_N``/``green_N``/``blue_N``, which
    Mitsuba cannot access by name.  This function converts every Color
    attribute after the first into three per-channel Scalar attributes named
    ``{attr_name}_0``, ``{attr_name}_1``, ``{attr_name}_2``; Mitsuba then
    exposes the group as ``vertex_{attr_name}`` (or ``face_{attr_name}``).

    Must run after ``_linearize_color_attributes`` so the data is already in
    linear RGB.

    Args:
        mesh: The mesh (export copy) to modify in place.
    """
    color_ids = mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.Color)
    for cid in color_ids[1:]:
        name = mesh.get_attribute_name(cid)
        if mesh.is_attribute_indexed(name):
            indexed_attr = mesh.indexed_attribute(name)
            values = np.array(indexed_attr.values.data, dtype=np.float64)
            indices = np.array(indexed_attr.indices.data)
            element = indexed_attr.element_type
            mesh.delete_attribute(name)
            for ch in range(min(values.shape[1], 3)):
                mesh.create_attribute(
                    f"{name}_{ch}",
                    element=element,
                    usage=lagrange.AttributeUsage.Scalar,
                    initial_values=values[:, ch : ch + 1],
                    initial_indices=indices,
                )
        else:
            attr = mesh.attribute(name)
            values = np.array(attr.data, dtype=np.float64)
            element = attr.element_type
            mesh.delete_attribute(name)
            for ch in range(min(values.shape[1], 3)):
                mesh.create_attribute(
                    f"{name}_{ch}",
                    element=element,
                    usage=lagrange.AttributeUsage.Scalar,
                    initial_values=values[:, ch : ch + 1],
                )


def generate_surface_config(view: View, stamp: str, index: int) -> dict:
    """Generate the mitsuba config for a mesh.

    It does the following things:
    1. Rename all generic scalar/vector attributes with _0 suffix.
    2. Save the mesh and all active attributes in ply format in a temp directory.
    3. Generate the bsdf config associated with the shape.

    Args:
        view: The view to generate mesh config from.
        stamp: The time stamp string used for creating a unique filename.
        index: The index of the view.

    Returns:
        The mitsuba config for the mesh view.
    """
    assert view.data_frame is not None
    mesh = copy.copy(view.data_frame.mesh)  # Shallow copy
    if not mesh.is_triangle_mesh:
        logger.debug("Convert dataframe to triangle mesh.")
        lagrange.triangulate_polygonal_facets(mesh)

    # Rename attributes in the shallow copy of mesh.
    _rename_attributes(mesh, view._active_attributes)

    # Decode baked sRGB color attributes to linear RGB for Mitsuba.
    _linearize_color_attributes(mesh)

    # PLY only exposes the first Color attribute set as ``vertex_color``.
    # Convert every additional Color attribute to a scalar triplet so Mitsuba
    # can access it as ``vertex_{attr_name}``.
    _expand_secondary_color_attributes(mesh)

    # Just a sanity check that all attributes are still present in the original mesh.
    for attr in view._active_attributes:
        name = attr._internal_name
        assert name is not None
        assert view.data_frame.mesh.has_attribute(name)

    # Normal handling for Mitsuba's ply plugin. The ply format only stores
    # per-vertex normals as the `nx, ny, nz` properties, and Mitsuba reads them
    # back when `face_normals=False`. To preserve custom normals exactly for any
    # element type, we convert the normal attribute to an indexed attribute and
    # unify the index buffer: this duplicates vertices across creases so the
    # resulting per-vertex normals reproduce the original facet / corner / vertex
    # normals (flat shading included) without losing any values.
    normal_ids = mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.Normal)
    clear_computed_edges = not normal_ids and not mesh.has_edges
    if not normal_ids:
        # Mitsuba otherwise smooths every shared vertex, even across cube
        # edges. Match WebGL's default feature-angle normals so hard surfaces
        # retain their facets while smooth meshes remain smooth.
        lagrange.compute_normal(mesh)
        normal_ids = mesh.get_matching_attribute_ids(
            usage=lagrange.AttributeUsage.Normal
        )
    use_facet_normal = False
    if len(normal_ids) > 0:
        keep_name = mesh.get_attribute_name(normal_ids[0])
        # Drop any extra normal attributes to avoid ambiguous ply output.
        for normal_id in normal_ids[1:]:
            mesh.delete_attribute(normal_id)
        if not mesh.is_attribute_indexed(keep_name):
            lagrange.map_attribute_in_place(
                mesh, keep_name, lagrange.AttributeElement.Indexed
            )
        mesh = lagrange.unify_index_buffer(mesh, [keep_name])
    if clear_computed_edges:
        # Computing feature-angle normals initializes edge adjacency. The PLY
        # plugin does not need it and warns about its extra "edge" element.
        mesh.clear_edges()

    tmp_dir = pathlib.Path(tempfile.gettempdir())
    filename = tmp_dir / f"{stamp}-view-{index:03}.ply"
    logger.debug(f"Saving mesh to '{str(filename)}'.")
    lagrange.io.save_mesh(filename, mesh)  # type: ignore

    mi_config = {
        "type": "ply",
        "filename": str(filename.resolve()),
        # Pass the modified mesh so that generate_scalar_field_config sees
        # the scalar triplets created by _expand_secondary_color_attributes
        # and derives the correct Mitsuba attribute names.
        "bsdf": _generate_bsdf_with_mesh(view, mesh, is_primitive=False),
        "face_normals": use_facet_normal,
        "to_world": mi.ScalarTransform4f(view.global_transform),  # type: ignore
    }

    # Generate medium setting for dielectric and its derived materials.
    if (
        view.material_channel is not None
        and isinstance(view.material_channel, Dielectric)
        and view.material_channel.medium is not None
    ):
        mi_config["interior"] = generate_medium_config(view)

    mi_config = {f"view_{index:03}_shape_000000": mi_config}
    return mi_config
