from .bsdf import generate_bsdf_config
from .icosphere import create_icosphere
from .medium import generate_medium_config
from ..common import logger
from ..compiler import View
from ..grammar.scale import Attribute
from ..grammar.channel.curvestyle import Bend
from ..grammar.channel.material import Dielectric

from typing import Any
import copy
import lagrange
import mitsuba as mi
import numpy as np
import numpy.typing as npt
import pathlib
import re
import tempfile


def extract_size(view: View, default_size=0.01):
    """Extract the size attribute from a view.

    :param view: The view to extract size from.
    :param n: The cardinality of size attribute.
    :param default_size: The default size if size attribute is not specified.

    :return: A list of size values of length n.
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
    else:
        return default_size


def extract_covariances(view: View):
    """Extract the covariance attribute from a view.

    :param view: The view to extract covariance from.

    :return: A list of n 3x3 covariance matrices.
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
    return attr.data.reshape(-1, 3, 3)


def generate_point_config(view: View, stamp: str, index: int):
    """Generate point cloud shapes from a View."""
    assert view.data_frame is not None
    mesh = view.data_frame.mesh
    shapes: list[dict[str, Any]] = []
    shape_group: dict[str, Any] = {}

    if view.covariance_channel is None:
        # Compute radii
        radii = extract_size(view)
        if np.isscalar(radii):
            radii = [radii] * mesh.num_vertices
        assert len(radii) == mesh.num_vertices

        # Generate spheres.
        global_transform = mi.ScalarTransform4f(view.global_transform)  # type: ignore
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
    else:
        # Generate point mark shape.
        sphere = create_icosphere(1)
        tmp_dir = pathlib.Path(tempfile.gettempdir())
        filename = tmp_dir / f"{stamp}-view-{index:03}.ply"
        logger.debug(f"Saving point mark shape to '{str(filename)}'.")
        lagrange.io.save_mesh(filename, sphere)  # type: ignore

        # Compute radii, with default radii as 1.
        radii = extract_size(view, 1)
        if np.isscalar(radii):
            radii = [radii] * mesh.num_vertices
        assert len(radii) == mesh.num_vertices

        covariances = extract_covariances(view)
        global_transform = mi.ScalarTransform4f(view.global_transform)  # type: ignore
        for i, v in enumerate(mesh.vertices):
            local_transform = np.eye(4)
            local_transform[:3, :3] = covariances[i] * radii[i]
            local_transform[:3, 3] = v
            local_transform = mi.ScalarTransform4f(local_transform)  # type: ignore
            shapes.append(
                {
                    "type": "ply",
                    "filename": str(filename.resolve()),
                    "face_normals": False,
                    "to_world": global_transform @ local_transform,
                }
            )

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
            assert (
                dir_attr.element_type == attr.element_type
            ), "Direction attribute must have the same element type as vector field attribute."
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
    else:
        tip_size = size

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


def generate_curve_config(view: View, stamp: str, index: int):
    assert view.data_frame is not None
    mesh = view.data_frame.mesh
    shapes: list[dict[str, Any]] = []

    # The radius of the linearcurve shape in Mitsuba 3.4.0 will not be transformed using the
    # `to_world` transform. This seems to be a bug on Mitsuba's part. Thus, we use a temporary fix
    # to bypass this bug.
    # TODO: Update once Mitsuba fixed the bug.
    scale_correction_factor = np.trace(view.global_transform[:3, :3]) / 3

    # Generate curve file
    if view.vector_field_channel is not None:
        base, ctrl_pts_1, ctrl_pts_2, tip, base_size, tip_size = extract_vector_field(
            view
        )
    else:
        # Use edges of the mesh.
        base, tip, base_size, tip_size = extract_edges(view)
        ctrl_pts_1 = ctrl_pts_2 = None

    tmp_dir = pathlib.Path(tempfile.gettempdir())
    filename = tmp_dir / f"{stamp}-view-{index:03}.txt"
    logger.debug(f"Saving curves to '{str(filename)}'.")

    assert len(base) == len(tip)
    assert len(base) == len(base_size)
    assert len(tip) == len(tip_size)
    with open(filename, "w") as fout:
        if ctrl_pts_1 is None or ctrl_pts_2 is None:
            curve_type = "linearcurve"
            for p0, p1, s0, s1 in zip(base, tip, base_size, tip_size):
                fout.write(f"{p0[0]} {p0[1]} {p0[2]} {s0 * scale_correction_factor}\n")
                fout.write(
                    f"{p1[0]} {p1[1]} {p1[2]} {s1 * scale_correction_factor}\n\n"
                )
        else:
            curve_type = "bsplinecurve"
            for p0, p1, p2, p3, s0, s3 in zip(
                base, ctrl_pts_1, ctrl_pts_2, tip, base_size, tip_size
            ):
                s1 = 0.75 * s0 + 0.25 * s3
                s2 = 0.25 * s0 + 0.75 * s3
                fout.write(f"{p0[0]} {p0[1]} {p0[2]} {s0 * scale_correction_factor}\n")
                fout.write(f"{p0[0]} {p0[1]} {p0[2]} {s0 * scale_correction_factor}\n")
                fout.write(f"{p0[0]} {p0[1]} {p0[2]} {s0 * scale_correction_factor}\n")
                fout.write(f"{p0[0]} {p0[1]} {p0[2]} {s0 * scale_correction_factor}\n")
                fout.write(f"{p1[0]} {p1[1]} {p1[2]} {s1 * scale_correction_factor}\n")
                fout.write(f"{p2[0]} {p2[1]} {p2[2]} {s2 * scale_correction_factor}\n")
                fout.write(f"{p3[0]} {p3[1]} {p3[2]} {s3 * scale_correction_factor}\n")
                fout.write(f"{p3[0]} {p3[1]} {p3[2]} {s3 * scale_correction_factor}\n")
                fout.write(f"{p3[0]} {p3[1]} {p3[2]} {s3 * scale_correction_factor}\n")
                fout.write(
                    f"{p3[0]} {p3[1]} {p3[2]} {s3 * scale_correction_factor}\n\n"
                )

    mi_config = {
        f"view_{index:03}_shape_000000": {
            "type": curve_type,
            "filename": str(filename.resolve()),
            "bsdf": generate_bsdf_config(view, is_primitive=False),
            "to_world": mi.ScalarTransform4f(view.global_transform),  # type: ignore
        }
    }
    return mi_config


def _rename_attributes(mesh: lagrange.SurfaceMesh, active_attributes: list[Attribute]):
    """Rename generic scalar and vector attribute with suffix "_0". This is required by mitsuba to
    correct parse them from a ply file.

    :param mesh: The mesh to rename attributes.
    :param active_attributes: The list of active attributes.
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
        mesh.rename_attribute(name, f"{name}_0")
        processed_names.add(name)

        # Note that we will keep attr._internal_name the same.


def generate_surface_config(view: View, stamp: str, index: int):
    """Generate the mitsuba config for a mesh.

    It does the following things:
    1. Rename all generic scalar/vector attributes with _0 suffix.
    2. Save the mesh and all active attributes in ply format in a temp directory.
    3. Generate the bsdf config associated with the shape.

    :param view: The view to generate mesh config from.
    :param stamp: The time stamp string used for creating a unique filename.
    :param index: The index of the view.

    :return: The mitsuba config for the mesh view.
    """
    assert view.data_frame is not None
    mesh = copy.copy(view.data_frame.mesh)  # Shallow copy
    if not mesh.is_triangle_mesh:
        logger.debug("Convert dataframe to triangle mesh.")
        lagrange.triangulate_polygonal_facets(mesh)

    # Rename attributes in the shallow copy of mesh.
    _rename_attributes(mesh, view._active_attributes)

    # Just a sanity check that all attributes are still present in the original mesh.
    for attr in view._active_attributes:
        name = attr._internal_name
        assert name is not None
        assert view.data_frame.mesh.has_attribute(name)

    tmp_dir = pathlib.Path(tempfile.gettempdir())
    filename = tmp_dir / f"{stamp}-view-{index:03}.ply"
    logger.debug(f"Saving mesh to '{str(filename)}'.")
    lagrange.io.save_mesh(filename, mesh)  # type: ignore

    normal_ids = mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.Normal)
    if len(normal_ids) > 0:
        normal_attr = mesh.attribute(normal_ids[0])  # type: ignore
        use_facet_normal = normal_attr.element_type == lagrange.AttributeElement.Facet
    else:
        use_facet_normal = False

    mi_config = {
        "type": "ply",
        "filename": str(filename.resolve()),
        "bsdf": generate_bsdf_config(view, is_primitive=False),
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
