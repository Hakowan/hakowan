from .bsdf import generate_bsdf_config
from ..common import logger
from ..compiler import View
from ..grammar.scale import Attribute

from typing import Any
import copy
import lagrange
import mitsuba as mi
import numpy as np
import numpy.typing as npt
import pathlib
import re
import tempfile


def extract_size(view: View, n: int, default_size=0.01):
    """Extract the size attribute from a view.

    :param view: The view to extract size from.
    :param n: The cardinality of size attribute.
    :param default_size: The default size if size attribute is not specified.

    :return: A list of size values of length n.
    """
    assert view.data_frame is not None
    mesh = view.data_frame.mesh

    radii = []
    if view.size_channel is not None:
        match view.size_channel.data:
            case float():
                radii = [view.size_channel.data] * n
            case Attribute():
                assert view.size_channel.data._internal_name is not None
                radii = mesh.attribute(
                    view.size_channel.data._internal_name
                ).data.tolist()
                assert len(radii) == n
            case _:
                raise NotImplementedError(
                    f"Unsupported size channel type: {type(view.size_channel.data)}"
                )
    else:
        radii = [default_size] * n

    return radii


def generate_point_config(view: View):
    """Generate point cloud shapes from a View."""
    assert view.data_frame is not None
    mesh = view.data_frame.mesh
    shapes: list[dict[str, Any]] = []

    # Compute radii
    radii = extract_size(view, mesh.num_vertices)

    # Generate spheres.
    assert len(radii) == mesh.num_vertices
    for i, v in enumerate(mesh.vertices):
        shapes.append({"type": "sphere", "center": v.tolist(), "radius": radii[i]})

    # Assign transform
    for shape in shapes:
        shape["to_world"] = mi.Transform4f(view.global_transform)

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
    return shapes


def extract_vector_field(view: View):
    assert view.data_frame is not None
    mesh = view.data_frame.mesh

    assert view.vector_field_channel is not None
    attr_name = view.vector_field_channel.data._internal_name
    assert attr_name is not None
    assert mesh.has_attribute(attr_name)

    attr = mesh.attribute(attr_name)
    match attr.element_type:
        case lagrange.AttributeElement.Vertex:
            base = mesh.vertices
            size = extract_size(view, mesh.num_vertices)
        case lagrange.AttributeElement.Facet:
            centroid_attr_id = lagrange.compute_facet_centroid(mesh)
            base = mesh.attribute(centroid_attr_id).data  # type: ignore
            size = extract_size(view, mesh.num_facets)
        case _:
            raise NotImplementedError(
                f"Unsupported vector field element type: {attr.element_type}"
            )
    tip = attr.data + base
    return [base, tip, size, size]


def extract_edges(view: View):
    assert view.data_frame is not None
    mesh = view.data_frame.mesh
    mesh.initialize_edges()

    base: npt.NDArray = np.ndarray((mesh.num_edges, mesh.dimension), dtype=np.float32)
    tip: npt.NDArray = np.ndarray((mesh.num_edges, mesh.dimension), dtype=np.float32)
    base_size: npt.NDArray = np.ndarray(mesh.num_edges, dtype=np.float32)
    tip_size: npt.NDArray = np.ndarray(mesh.num_edges, dtype=np.float32)

    sizes = extract_size(view, mesh.num_vertices)
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

    # Generate curve file
    if view.vector_field_channel is not None:
        base, tip, base_size, tip_size = extract_vector_field(view)
    else:
        # Use edges of the mesh.
        base, tip, base_size, tip_size = extract_edges(view)

    tmp_dir = pathlib.Path(tempfile.gettempdir())
    filename = tmp_dir / f"{stamp}-view-{index:03}.txt"
    logger.info(f"Saving curves to {str(filename)}")

    assert len(base) == len(tip)
    assert len(base) == len(base_size)
    assert len(tip) == len(tip_size)
    with open(filename, "w") as fout:
        for p0, p1, s0, s1 in zip(base, tip, base_size, tip_size):
            fout.write(f"{p0[0]} {p0[1]} {p0[2]} {s0}\n")
            fout.write(f"{p1[0]} {p1[1]} {p1[2]} {s1}\n\n")

    mi_config = {
        "type": "linearcurve",
        "filename": str(filename.resolve()),
        "bsdf": generate_bsdf_config(view, is_primitive=False),
        "to_world": mi.Transform4f(view.global_transform),
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

    # Rename attributes in the shallow copy of mesh.
    _rename_attributes(mesh, view._active_attributes)

    # Just a sanity check that all attributes are still present in the original mesh.
    for attr in view._active_attributes:
        name = attr._internal_name
        assert name is not None
        assert view.data_frame.mesh.has_attribute(name)

    tmp_dir = pathlib.Path(tempfile.gettempdir())
    filename = tmp_dir / f"{stamp}-view-{index:03}.ply"
    logger.info(f"Saving mesh to {str(filename)}")
    lagrange.io.save_mesh(filename, mesh)  # type: ignore

    mi_config = {
        "type": "ply",
        "filename": str(filename.resolve()),
        "bsdf": generate_bsdf_config(view, is_primitive=False),
        "to_world": mi.Transform4f(view.global_transform),
    }
    return mi_config
