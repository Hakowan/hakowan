from .bsdf import generate_bsdf_config
from ..common import logger
from ..compiler import View
from ..grammar.scale import Attribute

import lagrange
import pathlib
import re
import tempfile
from typing import Any
import copy


def generate_point_cloud_config(view: View):
    """Generate point cloud shapes from a View."""
    assert view.data_frame is not None
    mesh = view.data_frame.mesh
    shapes: list[dict[str, Any]] = []

    # Compute radii
    radii = []
    if view.size_channel is not None:
        match view.size_channel.data:
            case float():
                radii = [view.size_channel.data] * mesh.num_vertices
            case Attribute():
                assert view.size_channel.data._internal_name is not None
                radii = mesh.attribute(
                    view.size_channel.data._internal_name
                ).data.tolist()
                assert len(radii) == mesh.num_vertices
    else:
        radii = [0.01] * mesh.num_vertices

    # Generate spheres.
    for i, v in enumerate(mesh.vertices):
        shapes.append({"type": "sphere", "center": v.tolist(), "raiuds": radii[i]})

    # TODO: Generate bsdf.
    return shapes


def _rename_attributes(mesh: lagrange.SurfaceMesh, active_attributes: list[Attribute]):
    """ Rename generic scalar and vector attribute with suffix "_0". This is required by mitsuba to
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


def generate_mesh_config(view: View, stamp: str, index: int):
    """ Generate the mitsuba config for a mesh.

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
        "bsdf": generate_bsdf_config(view),
    }
    return mi_config
