from .bsdf import generate_bsdf_config
from ..compiler import View
from ..grammar.scale import Attribute

import lagrange
import pathlib
import tempfile
from typing import Any


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


def generate_mesh_config(view: View, stamp: str, index: int):
    assert view.data_frame is not None
    mesh = view.data_frame.mesh

    tmp_dir = pathlib.Path(tempfile.gettempdir())
    filename = tmp_dir / f"{stamp}-view-{index:03}.ply"
    lagrange.io.save_mesh(filename, mesh) # type: ignore

    mi_config = {
        "type": "ply",
        "filename": str(filename.resolve()),
        "bsdf": generate_bsdf_config(view),
    }
    return mi_config
