"""Lagrange ``SurfaceMesh`` → glTF attribute arrays."""

from __future__ import annotations

import copy

import lagrange
import numpy as np

from ...common import logger
from ...compiler import View


def extract_surface_arrays(view: View) -> dict[str, np.ndarray | None]:
    """Pull positions / indices / (optional) normals from a view's mesh.

    Returns a dict with keys ``positions`` (Nx3 float32), ``indices``
    (M*3 uint32, flat), and ``normals`` (Nx3 float32 or None).

    Facet-element normals are promoted to per-corner by de-indexing — three
    duplicate vertices per triangle — since glTF only supports per-vertex
    attributes.
    """
    assert view.data_frame is not None
    mesh = copy.copy(view.data_frame.mesh)

    if not mesh.is_triangle_mesh:
        logger.debug("WebGL backend: triangulating polygonal facets.")
        lagrange.triangulate_polygonal_facets(mesh)

    positions = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    facets = np.ascontiguousarray(mesh.facets, dtype=np.uint32)

    normals: np.ndarray | None = None
    normal_ids = mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.Normal)
    if len(normal_ids) > 0:
        normal_attr = mesh.attribute(normal_ids[0])  # type: ignore
        if normal_attr.element_type == lagrange.AttributeElement.Vertex:
            normals = np.ascontiguousarray(normal_attr.data, dtype=np.float32)
        elif normal_attr.element_type == lagrange.AttributeElement.Facet:
            # De-index: each triangle gets three unique vertices, each carrying
            # its facet normal. This expands positions too.
            facet_normals = np.asarray(normal_attr.data, dtype=np.float32)
            new_positions = positions[facets.reshape(-1)]
            new_normals = np.repeat(facet_normals, 3, axis=0)
            new_indices = np.arange(new_positions.shape[0], dtype=np.uint32)
            return {
                "positions": new_positions,
                "indices": new_indices,
                "normals": new_normals,
            }
        else:
            logger.warning(
                "WebGL backend: unsupported normal element type "
                f"'{normal_attr.element_type}', dropping normals."
            )

    return {
        "positions": positions,
        "indices": facets.reshape(-1),
        "normals": normals,
    }
