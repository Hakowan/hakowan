"""Lagrange ``SurfaceMesh`` → glTF attribute arrays."""

from __future__ import annotations

import copy
from typing import Any

import lagrange
import numpy as np

from ...common import logger
from ...compiler import View
from ...grammar.channel.material import (
    Diffuse,
    Principled,
    Plastic,
)
from ...grammar.texture import Image, ScalarField


def _find_color_field_name(view: View) -> str | None:
    """Return the resolved color-field attribute name if the view's material
    reflectance is a ``ScalarField``; otherwise ``None``.
    """
    mat = view.material_channel
    if mat is None:
        return None
    if isinstance(mat, Diffuse):
        reflectance = mat.reflectance
    elif isinstance(mat, Principled):
        reflectance = mat.color
    elif isinstance(mat, Plastic):
        reflectance = mat.diffuse_reflectance
    else:
        return None
    if not isinstance(reflectance, ScalarField):
        return None
    name = getattr(reflectance.data, "_internal_color_field", None)
    return name if isinstance(name, str) else None


def _find_image_texture(view: View) -> Image | None:
    mat = view.material_channel
    if mat is None:
        return None
    if isinstance(mat, Diffuse):
        reflectance: Any = mat.reflectance
    elif isinstance(mat, Principled):
        reflectance = mat.color
    elif isinstance(mat, Plastic):
        reflectance = mat.diffuse_reflectance
    else:
        return None
    return reflectance if isinstance(reflectance, Image) else None


def _find_uv_attribute_name(mesh: lagrange.SurfaceMesh) -> str | None:
    uv_ids = mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV)
    if not uv_ids:
        return None
    return mesh.get_attribute_name(uv_ids[0])


def _read_uv_coordinates(
    mesh: lagrange.SurfaceMesh,
    uv_name: str,
    corner_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Read UVs as (N, 2) float32.

    When ``corner_idx`` is set (de-indexed mesh path), expand indexed UVs per
    corner. Otherwise return one UV per mesh vertex.
    """
    if mesh.is_attribute_indexed(uv_name):
        indexed = mesh.indexed_attribute(uv_name)
        values = np.asarray(indexed.values.data, dtype=np.float32)
        indices = np.asarray(indexed.indices.data, dtype=np.uint32).reshape(-1)
        assert corner_idx is not None, (
            "indexed UV attribute requires corner_idx; "
            "caller must ensure uv_is_indexed forces the de-indexed path"
        )
        if indices.shape[0] != corner_idx.shape[0]:
            raise ValueError(
                f"UV index count {indices.shape[0]} != corner count "
                f"{corner_idx.shape[0]}"
            )
        return np.ascontiguousarray(values[indices])
    data = np.asarray(mesh.attribute(uv_name).data, dtype=np.float32)
    if corner_idx is not None:
        return np.ascontiguousarray(data[corner_idx])
    return np.ascontiguousarray(data)


def _read_color_attribute(mesh: lagrange.SurfaceMesh, name: str) -> np.ndarray:
    """Read a baked color attribute. Returns (N, 4) float32 in linear RGBA."""
    raw = np.asarray(mesh.attribute(name).data, dtype=np.float32)
    if raw.ndim == 1:
        raw = np.stack([raw, raw, raw], axis=1)
    if raw.shape[1] == 3:
        alpha = np.ones((raw.shape[0], 1), dtype=np.float32)
        raw = np.concatenate([raw, alpha], axis=1)
    # hakowan stores colors in sRGB. glTF expects COLOR_0 in linear RGB.
    raw = raw.copy()
    raw[:, :3] = _srgb_to_linear_array(raw[:, :3])
    return raw


def _srgb_to_linear_array(c: np.ndarray) -> np.ndarray:
    """sRGB → linear, vectorised over an arbitrary-shape float array."""
    c = np.clip(c, 0.0, 1.0)
    low = c / 12.92
    high = ((c + 0.055) / 1.055) ** 2.4
    return np.where(c <= 0.04045, low, high).astype(np.float32)


def extract_surface_arrays(
    view: View,
    custom_attrs: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Pull positions / indices / (optional) normals / colors / UVs.

    Returns dict with keys ``positions`` (Nx3 float32), ``indices``
    (M*3 uint32, flat), ``normals`` (Nx3 float32 | None), ``colors``
    (Nx4 float32 | None), ``uvs`` (Nx2 float32 | None), and
    ``custom_attributes`` (dict | None) — the latter rewritten to match the
    final vertex layout when de-indexing happens.
    """
    assert view.data_frame is not None
    mesh = copy.copy(view.data_frame.mesh)

    if not mesh.is_triangle_mesh:
        logger.debug("WebGL backend: triangulating polygonal facets.")
        lagrange.triangulate_polygonal_facets(mesh)

    positions = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    facets = np.ascontiguousarray(mesh.facets, dtype=np.uint32)
    indices = facets.reshape(-1)

    color_name = _find_color_field_name(view)
    uv_name = _find_uv_attribute_name(mesh)
    custom_attrs = custom_attrs or {}

    # Determine which attribute supplies normals.
    # Priority: explicit Normal channel > AttributeUsage.Normal in mesh > compute.
    normal_name: str | None = None
    if view.normal_channel is not None:
        _nc_name = view.normal_channel.data._internal_name
        if _nc_name and mesh.has_attribute(_nc_name):
            normal_name = _nc_name
        else:
            logger.warning(
                "WebGL backend: normal_channel attribute not found in mesh; "
                "falling back to auto-computed normals."
            )
    if normal_name is None:
        # Compute indexed normals if the input mesh has none. We use
        # ``lagrange.compute_normal`` (rather than ``compute_vertex_normal``) so
        # sharp edges above the default 45° feature-angle threshold split into
        # distinct normals, retaining creases on hard-surface models like cubes.
        # Three.js requires a NORMAL accessor; without one the normal/lit passes
        # render black.
        normal_ids = mesh.get_matching_attribute_ids(
            usage=lagrange.AttributeUsage.Normal
        )
        if not normal_ids:
            try:
                lagrange.compute_normal(mesh)
                normal_ids = mesh.get_matching_attribute_ids(
                    usage=lagrange.AttributeUsage.Normal
                )
            except Exception as e:  # pragma: no cover - lagrange edge case
                logger.debug(
                    f"WebGL backend: compute_normal failed ({e}); "
                    "falling back to per-vertex normal computation."
                )
                try:
                    lagrange.compute_vertex_normal(mesh)
                    normal_ids = mesh.get_matching_attribute_ids(
                        usage=lagrange.AttributeUsage.Normal
                    )
                except Exception as e2:  # pragma: no cover
                    logger.debug(
                        f"WebGL backend: compute_vertex_normal also failed ({e2}); "
                        "the normal pass will render black."
                    )
        if normal_ids:
            normal_name = mesh.get_attribute_name(normal_ids[0])

    normals: np.ndarray | None = None
    per_corner_normals: np.ndarray | None = None
    if normal_name is not None:
        if mesh.is_attribute_indexed(normal_name):
            # Indexed normal: per-corner lookup into a (V_unique, 3) value
            # table — preserves sharp edges by giving the same vertex
            # different normals across crease facets.
            indexed = mesh.indexed_attribute(normal_name)
            values = np.asarray(indexed.values.data, dtype=np.float32)
            norm_idx = np.asarray(indexed.indices.data, dtype=np.uint32).reshape(-1)
            per_corner_normals = values[norm_idx]
        else:
            normal_attr = mesh.attribute(normal_name)
            if normal_attr.element_type == lagrange.AttributeElement.Vertex:
                normals = np.ascontiguousarray(normal_attr.data, dtype=np.float32)
            elif normal_attr.element_type == lagrange.AttributeElement.Facet:
                per_corner_normals = np.repeat(
                    np.asarray(normal_attr.data, dtype=np.float32), 3, axis=0
                )
            elif normal_attr.element_type == lagrange.AttributeElement.Corner:
                per_corner_normals = np.ascontiguousarray(
                    normal_attr.data, dtype=np.float32
                )
            else:
                logger.warning(
                    "WebGL backend: unsupported normal element type "
                    f"'{normal_attr.element_type}', dropping normals."
                )

    # A per-facet colour field (baked from a facet-element ScalarField) has one
    # value per facet, so it must be expanded to per-corner — which also forces
    # the de-indexed output path even when the normals are per-vertex. Without
    # this the facet-length colour array is indexed by vertex ids (IndexError)
    # or handed to the builder with a vertex-count mismatch.
    color_is_facet = (
        color_name is not None
        and not mesh.is_attribute_indexed(color_name)
        and mesh.attribute(color_name).element_type == lagrange.AttributeElement.Facet
    )
    uv_is_indexed = uv_name is not None and mesh.is_attribute_indexed(uv_name)
    corner_idx: np.ndarray | None = None
    if (color_is_facet or uv_is_indexed) and per_corner_normals is None:
        corner_idx = facets.reshape(-1)
        per_corner_normals = (
            normals[corner_idx]
            if normals is not None
            else np.zeros((corner_idx.shape[0], 3), dtype=np.float32)
        )
        normals = None

    if per_corner_normals is not None:
        # De-index every other vertex-element attribute (positions, colors,
        # UVs, custom shader attrs) to per-corner so they all line up with
        # the per-corner normal array.
        if corner_idx is None:
            corner_idx = facets.reshape(-1)
        new_positions = positions[corner_idx]
        if color_name is None:
            colors = None
        elif color_is_facet:
            # (num_facets, 4) → repeat for each of the 3 triangle corners; the
            # mesh is triangulated above, so corners are laid out facet-major.
            colors = np.repeat(_read_color_attribute(mesh, color_name), 3, axis=0)
        else:
            colors = _read_color_attribute(mesh, color_name)[corner_idx]
        uvs = (
            _read_uv_coordinates(mesh, uv_name, corner_idx)
            if uv_name is not None
            else None
        )
        remapped_custom = {
            name: np.ascontiguousarray(arr)[corner_idx]
            for name, arr in custom_attrs.items()
        }
        return {
            "positions": new_positions,
            "indices": np.arange(new_positions.shape[0], dtype=np.uint32),
            "normals": per_corner_normals,
            "colors": colors,
            "uvs": uvs,
            "custom_attributes": remapped_custom or None,
        }

    colors = _read_color_attribute(mesh, color_name) if color_name is not None else None
    uvs = _read_uv_coordinates(mesh, uv_name) if uv_name is not None else None

    return {
        "positions": positions,
        "indices": indices,
        "normals": normals,
        "colors": colors,
        "uvs": uvs,
        "custom_attributes": custom_attrs or None,
    }
