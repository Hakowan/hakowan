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

    normals: np.ndarray | None = None
    normal_ids = mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.Normal)
    if normal_ids:
        normal_attr = mesh.attribute(normal_ids[0])  # type: ignore
        if normal_attr.element_type == lagrange.AttributeElement.Vertex:
            normals = np.ascontiguousarray(normal_attr.data, dtype=np.float32)
        elif normal_attr.element_type == lagrange.AttributeElement.Facet:
            # De-index facet normals to per-corner; every other vertex-element
            # attribute (incl. custom shader attrs) must follow the same map.
            corner_idx = facets.reshape(-1)
            new_positions = positions[corner_idx]
            new_normals = np.repeat(
                np.asarray(normal_attr.data, dtype=np.float32), 3, axis=0
            )
            colors = (
                _read_color_attribute(mesh, color_name)[corner_idx]
                if color_name is not None
                else None
            )
            uvs = (
                np.asarray(mesh.attribute(uv_name).data, dtype=np.float32)[corner_idx]
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
                "normals": new_normals,
                "colors": colors,
                "uvs": uvs,
                "custom_attributes": remapped_custom or None,
            }
        else:
            logger.warning(
                "WebGL backend: unsupported normal element type "
                f"'{normal_attr.element_type}', dropping normals."
            )

    colors = (
        _read_color_attribute(mesh, color_name) if color_name is not None else None
    )
    uvs = (
        np.asarray(mesh.attribute(uv_name).data, dtype=np.float32)
        if uv_name is not None
        else None
    )

    return {
        "positions": positions,
        "indices": indices,
        "normals": normals,
        "colors": colors,
        "uvs": uvs,
        "custom_attributes": custom_attrs or None,
    }
