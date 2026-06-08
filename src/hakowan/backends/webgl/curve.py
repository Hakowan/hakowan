"""Curve mark renderer.

Two output modes:

  * **Flat LINES** primitive when there is no ``size_channel`` and no
    vector-field arrow heads — cheap, but the line width is one pixel.
  * **Tubes** (n-sided extrusion, default 8 sides) otherwise. Each endpoint
    can carry its own radius, which lets the vector-field "arrow" end-style
    taper the cone head down to zero radius.

Vector-field features:
  * ``refinement_level`` — barycentric subdivision of base/tip positions on
    each triangle, ported from ``backends/mitsuba/shape.py:extract_vector_field``.
  * ``end_type='arrow'`` — split each vector into a constant-width shaft +
    a cone (radius 2*size → 0) at the tip.

Per-endpoint colour comes from the view's ``ScalarField`` reflectance via
``COLOR_0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import lagrange
import numpy as np

from ...common import logger
from ...compiler import View
from ...grammar.scale import Attribute

from .builder import GLTFBuilder, MODE_LINES, MODE_TRIANGLES
from .mesh_extract import _find_color_field_name, _read_color_attribute
from .material_translate import translate_material


_DEFAULT_TUBE_SIDES = 8
_DEFAULT_SIZE = 0.01


@dataclass
class _SegmentData:
    """Endpoints (E*2, 3) and per-endpoint metadata.

    ``vertex_idx`` and ``sizes`` may be None when not derivable (e.g. facet-
    based vector field).
    """

    endpoints: np.ndarray
    vertex_idx: np.ndarray | None
    sizes: np.ndarray | None


# ---------------------------------------------------------------------- #
# Mesh-edge path                                                          #
# ---------------------------------------------------------------------- #


def _extract_mesh_edges(view: View) -> _SegmentData:
    assert view.data_frame is not None
    mesh = view.data_frame.mesh
    mesh.initialize_edges()
    if mesh.num_edges == 0:
        return _SegmentData(np.zeros((0, 3), dtype=np.float32), None, None)

    endpoints = np.empty((mesh.num_edges * 2, 3), dtype=np.float32)
    vertex_idx = np.empty(mesh.num_edges * 2, dtype=np.uint32)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    for i in range(mesh.num_edges):
        v0, v1 = mesh.get_edge_vertices(i)
        endpoints[2 * i] = vertices[v0]
        endpoints[2 * i + 1] = vertices[v1]
        vertex_idx[2 * i] = v0
        vertex_idx[2 * i + 1] = v1
    return _SegmentData(endpoints, vertex_idx, None)


# ---------------------------------------------------------------------- #
# Vector-field path                                                       #
# ---------------------------------------------------------------------- #


def _refine_barycentric(
    mesh: lagrange.SurfaceMesh, data: np.ndarray, level: int
) -> np.ndarray:
    """Subdivide ``data`` (vertex-attribute, Nx3) via barycentric sampling
    of each triangle. Returns a flat (n_facets * (level+2)*(level+1)/2, 3)
    array — same ordering as Mitsuba's ``extract_vector_field.refine``.
    """
    facets = np.asarray(mesh.facets, dtype=np.uint32)
    n = level + 1
    out: list[np.ndarray] = []
    B0, B1, B2 = np.mgrid[0 : n + 1, 0 : n + 1, 0 : n + 1]
    for b0, b1, b2 in zip(B0.ravel(), B1.ravel(), B2.ravel()):
        if b0 + b1 + b2 != n:
            continue
        d = (
            data[facets[:, 0]] * b0 + data[facets[:, 1]] * b1 + data[facets[:, 2]] * b2
        ) / n
        out.append(d)
    return np.vstack(out).astype(np.float32)


def _extract_vector_field(view: View) -> _SegmentData:
    assert view.data_frame is not None
    mesh = view.data_frame.mesh
    assert view.vector_field_channel is not None
    vf = view.vector_field_channel

    attr_name = vf.data._internal_name  # type: ignore[union-attr]
    assert attr_name is not None
    attr = mesh.attribute(attr_name)
    vectors = np.asarray(attr.data, dtype=np.float32)
    element_type = attr.element_type

    if element_type == lagrange.AttributeElement.Vertex:
        base = np.asarray(mesh.vertices, dtype=np.float32)
        if vectors.shape[0] != base.shape[0]:
            logger.warning(
                f"WebGL backend: vector field length {vectors.shape[0]} != "
                f"vertex count {base.shape[0]}; truncating."
            )
            n_min = min(vectors.shape[0], base.shape[0])
            vectors = vectors[:n_min]
            base = base[:n_min]
        per_vertex_size = _resolve_per_vertex_size(view, base.shape[0])
        if vf.refinement_level > 0:
            base = _refine_barycentric(mesh, base, vf.refinement_level)
            vectors = _refine_barycentric(mesh, vectors, vf.refinement_level)
            per_vertex_size = _refine_barycentric(
                mesh,
                per_vertex_size.reshape(-1, 1).repeat(3, axis=1),
                vf.refinement_level,
            )[:, 0]
    elif element_type == lagrange.AttributeElement.Facet:
        centroid_attr_id = lagrange.compute_facet_centroid(mesh)
        base = np.asarray(
            mesh.attribute(centroid_attr_id).data,
            dtype=np.float32,  # type: ignore
        )
        per_vertex_size = _resolve_facet_size(view, mesh.num_facets)
    else:
        logger.warning(
            f"WebGL backend: vector field element type {element_type} not supported."
        )
        return _SegmentData(np.zeros((0, 3), dtype=np.float32), None, None)

    tip = base + vectors

    if vf.end_type == "arrow":
        # Each vector becomes: shaft (base → stem) at full size,
        # then cone (stem → tip) tapering 2*size → 0.
        stem = 0.25 * base + 0.75 * tip
        endpoints = np.empty((base.shape[0] * 4, 3), dtype=np.float32)
        sizes = np.empty(base.shape[0] * 4, dtype=np.float32)
        endpoints[0::4] = base
        endpoints[1::4] = stem
        endpoints[2::4] = stem
        endpoints[3::4] = tip
        sizes[0::4] = per_vertex_size  # shaft base
        sizes[1::4] = per_vertex_size  # shaft end
        sizes[2::4] = 2.0 * per_vertex_size  # cone base
        sizes[3::4] = 0.0  # cone tip
        if element_type == lagrange.AttributeElement.Vertex:
            base_idx = np.arange(base.shape[0], dtype=np.uint32)
            vertex_idx = np.repeat(base_idx, 4)
        else:
            vertex_idx = None
    else:  # "point" and "flat" both render as a plain constant-radius tube in WebGL
        endpoints = np.empty((base.shape[0] * 2, 3), dtype=np.float32)
        endpoints[0::2] = base
        endpoints[1::2] = tip
        sizes = np.repeat(per_vertex_size, 2)
        if element_type == lagrange.AttributeElement.Vertex:
            base_idx = np.arange(base.shape[0], dtype=np.uint32)
            vertex_idx = np.repeat(base_idx, 2)
        else:
            vertex_idx = None

    return _SegmentData(endpoints, vertex_idx, sizes)


def _resolve_per_vertex_size(view: View, n: int) -> np.ndarray:
    """Return one radius per vertex (length n), reading view.size_channel."""
    if view.size_channel is None:
        return np.full(n, _DEFAULT_SIZE, dtype=np.float32)
    data = view.size_channel.data
    if isinstance(data, float):
        return np.full(n, float(data), dtype=np.float32)
    if isinstance(data, Attribute):
        assert view.data_frame is not None
        name = data._internal_name
        assert name is not None
        arr = np.asarray(
            view.data_frame.mesh.attribute(name).data, dtype=np.float32
        ).reshape(-1)
        if arr.shape[0] != n:
            logger.warning(
                f"WebGL backend: size attr length {arr.shape[0]} != n {n}; "
                "padding/truncating."
            )
            if arr.shape[0] < n:
                arr = np.pad(arr, (0, n - arr.shape[0]), constant_values=_DEFAULT_SIZE)
            else:
                arr = arr[:n]
        return arr
    return np.full(n, _DEFAULT_SIZE, dtype=np.float32)


def _resolve_facet_size(view: View, n_facets: int) -> np.ndarray:
    if view.size_channel is None:
        return np.full(n_facets, _DEFAULT_SIZE, dtype=np.float32)
    data = view.size_channel.data
    if isinstance(data, float):
        return np.full(n_facets, float(data), dtype=np.float32)
    # Per-vertex attribute on facet vector field is meaningless; warn + default.
    logger.warning(
        "WebGL backend: per-vertex size attribute with facet-element vector "
        "field; using default size."
    )
    return np.full(n_facets, _DEFAULT_SIZE, dtype=np.float32)


# ---------------------------------------------------------------------- #
# Tube extrusion                                                          #
# ---------------------------------------------------------------------- #


def _perpendicular_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to ``direction`` (unit) forming a right-
    handed frame.
    """
    if abs(direction[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, direction)
    u_n = np.linalg.norm(u)
    if u_n < 1e-9:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = u / u_n
    v = np.cross(direction, u)
    return u.astype(np.float32), v.astype(np.float32)


def _extrude_tubes(
    endpoints: np.ndarray, sizes: np.ndarray, sides: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_segments = endpoints.shape[0] // 2
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cos_t = np.cos(angles).astype(np.float32)
    sin_t = np.sin(angles).astype(np.float32)

    positions = np.empty((n_segments * sides * 2, 3), dtype=np.float32)
    normals = np.empty_like(positions)
    ring_to_endpoint = np.empty(n_segments * sides * 2, dtype=np.uint32)

    for s in range(n_segments):
        p0 = endpoints[2 * s]
        p1 = endpoints[2 * s + 1]
        r0 = float(sizes[2 * s])
        r1 = float(sizes[2 * s + 1])

        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            direction = (seg / seg_len).astype(np.float32)
        u, v = _perpendicular_basis(direction)

        ring_offsets = u[None, :] * cos_t[:, None] + v[None, :] * sin_t[:, None]
        ring0 = p0[None, :] + r0 * ring_offsets
        ring1 = p1[None, :] + r1 * ring_offsets

        base = s * sides * 2
        positions[base : base + sides] = ring0
        positions[base + sides : base + 2 * sides] = ring1
        normals[base : base + sides] = ring_offsets
        normals[base + sides : base + 2 * sides] = ring_offsets

        ring_to_endpoint[base : base + sides] = 2 * s
        ring_to_endpoint[base + sides : base + 2 * sides] = 2 * s + 1

    tris = np.empty((n_segments, sides, 2, 3), dtype=np.uint32)
    side_arange = np.arange(sides, dtype=np.uint32)
    side_next = (side_arange + 1) % sides
    seg_base = np.arange(n_segments, dtype=np.uint32) * (sides * 2)
    a = seg_base[:, None] + side_arange[None, :]
    b = seg_base[:, None] + side_next[None, :]
    c = seg_base[:, None] + side_arange[None, :] + sides
    d = seg_base[:, None] + side_next[None, :] + sides
    tris[..., 0, 0] = a
    tris[..., 0, 1] = b
    tris[..., 0, 2] = d
    tris[..., 1, 0] = a
    tris[..., 1, 1] = d
    tris[..., 1, 2] = c
    indices = tris.reshape(-1)

    return positions, normals, indices, ring_to_endpoint


# ---------------------------------------------------------------------- #
# Top-level                                                               #
# ---------------------------------------------------------------------- #


def _extract_segments(view: View) -> _SegmentData:
    if view.vector_field_channel is not None:
        return _extract_vector_field(view)
    return _extract_mesh_edges(view)


def _segment_colors(view: View, vertex_idx: np.ndarray | None) -> np.ndarray | None:
    if vertex_idx is None:
        return None
    color_name = _find_color_field_name(view)
    if color_name is None:
        return None
    assert view.data_frame is not None
    per_vertex = _read_color_attribute(view.data_frame.mesh, color_name)
    return per_vertex[vertex_idx].astype(np.float32)


def add_curve_view(builder: GLTFBuilder, view: View) -> int:
    data = _extract_segments(view)
    if data.endpoints.shape[0] == 0:
        logger.warning("WebGL backend: curve view has no segments; skipping.")
        return -1

    # Decide tube-vs-lines: vector fields with arrow heads or any sizes
    # force tubes; mesh edges without size channel fall back to lines.
    use_tubes = data.sizes is not None or view.size_channel is not None
    result = translate_material(view, builder)
    pbr = result.pbr
    double_sided = result.double_sided
    # Tag the material so the viewer's render-pass switcher can treat curve
    # decorations (wireframe, seams, …) specially — extruded-tube normals
    # would otherwise overwrite the underlying surface in the normal pass.
    extras = result.extras if result.extras is not None else {}
    extras.setdefault("hakowan", {})["kind"] = "curve"
    pbr["extras"] = extras

    if use_tubes:
        if data.sizes is not None:
            sizes = data.sizes
        else:
            # Mesh-edge view with size_channel.
            sizes = _resolve_endpoint_sizes_from_edges(view, data)
        positions, normals, indices, ring_to_endpoint = _extrude_tubes(
            data.endpoints, sizes, sides=_DEFAULT_TUBE_SIDES
        )

        colors: np.ndarray | None = None
        endpoint_colors = _segment_colors(view, data.vertex_idx)
        if endpoint_colors is not None:
            colors = endpoint_colors[ring_to_endpoint]
            pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]

        material_idx = builder.add_material(pbr, double_sided=double_sided)
        transform = np.asarray(view.global_transform, dtype=np.float64)
        logger.debug(
            f"WebGL backend: curve view → {data.endpoints.shape[0] // 2} tubes "
            f"({_DEFAULT_TUBE_SIDES} sides each)."
        )
        return builder.add_mesh_node(
            positions=positions,
            indices=indices,
            normals=normals,
            colors=colors,
            material_idx=material_idx,
            mode=MODE_TRIANGLES,
            transform_4x4=transform,
        )

    indices = np.arange(data.endpoints.shape[0], dtype=np.uint32)
    colors = _segment_colors(view, data.vertex_idx)
    if colors is not None:
        pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
    material_idx = builder.add_material(pbr, double_sided=double_sided)

    transform = np.asarray(view.global_transform, dtype=np.float64)
    logger.debug(
        f"WebGL backend: curve view → {data.endpoints.shape[0] // 2} flat lines."
    )
    return builder.add_mesh_node(
        positions=data.endpoints,
        indices=indices,
        normals=None,
        colors=colors,
        material_idx=material_idx,
        mode=MODE_LINES,
        transform_4x4=transform,
    )


def _resolve_endpoint_sizes_from_edges(view: View, data: _SegmentData) -> np.ndarray:
    """Resolve per-endpoint sizes for the mesh-edge path."""
    n = data.endpoints.shape[0]
    if view.size_channel is None:
        return np.full(n, _DEFAULT_SIZE, dtype=np.float32)
    sd = view.size_channel.data
    if isinstance(sd, float):
        return np.full(n, float(sd), dtype=np.float32)
    if isinstance(sd, Attribute) and data.vertex_idx is not None:
        assert view.data_frame is not None
        name = sd._internal_name
        assert name is not None
        per_vertex = np.asarray(
            view.data_frame.mesh.attribute(name).data, dtype=np.float32
        ).reshape(-1)
        return per_vertex[data.vertex_idx].astype(np.float32)
    return np.full(n, _DEFAULT_SIZE, dtype=np.float32)
