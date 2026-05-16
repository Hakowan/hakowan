"""Streamlines on triangle meshes via face-hopping.

Directions are traced intrinsically: when a streamline crosses an edge, the
direction is parallel-transported into the adjacent face's local frame.  For
4-RoSy cross fields the transported direction is snapped to the nearest of the
four arms in the next face.
"""

from __future__ import annotations

import lagrange
import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _poisson_disk_seeds(
    centroids: npt.NDArray,
    n: int,
    min_dist: float,
    rng: np.random.Generator,
) -> npt.NDArray:
    """Return up to *n* face indices with pairwise distance >= min_dist.

    Uses a grid-accelerated dart-throwing approach: O(num_faces) expected
    for reasonable min_dist values.
    """
    num_faces = centroids.shape[0]
    cell = min_dist / np.sqrt(3)  # grid cell side so one cell holds one disk

    # Map centroids to grid cells
    lo = centroids.min(axis=0)
    cell_idx = ((centroids - lo) / cell).astype(np.int64)

    grid: dict[tuple, int] = {}  # cell -> face index of occupant seed
    selected: list[int] = []

    order = rng.permutation(num_faces)
    for fi in order:
        if len(selected) >= n:
            break
        ci = tuple(cell_idx[fi])
        pt = centroids[fi]

        # Check all neighboring cells within radius min_dist
        r = int(np.ceil(min_dist / cell))
        conflict = False
        for dx in range(-r, r + 1):
            if conflict:
                break
            for dy in range(-r, r + 1):
                if conflict:
                    break
                for dz in range(-r, r + 1):
                    nb = (ci[0] + dx, ci[1] + dy, ci[2] + dz)
                    occ = grid.get(nb)
                    if occ is not None:
                        if float(np.linalg.norm(pt - centroids[occ])) < min_dist:
                            conflict = True
                            break
        if not conflict:
            grid[ci] = fi
            selected.append(fi)

    return np.array(selected, dtype=np.int64)


def _compute_streamlines(
    mesh: lagrange.SurfaceMesh,
    vec_field_attr: str,
    *,
    n: int = 50,
    cross_field: bool = True,
    num_steps: int = 200,
    step_factor: float = 0.4,
    seed: int = 0,
    min_length: int = 3,
    poisson_disk: bool = False,
    min_seed_dist: float | None = None,
) -> lagrange.SurfaceMesh:
    """Compute surface streamlines from a per-facet vector/cross field.

    ``n`` face centroids are chosen at random and used as seeds.  Each seed
    is traced bidirectionally (forward + backward) along every field arm:

    - **cross field** (``cross_field=True``): two bidirectional streamlines
      per seed — one along the representative arm *d* and one along its
      90° rotation *perp(d)* — giving up to ``2n`` streamlines in total.
    - **plain vector field** (``cross_field=False``): one bidirectional
      streamline per seed, giving up to ``n`` streamlines in total.

    Args:
        mesh: Triangulated surface mesh.
        vec_field_attr: Name of the per-facet 3D vector attribute.  Vertex-
            or corner-domain attributes are averaged to per-facet first.
        n: Number of seed faces to sample.
        cross_field: Treat the field as 4-RoSy.  At each edge crossing the
            transported direction snaps to the nearest of the four arms.
        num_steps: Maximum face-hopping steps per half-trace (forward and
            backward each get this budget).  Default 200.
        step_factor: Step length as a fraction of the mean face inradius.
            Default 0.4.
        seed: RNG seed for the random face selection.
        min_length: Discard streamlines shorter than this many sample points.
        poisson_disk: Use Poisson-disk seeding for even spatial distribution.
        min_seed_dist: Minimum distance between seeds.  Auto-computed if None.

    Returns:
        A :class:`lagrange.SurfaceMesh` with only vertices (no faces).  Each
        vertex is a sample point on a streamline.  A per-vertex ``int32``
        attribute ``_hakowan_streamline_id`` identifies which streamline each
        point belongs to; vertices within the same streamline are stored in
        consecutive order.
    """
    if mesh.vertex_per_facet != 3:
        raise ValueError("Streamline transform requires a triangle mesh.")
    if not mesh.has_attribute(vec_field_attr):
        raise ValueError(f"Mesh has no attribute '{vec_field_attr}'.")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    facets = np.asarray(mesh.facets, dtype=np.int64).reshape(-1, 3)
    num_faces = facets.shape[0]

    e1, e2, normals = _build_face_frames(vertices, facets)
    verts_2d = _build_face_verts_2d(vertices, facets, e1, e2)
    h = float(step_factor * _face_inradius(verts_2d).mean())

    vec_field_3d = _resolve_facet_vector_field(mesh, vec_field_attr, num_faces, facets)
    vec_2d = _project_to_2d(vec_field_3d, e1, e2)

    mesh.initialize_edges()
    adj_face, adj_edge = _build_face_adjacency(mesh, num_faces)

    centroids_2d = verts_2d.mean(axis=1)   # (F, 2)
    centroids_3d = vertices[facets].mean(axis=1)  # (F, 3)

    rng = np.random.default_rng(seed)
    if poisson_disk:
        if min_seed_dist is None:
            lo = centroids_3d.min(axis=0)
            hi = centroids_3d.max(axis=0)
            diagonal = float(np.linalg.norm(hi - lo))
            min_seed_dist = diagonal / float(np.sqrt(n)) if n > 0 else diagonal
        seed_faces = _poisson_disk_seeds(centroids_3d, n, min_seed_dist, rng)
    else:
        seed_faces = rng.choice(num_faces, size=min(n, num_faces), replace=False)

    all_polylines: list[npt.NDArray] = []

    for fi in seed_faces:
        d = vec_2d[fi].copy()
        p = centroids_2d[fi].copy()
        seed_pt = centroids_3d[fi]

        # Directions to trace: d-arm and (if cross field) perp-arm.
        arms = [d]
        if cross_field:
            arms.append(np.array([-d[1], d[0]]))

        for arm in arms:
            fwd = _trace_streamline_3d(
                fi, p, arm, num_steps, h,
                verts_2d, vec_2d, adj_face, adj_edge,
                vertices, facets, e1, e2, normals, centroids_2d, cross_field,
            )
            bwd = _trace_streamline_3d(
                fi, p, -arm, num_steps, h,
                verts_2d, vec_2d, adj_face, adj_edge,
                vertices, facets, e1, e2, normals, centroids_2d, cross_field,
            )
            pts = np.array(
                list(reversed(bwd)) + [seed_pt] + fwd, dtype=np.float64
            )
            if len(pts) >= min_length:
                all_polylines.append(pts)

    out_mesh = lagrange.SurfaceMesh()
    if not all_polylines:
        return out_mesh

    all_pts = np.concatenate(all_polylines, axis=0)
    out_mesh.add_vertices(all_pts)

    # Encode each consecutive pair of points as a 2-vertex polygonal face.
    segments = []
    ids = np.empty(all_pts.shape[0], dtype=np.int32)
    offset = 0
    for sid, pl in enumerate(all_polylines):
        ids[offset:offset + len(pl)] = sid
        for k in range(len(pl) - 1):
            segments.append([offset + k, offset + k + 1])
        offset += len(pl)

    if segments:
        out_mesh.add_polygons(np.array(segments, dtype=np.uint32))

    out_mesh.create_attribute(
        "_hakowan_streamline_id",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=ids,
    )

    return out_mesh


# ---------------------------------------------------------------------------
# Streamline tracing
# ---------------------------------------------------------------------------


def _trace_streamline_3d(
    f0: int,
    p0_2d: npt.NDArray,
    d0: npt.NDArray,
    num_steps: int,
    h: float,
    verts_2d: npt.NDArray,
    vec_2d: npt.NDArray,
    adj_face: npt.NDArray,
    adj_edge: npt.NDArray,
    vertices: npt.NDArray,
    facets: npt.NDArray,
    e1: npt.NDArray,
    e2: npt.NDArray,
    normals: npt.NDArray,
    centroids_2d: npt.NDArray,
    cross_field: bool,
) -> list:
    """Trace one half-streamline; returns list of 3D points (not including seed)."""
    pts: list = []
    f = f0
    p = p0_2d.copy()
    d = d0.copy()

    for _ in range(num_steps):
        d_norm = float(np.hypot(d[0], d[1]))
        if d_norm < 1e-12:
            break
        d = d / d_norm

        remaining = h
        terminate = False
        for _inner in range(32):
            t_exit, edge_idx = _ray_exits_face(p, d, verts_2d[f])
            # No forward exit: the position has drifted outside the triangle
            # (numerical issue), or the direction is degenerate.  Stop the
            # trace rather than stepping into the void on an extended plane.
            if t_exit == np.inf:
                terminate = True
                break
            if t_exit >= remaining:
                p = p + remaining * d
                remaining = 0.0
                break
            p = p + t_exit * d
            remaining -= t_exit
            fp = int(adj_face[f, edge_idx])
            if fp < 0:
                terminate = True
                break
            ep = int(adj_edge[f, edge_idx])
            p = _transition_point(p, edge_idx, f, fp, ep, verts_2d)
            d = _transport_direction(d, f, fp, edge_idx, vertices, facets, e1, e2, normals)
            if cross_field:
                d = _snap_to_cross(d, vec_2d[fp])
            f = fp
            # Nudge toward the new face's centroid (always interior) instead of
            # along d, which after snap may point back across the entry edge
            # and push p outside fp.
            to_c = centroids_2d[f] - p
            to_c_norm = float(np.hypot(to_c[0], to_c[1]))
            if to_c_norm > 1e-20:
                p = p + (1e-7 / to_c_norm) * to_c

        if terminate:
            break

        v0 = vertices[facets[f, 0]]
        pts.append(v0 + p[0] * e1[f] + p[1] * e2[f])

    return pts


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _resolve_facet_vector_field(mesh, attr_name, num_faces, facets):
    if mesh.is_attribute_indexed(attr_name):
        idx = mesh.indexed_attribute(attr_name)
        values = np.asarray(idx.values.data, dtype=np.float64)
        indices = np.asarray(idx.indices.data).reshape(-1)
        return values[indices].reshape(num_faces, 3, 3).mean(axis=1)
    attr = mesh.attribute(attr_name)
    data = np.asarray(attr.data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elem = attr.element_type
    if elem == lagrange.AttributeElement.Facet:
        return data
    if elem == lagrange.AttributeElement.Corner:
        return data.reshape(num_faces, 3, 3).mean(axis=1)
    if elem == lagrange.AttributeElement.Vertex:
        return data[facets].mean(axis=1)
    raise NotImplementedError(f"Unsupported element type: {elem}")


def _project_to_2d(vec_field_3d, e1, e2):
    vec_2d = np.stack([
        np.einsum("fi,fi->f", vec_field_3d, e1),
        np.einsum("fi,fi->f", vec_field_3d, e2),
    ], axis=1)
    norms = np.linalg.norm(vec_2d, axis=1)
    safe = norms > 1e-12
    vec_2d[safe] /= norms[safe, None]
    vec_2d[~safe] = np.array([1.0, 0.0])
    return vec_2d


def _build_face_frames(vertices, facets):
    p0, p1, p2 = vertices[facets[:, 0]], vertices[facets[:, 1]], vertices[facets[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    e1 = p1 - p0
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-20)
    e2 = np.cross(normals, e1)
    return e1, e2, normals


def _build_face_verts_2d(vertices, facets, e1, e2):
    p = vertices[facets]
    rel = p - p[:, 0:1, :]
    return np.stack([
        np.einsum("fij,fj->fi", rel, e1),
        np.einsum("fij,fj->fi", rel, e2),
    ], axis=2)


def _face_inradius(verts_2d):
    a = verts_2d[:, 1] - verts_2d[:, 0]
    b = verts_2d[:, 2] - verts_2d[:, 1]
    c = verts_2d[:, 0] - verts_2d[:, 2]
    la = np.linalg.norm(a, axis=1)
    lb = np.linalg.norm(b, axis=1)
    lc = np.linalg.norm(c, axis=1)
    s = 0.5 * (la + lb + lc)
    cross_z = a[:, 0] * (-c[:, 1]) - a[:, 1] * (-c[:, 0])
    return 0.5 * np.abs(cross_z) / np.maximum(s, 1e-20)


def _build_face_adjacency(mesh, num_faces):
    corner_to_edge = np.asarray(
        mesh.attribute(mesh.attr_name_corner_to_edge).data
    ).astype(np.int64)
    adj_face = -np.ones((num_faces, 3), dtype=np.int64)
    adj_edge = -np.ones((num_faces, 3), dtype=np.int64)
    for f in range(num_faces):
        for i in range(3):
            e = int(corner_to_edge[3 * f + i])
            for fp in mesh.facets_around_edge(e):
                fp = int(fp)
                if fp == f:
                    continue
                adj_face[f, i] = fp
                for j in range(3):
                    if int(corner_to_edge[3 * fp + j]) == e:
                        adj_edge[f, i] = j
                        break
                break
    return adj_face, adj_edge


def _ray_exits_face(p, d, tri_2d):
    best_t = np.inf
    best_e = -1
    for i in range(3):
        v0 = tri_2d[i]
        v1 = tri_2d[(i + 1) % 3]
        edge = v1 - v0
        det = d[0] * (-edge[1]) - d[1] * (-edge[0])
        if abs(det) < 1e-14:
            continue
        rhs0 = v0[0] - p[0]
        rhs1 = v0[1] - p[1]
        t = (rhs0 * (-edge[1]) - rhs1 * (-edge[0])) / det
        s = (d[0] * rhs1 - d[1] * rhs0) / det
        if t > 1e-9 and -1e-7 <= s <= 1.0 + 1e-7:
            if t < best_t:
                best_t = t
                best_e = i
    return best_t, best_e


def _transition_point(p_in_f, edge_idx, f, fp, edge_idx_in_fp, verts_2d):
    a = verts_2d[f, edge_idx]
    b = verts_2d[f, (edge_idx + 1) % 3]
    seg = b - a
    seg_len_sq = float(seg[0] * seg[0] + seg[1] * seg[1])
    if seg_len_sq < 1e-24:
        return verts_2d[fp, edge_idx_in_fp].copy()
    s = float(((p_in_f[0] - a[0]) * seg[0] + (p_in_f[1] - a[1]) * seg[1]) / seg_len_sq)
    s = min(max(s, 0.0), 1.0)
    a_fp = verts_2d[fp, (edge_idx_in_fp + 1) % 3]
    b_fp = verts_2d[fp, edge_idx_in_fp]
    return a_fp + s * (b_fp - a_fp)


def _transport_direction(d_2d, f, fp, edge_idx_in_f, vertices, facets, e1, e2, normals):
    d_3d = e1[f] * d_2d[0] + e2[f] * d_2d[1]
    v0 = vertices[facets[f, edge_idx_in_f]]
    v1 = vertices[facets[f, (edge_idx_in_f + 1) % 3]]
    edge_vec = v1 - v0
    edge_len = float(np.linalg.norm(edge_vec))
    if edge_len < 1e-20:
        return np.array([
            d_2d[0] * np.dot(e1[f], e1[fp]) + d_2d[1] * np.dot(e2[f], e1[fp]),
            d_2d[0] * np.dot(e1[f], e2[fp]) + d_2d[1] * np.dot(e2[f], e2[fp]),
        ])
    edge_unit = edge_vec / edge_len
    par = np.dot(d_3d, edge_unit) * edge_unit
    per = d_3d - par
    perp_f = np.cross(normals[f], edge_unit)
    perp_f_norm = float(np.linalg.norm(perp_f))
    if perp_f_norm < 1e-20:
        d_3d_new = d_3d
    else:
        perp_f /= perp_f_norm
        coef = float(np.dot(per, perp_f))
        perp_fp = np.cross(normals[fp], edge_unit)
        perp_fp_norm = float(np.linalg.norm(perp_fp))
        if perp_fp_norm < 1e-20:
            d_3d_new = par
        else:
            perp_fp /= perp_fp_norm
            d_3d_new = par + coef * perp_fp
    return np.array([float(np.dot(d_3d_new, e1[fp])), float(np.dot(d_3d_new, e2[fp]))])


def _snap_to_cross(d_2d, c_2d):
    candidates = (
        c_2d,
        np.array([-c_2d[1], c_2d[0]]),
        -c_2d,
        np.array([c_2d[1], -c_2d[0]]),
    )
    best = candidates[0]
    best_dot = float(np.dot(d_2d, best))
    for cand in candidates[1:]:
        dot = float(np.dot(d_2d, cand))
        if dot > best_dot:
            best_dot = dot
            best = cand
    return best.copy()
