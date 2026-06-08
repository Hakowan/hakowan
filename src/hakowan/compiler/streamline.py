"""Streamlines on triangle meshes via exact edge-crossing tracing.

Directions are transported intrinsically across edges using a rotation matrix
that aligns the source face normal to the destination face normal.  For 4-RoSy
cross fields the transported direction is snapped to the nearest arm in the
next face.  Seeding uses lagrange blue-noise sampling for even coverage.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os

import lagrange
import numpy as np
import numpy.typing as npt

from ..common import logger


# ---------------------------------------------------------------------------
# Per-worker shared state (populated by _worker_init, read by _trace_seed_task)
# ---------------------------------------------------------------------------

_W: dict = {}


def _worker_init(
    vertices, facets, normals, e1, e2, vec_2d, adj_face, adj_edge, cross_field
):
    _W["vertices"] = vertices
    _W["facets"] = facets
    _W["normals"] = normals
    _W["e1"] = e1
    _W["e2"] = e2
    _W["vec_2d"] = vec_2d
    _W["adj_face"] = adj_face
    _W["adj_edge"] = adj_edge
    _W["cross_field"] = cross_field


def _trace_seed_task(task):
    """Worker entry point: trace one seed (both arms, both directions)."""
    fi, seed_pt, arms, max_length, min_length = task
    w = _W
    num_faces = w["facets"].shape[0]
    polylines = []
    for arm in arms:
        fwd = _trace_half(
            fi,
            seed_pt,
            arm,
            max_length,
            num_faces,
            w["vertices"],
            w["facets"],
            w["normals"],
            w["e1"],
            w["e2"],
            w["vec_2d"],
            w["adj_face"],
            w["adj_edge"],
            w["cross_field"],
        )
        bwd = _trace_half(
            fi,
            seed_pt,
            -arm,
            max_length,
            num_faces,
            w["vertices"],
            w["facets"],
            w["normals"],
            w["e1"],
            w["e2"],
            w["vec_2d"],
            w["adj_face"],
            w["adj_edge"],
            w["cross_field"],
        )
        pts = np.array(list(reversed(bwd)) + [seed_pt] + fwd, dtype=np.float64)
        if len(pts) >= min_length:
            polylines.append(pts)
    return polylines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _compute_streamlines(
    mesh: lagrange.SurfaceMesh,
    vec_field_attr: str,
    *,
    n: int = 50,
    cross_field: bool = True,
    length: float | None = None,
    seed: int = 0,
    min_length: int = 3,
) -> lagrange.SurfaceMesh:
    """Compute surface streamlines from a per-facet vector/cross field.

    ``n`` face centroids chosen via blue-noise sampling are used as seeds.
    Each seed is traced bidirectionally (forward + backward) along every field
    arm:

    - **cross field** (``cross_field=True``): two bidirectional streamlines
      per seed — one along the representative arm *d* and one along its
      90° rotation *perp(d)* — giving up to ``2n`` streamlines in total.
    - **plain vector field** (``cross_field=False``): one bidirectional
      streamline per seed, giving up to ``n`` streamlines in total.

    Tracing uses exact edge crossings (scale-independent) and parallel
    transport via rotation matrices.

    Args:
        mesh: Triangulated surface mesh.
        vec_field_attr: Name of the per-facet 3D vector attribute.  Vertex-
            or corner-domain attributes are averaged to per-facet first.
        n: Number of seed faces to sample.
        cross_field: Treat the field as 4-RoSy.  At each edge crossing the
            transported direction snaps to the nearest of the four arms.
        length: Maximum object-space length per half-trace.  ``None`` means no
            limit (trace until mesh boundary).
        seed: RNG seed used when blue-noise sampling falls back to random.
        min_length: Discard streamlines shorter than this many sample points.

    Returns:
        A :class:`lagrange.SurfaceMesh` whose vertices are streamline sample
        points and whose 2-vertex polygons encode the line segments connecting
        them (suitable for the ``curve`` mark).  A per-vertex ``int32``
        attribute ``_hakowan_streamline_id`` identifies which streamline each
        point belongs to; vertices within the same streamline are stored in
        consecutive order.
    """
    if mesh.vertex_per_facet != 3:
        raise ValueError("Streamline transform requires a triangle mesh.")
    if not mesh.has_attribute(vec_field_attr):
        raise ValueError(f"Mesh has no attribute '{vec_field_attr}'.")
    if n <= 0 or mesh.num_facets == 0:
        return lagrange.SurfaceMesh()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    facets = np.asarray(mesh.facets, dtype=np.int64).reshape(-1, 3)
    num_faces = facets.shape[0]

    e1, e2, normals = _build_face_frames(vertices, facets)

    vec_field_3d = _resolve_facet_vector_field(
        mesh, vec_field_attr, num_faces, facets, cross_field=cross_field
    )
    vec_2d = _project_to_2d(vec_field_3d, e1, e2)

    mesh.initialize_edges()
    adj_face, adj_edge = _build_face_adjacency(mesh, num_faces)

    centroids_3d = vertices[facets].mean(axis=1)  # (F, 3)

    try:
        _, seed_facet_ids, _ = lagrange.sampling.blue_noise_sample(mesh, n)
        seed_faces = np.unique(seed_facet_ids)
    except AttributeError:
        rng = np.random.default_rng(seed)
        seed_faces = rng.choice(num_faces, size=min(n, num_faces), replace=False)

    if len(seed_faces) == 0:
        return lagrange.SurfaceMesh()

    tasks = []
    for fi in seed_faces:
        seed_pt = centroids_3d[fi]
        d_2d = vec_2d[fi].copy()
        d = d_2d[0] * e1[fi] + d_2d[1] * e2[fi]
        arms = [d]
        if cross_field:
            perp_2d = np.array([-d_2d[1], d_2d[0]])
            arms.append(perp_2d[0] * e1[fi] + perp_2d[1] * e2[fi])
        tasks.append((fi, seed_pt, arms, length, min_length))

    shared = (
        vertices,
        facets,
        normals,
        e1,
        e2,
        vec_2d,
        adj_face,
        adj_edge,
        cross_field,
    )
    n_workers = min(len(tasks), os.cpu_count() or 1)

    # Pool setup may fail on some platforms or when the OS rejects new worker
    # processes (resource limits).  "spawn" is safe across all platforms
    # (including Windows) and avoids fork-in-multithreaded-process deadlocks.
    pool_cm = None
    try:
        ctx = multiprocessing.get_context("spawn")
        pool_cm = concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=shared,
        )
    except (ValueError, OSError, NotImplementedError) as e:
        logger.debug(
            f"Streamline parallel pool unavailable, falling back to sequential: {e}"
        )
        pool_cm = None

    all_polylines: list[npt.NDArray] = []
    if pool_cm is not None:
        with pool_cm as pool:
            for polylines in pool.map(_trace_seed_task, tasks):
                all_polylines.extend(polylines)
    else:
        _worker_init(*shared)
        for task in tasks:
            all_polylines.extend(_trace_seed_task(task))

    out_mesh = lagrange.SurfaceMesh()
    if not all_polylines:
        return out_mesh

    all_pts = np.concatenate(all_polylines, axis=0)
    out_mesh.add_vertices(all_pts)

    segments = []
    ids = np.empty(all_pts.shape[0], dtype=np.int32)
    offset = 0
    for sid, pl in enumerate(all_polylines):
        ids[offset : offset + len(pl)] = sid
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


def _trace_half(
    f0: int,
    p0: npt.NDArray,
    d0: npt.NDArray,
    max_length: float | None,
    max_steps: int,
    vertices: npt.NDArray,
    facets: npt.NDArray,
    normals: npt.NDArray,
    e1: npt.NDArray,
    e2: npt.NDArray,
    vec_2d: npt.NDArray,
    adj_face: npt.NDArray,
    adj_edge: npt.NDArray,
    cross_field: bool,
) -> list:
    """Trace one half-streamline via exact edge crossings.

    Returns list of 3D crossing points (not including the seed point).
    """
    pts: list = []
    f = f0
    p = p0.copy()
    d = d0.copy()
    d_norm = float(np.linalg.norm(d))
    if d_norm < 1e-12:
        return pts
    d = d / d_norm
    accumulated = 0.0
    entry_local = -1  # local edge index we entered from (-1 for seed centroid)

    for _ in range(max_steps):
        n = normals[f]
        perp = np.cross(n, d)
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm < 1e-12:
            break
        perp = perp / perp_norm

        vi = facets[f]
        vals = np.array([float(np.dot(vertices[vi[k]] - p, perp)) for k in range(3)])

        # Nudge near-zero vertex values to avoid degenerate exit choices
        for k in range(3):
            if abs(vals[k]) < 1e-10:
                vals[k] = 1e-10

        # Pick the exit edge. The sign change of ``vals`` (perpendicular
        # distance) marks where the *infinite* line through ``p`` along ``±d``
        # crosses an edge, so we must keep only the *forward* crossing
        # (``dot(exit - p, d) > 0``) and, of those, the nearest one. Without the
        # forward test a near-vertex / sliver crossing can select the backward
        # intersection, which makes the streamline jump straight backwards
        # (a 180° reversal).
        exit_local = -1
        exit_pt: npt.NDArray | None = None
        best_fwd = np.inf
        for k in range(3):
            if k == entry_local:
                continue
            a, b = k, (k + 1) % 3
            if vals[a] * vals[b] < 0:
                t = vals[a] / (vals[a] - vals[b])
                cand = vertices[vi[a]] + t * (vertices[vi[b]] - vertices[vi[a]])
                fwd = float(np.dot(cand - p, d))
                if fwd <= 0.0:
                    continue  # backward crossing — skip
                if fwd < best_fwd:
                    best_fwd = fwd
                    exit_local = k
                    exit_pt = cand

        if exit_local < 0 or exit_pt is None:
            break

        step_len = float(np.linalg.norm(exit_pt - p))
        if max_length is not None and accumulated + step_len > max_length:
            break
        accumulated += step_len

        pts.append(exit_pt)
        p = exit_pt

        fp = int(adj_face[f, exit_local])
        if fp < 0:
            break

        entry_local = int(adj_edge[f, exit_local])

        # Parallel-transport direction across the edge
        R = _rotation_matrix(normals[f], normals[fp])
        d = R @ d

        if cross_field:
            d_2d = np.array([float(np.dot(d, e1[fp])), float(np.dot(d, e2[fp]))])
            d_2d = _snap_to_cross(d_2d, vec_2d[fp])
            d = d_2d[0] * e1[fp] + d_2d[1] * e2[fp]

        d_norm = float(np.linalg.norm(d))
        if d_norm < 1e-12:
            break
        d = d / d_norm
        f = fp

    return pts


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _rotation_matrix(n1: npt.NDArray, n2: npt.NDArray) -> npt.NDArray:
    """3x3 rotation matrix that rotates n1 onto n2."""
    a = n1 / np.linalg.norm(n1)
    b = n2 / np.linalg.norm(n2)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        perp = (
            np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        )
        perp = perp - np.dot(perp, a) * a
        perp /= np.linalg.norm(perp)
        return 2.0 * np.outer(perp, perp) - np.eye(3)
    kmat = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1.0 - c) / (s**2))


def _snap_to_cross(d_2d: npt.NDArray, c_2d: npt.NDArray) -> npt.NDArray:
    """Snap 2D direction d to the nearest arm of the 4-RoSy cross field c."""
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


def _resolve_facet_vector_field(mesh, attr_name, num_faces, facets, cross_field=False):
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
        n_rosy = 4 if cross_field else 1
        return _average_vertex_field_to_facets(mesh, data, num_faces, facets, n_rosy)
    raise NotImplementedError(f"Unsupported element type: {elem}")


def _average_vertex_field_to_facets(mesh, vert_field, num_faces, facets, n_rosy):
    """Average a per-vertex N-RoSy field onto per-facet directions.

    Uses Levi-Civita parallel transport (with N-fold symmetry) to bring each
    vertex direction into the facet's tangent frame, applies the xN angle
    trick, sums the three vertex contributions, divides the resulting angle
    by N, and lifts back to 3D.

    n_rosy = 1 for plain vector field, 4 for cross field.
    """
    import lagrange.polyddg as polyddg

    ops = polyddg.DifferentialOperators(mesh)
    out = np.zeros((num_faces, 3), dtype=np.float64)
    n = float(n_rosy)

    for fid in range(num_faces):
        fb = ops.facet_basis(fid)  # 3x2
        accum_xn = np.zeros(2)
        for lv in range(3):
            vid = int(facets[fid, lv])
            tv = vert_field[vid]
            if np.linalg.norm(tv) < 1e-12:
                continue
            vb = ops.vertex_basis(vid)  # 3x2
            conn = ops.levi_civita_nrosy(fid, lv, n=n_rosy)  # 2x2 (vertex->facet)

            local_v = vb.T @ tv
            angle = float(np.arctan2(local_v[1], local_v[0]))
            local_v_xn = np.array([np.cos(n * angle), np.sin(n * angle)])
            accum_xn += conn @ local_v_xn

        norm = float(np.linalg.norm(accum_xn))
        if norm < 1e-12:
            continue
        angle_xn = float(np.arctan2(accum_xn[1], accum_xn[0]))
        angle_f = angle_xn / n
        local_f = np.array([np.cos(angle_f), np.sin(angle_f)])
        out[fid] = fb @ local_f

    return out


def _project_to_2d(vec_field_3d, e1, e2):
    vec_2d = np.stack(
        [
            np.einsum("fi,fi->f", vec_field_3d, e1),
            np.einsum("fi,fi->f", vec_field_3d, e2),
        ],
        axis=1,
    )
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


def _build_face_adjacency(mesh, num_faces):
    corner_to_edge = np.asarray(
        mesh.attribute(mesh.attr_name_corner_to_edge).data
    ).astype(np.int64)

    # For each corner c = 3*f + i, record its face and local index.
    face_of_corner = np.repeat(np.arange(num_faces, dtype=np.int64), 3)
    local_of_corner = np.tile(np.arange(3, dtype=np.int64), num_faces)

    # Sort corners by edge so that corners sharing an edge are adjacent.
    order = np.argsort(corner_to_edge, kind="stable")
    edges_sorted = corner_to_edge[order]
    faces_sorted = face_of_corner[order]
    local_sorted = local_of_corner[order]

    # Consecutive pairs with the same edge ID are manifold interior edges.
    pair_mask = edges_sorted[:-1] == edges_sorted[1:]
    s = np.where(pair_mask)[0]  # index of first corner in each pair

    f1 = faces_sorted[s]
    l1 = local_sorted[s]
    f2 = faces_sorted[s + 1]
    l2 = local_sorted[s + 1]

    adj_face = -np.ones((num_faces, 3), dtype=np.int64)
    adj_edge = -np.ones((num_faces, 3), dtype=np.int64)

    adj_face[f1, l1] = f2
    adj_edge[f1, l1] = l2
    adj_face[f2, l2] = f1
    adj_edge[f2, l2] = l1

    return adj_face, adj_edge
