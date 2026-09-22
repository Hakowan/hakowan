"""Fur/hair strands flowing along a surface vector field.

Each strand is a short tapered curve that flows in the direction of the
(tangent-projected) vector field.  A dense collection of such strands reads as
fur combed along the field.  Two strand shapes are available:

* **Off-surface** (default): an analytic strand that leans along the flow,
  lifts off the surface by ``lift`` and curls back toward the flow.  Fast and
  robust; strands stand off the surface like real fur.

* **Surface-following** (``follow_surface=True``): a short streamline segment
  traced *on* the surface (so it follows both the field and the surface
  curvature), then given a gentle normal-direction lift and curl.  The strand
  hugs the surface and generally does not stick out — think fur combed flat
  along the flow.

The per-facet field direction, per-face tangent frames, edge adjacency and the
edge-crossing tracer are reused from the streamline machinery.
"""

from __future__ import annotations

import lagrange
import numpy as np

from .streamline import (
    _build_face_adjacency,
    _build_face_frames,
    _project_to_2d,
    _resolve_facet_vector_field,
    _rotation_matrix,
)

STRAND_ID_ATTR = "_hakowan_strand_id"
STRAND_RADIUS_ATTR = "_hakowan_strand_radius"

# Child/clump settings baked onto the output mesh (read by the Blender backend
# to build a Geometry Nodes child-hair modifier). Constant across all vertices.
FUR_CHILDREN_ATTR = "_hakowan_fur_children"
FUR_CLUMP_ATTR = "_hakowan_fur_clump"
FUR_SPREAD_ATTR = "_hakowan_fur_spread"
# Per-strand seed surface normal (constant along each strand); the child-hair
# modifier uses it to scatter child roots tangentially (along the surface).
STRAND_NORMAL_ATTR = "_hakowan_strand_normal"


def _compute_fur(
    mesh: lagrange.SurfaceMesh,
    vec_field_attr: str,
    *,
    n: int = 2000,
    length: float | None = None,
    lift: float = 30.0,
    curl: float = 0.35,
    segments: int = 6,
    root_radius: float | None = None,
    tip_radius: float = 0.0,
    randomness: float = 0.3,
    follow_surface: bool = False,
    children: int = 0,
    clump: float = 0.6,
    spread: float | None = None,
    seed: int = 0,
) -> lagrange.SurfaceMesh:
    """Grow fur strands from a per-facet vector/cross field.

    Args:
        mesh: Triangulated surface mesh.
        vec_field_attr: Name of the per-facet 3D vector attribute.  Vertex- or
            corner-domain attributes are averaged to per-facet first.
        n: Number of strands (area-uniform surface seed points).
        length: Strand length in object space.  ``None`` -> 5% of bbox diagonal.
        lift: Off-surface mode: root lift-off angle in degrees (0 flat, 90
            straight up).  Surface-following mode: tip rise angle (the tip lifts
            by ``length * sin(lift)`` along the surface normal).
        curl: Strength of the curl (toward the flow off-surface, or an extra
            gentle tip rise when surface-following).
        segments: Segments per strand (``segments + 1`` points).
        root_radius: Radius at the root.  ``None`` -> 6% of ``length``.
        tip_radius: Radius at the tip.
        randomness: Per-strand variation in ``[0, 1]``.
        follow_surface: Trace each strand as a short streamline on the surface
            (with a gentle lift/curl) so it hugs the surface instead of
            standing off it.  Default False.
        children: Child hairs grown per guide strand (Blender backend only).
        clump: Child-tip convergence strength in ``[0, 1]``.
        spread: Child-root scatter radius.  ``None`` -> 15% of ``length``.
        seed: RNG seed.

    Returns:
        A :class:`lagrange.SurfaceMesh` whose vertices are strand sample points
        and whose 2-vertex polygons encode the segments connecting them.  Two
        per-vertex attributes are attached: ``_hakowan_strand_id`` (``int32``)
        and ``_hakowan_strand_radius`` (``float64``).  Points belonging to the
        same strand are stored consecutively, root first.
    """
    if mesh.vertex_per_facet != 3:
        raise ValueError("Fur transform requires a triangle mesh.")
    if not mesh.has_attribute(vec_field_attr):
        raise ValueError(f"Mesh has no attribute '{vec_field_attr}'.")
    if n <= 0 or mesh.num_facets == 0 or segments < 1:
        return lagrange.SurfaceMesh()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    facets = np.asarray(mesh.facets, dtype=np.int64).reshape(-1, 3)
    num_faces = facets.shape[0]

    e1, e2, normals = _build_face_frames(vertices, facets)
    vec_field_3d = _resolve_facet_vector_field(
        mesh, vec_field_attr, num_faces, facets, cross_field=False
    )

    # Project the field onto each face tangent plane and normalize -> flow dir.
    tangent = (
        vec_field_3d - normals * np.einsum("fi,fi->f", vec_field_3d, normals)[:, None]
    )
    tnorm = np.linalg.norm(tangent, axis=1)
    valid_face = tnorm > 1e-12
    tangent[valid_face] /= tnorm[valid_face, None]
    tangent[~valid_face] = e1[~valid_face]  # arbitrary in-plane fallback

    # Resolve default lengths relative to the mesh extent.
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if length is None:
        length = 0.05 * diag if diag > 0 else 1.0
    if root_radius is None:
        root_radius = 0.06 * length

    # Area-uniform surface sampling: pick faces weighted by area, then a random
    # barycentric point within each.
    p0, p1, p2 = vertices[facets[:, 0]], vertices[facets[:, 1]], vertices[facets[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    total_area = float(areas.sum())
    rng = np.random.default_rng(seed)
    if total_area <= 0:
        face_ids = rng.integers(0, num_faces, size=n)
    else:
        face_ids = rng.choice(num_faces, size=n, p=areas / total_area)

    u = rng.random(n)
    v = rng.random(n)
    su = np.sqrt(u)
    b0 = (1.0 - su)[:, None]
    b1 = (su * (1.0 - v))[:, None]
    b2 = (su * v)[:, None]
    seed_pts = b0 * p0[face_ids] + b1 * p1[face_ids] + b2 * p2[face_ids]

    fn = normals[face_ids]  # (n, 3)
    ft = tangent[face_ids]  # (n, 3)
    fb = np.cross(fn, ft)  # in-plane, perpendicular to flow

    # Per-strand random variation. Lift/curl vary *multiplicatively* so that
    # ``lift=0`` (or ``curl=0``) stays exactly zero — randomness then perturbs
    # length and in-plane direction without pushing strands off the surface.
    r = float(np.clip(randomness, 0.0, 1.0))
    len_k = length * (1.0 + r * (rng.random(n) - 0.5) * 0.8)  # +/-40% * r
    lift_k = np.radians(
        np.maximum(0.0, lift * (1.0 + r * rng.standard_normal(n) * 0.5))
    )
    curl_k = curl * (1.0 + r * (rng.random(n) - 0.5))
    azim_k = r * rng.standard_normal(n) * np.radians(18.0)

    npts = segments + 1
    frac = np.linspace(0.0, 1.0, npts)  # (npts,)

    if follow_surface:
        pos = _build_surface_strands(
            mesh,
            vertices,
            facets,
            num_faces,
            normals,
            e1,
            e2,
            tangent,
            face_ids,
            seed_pts,
            ft,
            fb,
            len_k,
            lift_k,
            curl_k,
            azim_k,
            frac,
            segments,
        )
    else:
        # Rotate the flow direction within the tangent plane for a natural spread.
        ft_j = np.cos(azim_k)[:, None] * ft + np.sin(azim_k)[:, None] * fb

        # Initial growth direction: lean along the flow, lifted off the surface.
        grow = np.cos(lift_k)[:, None] * ft_j + np.sin(lift_k)[:, None] * fn
        grow /= np.maximum(np.linalg.norm(grow, axis=1, keepdims=True), 1e-20)

        # Curl bends the strand from ``grow`` toward the pure in-plane flow.
        flow_perp = ft_j - np.einsum("ni,ni->n", ft_j, grow)[:, None] * grow

        # positions: (n, npts, 3)
        pos = (
            seed_pts[:, None, :]
            + len_k[:, None, None] * frac[None, :, None] * grow[:, None, :]
            + (len_k * curl_k)[:, None, None]
            * (frac[None, :, None] ** 2)
            * flow_perp[:, None, :]
        )

    radius = root_radius * (1.0 - frac) + tip_radius * frac  # (npts,)

    out_mesh = lagrange.SurfaceMesh()
    all_pts = pos.reshape(-1, 3)
    out_mesh.add_vertices(all_pts)

    # 2-vertex segment polygons + per-vertex strand id / radius.
    base = np.arange(n)[:, None] * npts  # (n, 1)
    starts = base + np.arange(npts - 1)[None, :]  # (n, npts-1)
    seg = np.stack([starts, starts + 1], axis=-1).reshape(-1, 2)
    out_mesh.add_polygons(seg.astype(np.uint32))

    ids = np.repeat(np.arange(n, dtype=np.int32), npts)
    radii = np.tile(radius, n).astype(np.float64)

    out_mesh.create_attribute(
        STRAND_ID_ATTR,
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=ids,
    )
    out_mesh.create_attribute(
        STRAND_RADIUS_ATTR,
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=radii,
    )

    # Carry the source mesh's data attributes onto the strands (sampled at each
    # strand's seed, constant along the strand) so fur can be colored by an
    # attribute — the color pipeline runs after this transform, on the strand
    # mesh, and needs the attribute present there.
    _transfer_source_attributes(mesh, out_mesh, face_ids, (b0, b1, b2), npts)

    # Bake child/clump settings (constant across vertices) for the Blender
    # backend's Geometry Nodes child-hair modifier. Only when children are asked
    # for, so guide-only fur carries no extra attributes.
    if children > 0:
        num_v = out_mesh.num_vertices
        if spread is None:
            spread = 0.15 * length
        # Seed surface normal repeated along each strand — lets the child-hair
        # modifier scatter roots tangentially so children stay on the surface.
        out_mesh.create_attribute(
            STRAND_NORMAL_ATTR,
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Vector,
            initial_values=np.repeat(fn, npts, axis=0).astype(np.float64),
        )
        out_mesh.create_attribute(
            FUR_CHILDREN_ATTR,
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=np.full(num_v, int(children), dtype=np.int32),
        )
        out_mesh.create_attribute(
            FUR_CLUMP_ATTR,
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=np.full(num_v, float(clump), dtype=np.float64),
        )
        out_mesh.create_attribute(
            FUR_SPREAD_ATTR,
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=np.full(num_v, float(spread), dtype=np.float64),
        )

    return out_mesh


# ---------------------------------------------------------------------------
# Attribute transfer
# ---------------------------------------------------------------------------


def _transfer_source_attributes(src_mesh, out_mesh, face_ids, bary, npts):
    """Copy the source mesh's data attributes onto the strand mesh.

    Each attribute is sampled once per strand at the seed (facet value, or
    barycentric-interpolated vertex value) and repeated along the strand's
    ``npts`` points, so any of them can drive fur color via a colormap.  Only
    plain per-vertex / per-facet numeric attributes with Scalar / Vector /
    Color usage are carried; reserved (``@*``), internal (``_hakowan*``),
    indexed and other-domain attributes are skipped.
    """
    facets = np.asarray(src_mesh.facets, dtype=np.int64).reshape(-1, 3)
    b0, b1, b2 = bary
    carry_usages = {
        lagrange.AttributeUsage.Scalar,
        lagrange.AttributeUsage.Vector,
        lagrange.AttributeUsage.Color,
    }

    for aid in src_mesh.get_matching_attribute_ids():
        name = src_mesh.get_attribute_name(aid)
        if name.startswith("@") or name.startswith("_hakowan"):
            continue
        if src_mesh.is_attribute_indexed(name):
            continue
        attr = src_mesh.attribute(name)
        if attr.usage not in carry_usages:
            continue

        data = np.asarray(attr.data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if attr.element_type == lagrange.AttributeElement.Facet:
            per_strand = data[face_ids]
        elif attr.element_type == lagrange.AttributeElement.Vertex:
            tri = facets[face_ids]
            per_strand = (
                b0 * data[tri[:, 0]] + b1 * data[tri[:, 1]] + b2 * data[tri[:, 2]]
            )
        else:
            continue

        vals = np.repeat(per_strand, npts, axis=0)
        if vals.shape[1] == 1:
            vals = vals.reshape(-1)
        out_mesh.create_attribute(
            name,
            element=lagrange.AttributeElement.Vertex,
            usage=attr.usage,
            initial_values=vals,
        )


# ---------------------------------------------------------------------------
# Surface-following strands
# ---------------------------------------------------------------------------


def _build_surface_strands(
    mesh,
    vertices,
    facets,
    num_faces,
    normals,
    e1,
    e2,
    tangent,
    face_ids,
    seed_pts,
    ft,
    fb,
    len_k,
    lift_k,
    curl_k,
    azim_k,
    frac,
    segments,
):
    """Trace short on-surface streamlines and lift/curl them into fur strands.

    Returns an ``(n, npts, 3)`` array of strand sample points that hug the
    surface: each strand follows the field along the surface for ``len_k`` of
    arc length, then rises gently along the interpolated surface normal by
    ``L * (sin(lift) * frac + curl * frac**2)`` (``L`` the strand's traced
    length), so it generally does not stick out of the surface.
    """
    mesh.initialize_edges()
    adj_face, adj_edge = _build_face_adjacency(mesh, num_faces)
    # Normalized field direction per face in the local 2D frame.
    vec_2d = _project_to_2d(tangent.copy(), e1, e2)

    n = len(face_ids)
    npts = segments + 1
    pos = np.empty((n, npts, 3), dtype=np.float64)
    max_steps = max(4 * segments, int(num_faces), 64)

    for k in range(n):
        fi = int(face_ids[k])
        # Initial direction: the seed-face flow, rotated in-plane by azim_k.
        d0 = np.cos(azim_k[k]) * ft[k] + np.sin(azim_k[k]) * fb[k]

        pts, nrms = _trace_surface_strand(
            fi,
            seed_pts[k],
            d0,
            float(len_k[k]),
            max_steps,
            vertices,
            facets,
            normals,
            e1,
            e2,
            vec_2d,
            adj_face,
            adj_edge,
        )

        if len(pts) >= 2:
            surf, nrm, arc_len = _resample_polyline(pts, nrms, npts)
        else:
            surf = nrm = None
            arc_len = 0.0

        if surf is None or arc_len < 1e-9:
            # Degenerate trace (e.g. immediate boundary): fall back to a short
            # straight strand along the seed flow direction, on the tangent plane.
            surf = seed_pts[k][None, :] + (frac[:, None] * len_k[k]) * d0[None, :]
            nrm = np.repeat(normals[fi][None, :], npts, axis=0)
            arc_len = float(len_k[k])

        lift_off = arc_len * (np.sin(lift_k[k]) * frac + curl_k[k] * frac**2)
        pos[k] = surf + nrm * lift_off[:, None]

    return pos


def _trace_surface_strand(
    f0,
    p0,
    d0,
    target_length,
    max_steps,
    vertices,
    facets,
    normals,
    e1,
    e2,
    vec_2d,
    adj_face,
    adj_edge,
):
    """Trace one short streamline of the field on the surface.

    Mirrors the streamline tracer's exact edge-crossing walk, but re-reads the
    field direction on each new face (oriented consistently with the incoming
    direction) so the path truly follows the flow.  Returns ``(points,
    normals)`` lists including the seed point (face normals of the faces the
    path passes through), stopping once ``target_length`` of arc length is
    reached or the mesh boundary is hit.
    """
    pts = [p0.copy()]
    nrms = [normals[f0].copy()]
    f = int(f0)
    p = p0.copy()
    d = d0.copy()
    d_norm = float(np.linalg.norm(d))
    if d_norm < 1e-12:
        return pts, nrms
    d = d / d_norm
    accumulated = 0.0
    entry_local = -1

    for _ in range(max_steps):
        n = normals[f]
        perp = np.cross(n, d)
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm < 1e-12:
            break
        perp = perp / perp_norm

        vi = facets[f]
        vals = np.array([float(np.dot(vertices[vi[k]] - p, perp)) for k in range(3)])
        for k in range(3):
            if abs(vals[k]) < 1e-10:
                vals[k] = 1e-10

        exit_local = -1
        exit_pt = None
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
                    continue
                if fwd < best_fwd:
                    best_fwd = fwd
                    exit_local = k
                    exit_pt = cand

        if exit_local < 0 or exit_pt is None:
            break

        step_len = float(np.linalg.norm(exit_pt - p))
        if accumulated + step_len >= target_length:
            # Truncate the final step so the strand length is exactly the target.
            remain = target_length - accumulated
            if step_len > 1e-12 and remain > 0.0:
                exit_pt = p + (exit_pt - p) * (remain / step_len)
            pts.append(exit_pt)
            nrms.append(normals[f].copy())
            break

        accumulated += step_len
        pts.append(exit_pt)
        nrms.append(normals[f].copy())
        p = exit_pt

        fp = int(adj_face[f, exit_local])
        if fp < 0:
            break
        entry_local = int(adj_edge[f, exit_local])

        # Follow the field: re-read the flow direction on the new face, oriented
        # to agree with the parallel-transported incoming direction.
        R = _rotation_matrix(normals[f], normals[fp])
        d_transported = R @ d
        field = vec_2d[fp, 0] * e1[fp] + vec_2d[fp, 1] * e2[fp]
        if float(np.dot(field, d_transported)) < 0.0:
            field = -field
        d_norm = float(np.linalg.norm(field))
        if d_norm < 1e-12:
            break
        d = field / d_norm
        f = fp

    return pts, nrms


def _resample_polyline(pts, nrms, npts):
    """Resample a polyline (and its per-point normals) to ``npts`` points evenly
    spaced by arc length.  Returns ``(positions, unit_normals, total_length)``.
    """
    P = np.asarray(pts, dtype=np.float64)
    N = np.asarray(nrms, dtype=np.float64)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-12:
        return None, None, 0.0

    targets = np.linspace(0.0, total, npts)
    surf = np.empty((npts, 3), dtype=np.float64)
    nrm = np.empty((npts, 3), dtype=np.float64)
    for ax in range(3):
        surf[:, ax] = np.interp(targets, s, P[:, ax])
        nrm[:, ax] = np.interp(targets, s, N[:, ax])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)
    return surf, nrm, total
