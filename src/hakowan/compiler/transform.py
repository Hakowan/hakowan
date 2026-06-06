from .view import View
from ..grammar.mark import Mark
from ..grammar.scale import Attribute
from ..grammar.transform import (
    Affine,
    Boundary,
    Clip,
    Compute,
    Explode,
    Filter,
    Norm,
    PrincipalAxes,
    Streamline,
    Transform,
    UVMesh,
)
from .streamline import _compute_streamlines
from ..common import logger

import copy
import lagrange
import numpy as np


def principal_axes_affine_matrix(
    vertices: np.ndarray,
    frame: np.ndarray,
    *,
    orthonormalize_frame: bool = True,
) -> np.ndarray:
    """4x4 affine: centroid at origin; PCA axes (major first) aligned to columns of ``frame``."""
    if vertices.size == 0:
        return np.eye(4, dtype=np.float64)
    f = np.asarray(frame, dtype=np.float64).reshape(3, 3)
    if orthonormalize_frame:
        f, _ = np.linalg.qr(f)
        if np.linalg.det(f) < 0.0:
            f[:, -1] *= -1.0
    mu = np.mean(vertices, axis=0)
    x = vertices - mu
    n = x.shape[0]
    if n < 2:
        m = np.eye(4, dtype=np.float64)
        m[:3, 3] = -mu
        return m
    cov = (x.T @ x) / max(n - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order]
    if np.linalg.det(basis) < 0.0:
        basis[:, 2] *= -1.0
    r = f @ basis.T
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r
    m[:3, 3] = -r @ mu
    return m


def _apply_filter_transform(view: View, transform: Filter):
    """Filter the data based on attribute value and condition specified in the transform."""
    df = view.data_frame
    assert df is not None
    assert transform is not None
    mesh = df.mesh
    if mesh.num_vertices == 0:
        return

    # Compute and store original bbox
    assert view.bbox is not None

    if transform.data is None:
        transform.data = Attribute(name=mesh.attr_name_vertex_to_position)
    if isinstance(transform.data, str):
        transform.data = Attribute(name=transform.data)
    assert isinstance(transform.data, Attribute)

    attr_name = transform.data.name
    if transform.data.scale is not None:
        logger.warning("Attribute scale is ignored when applying transform.")
    assert mesh.has_attribute(attr_name), (
        f"Attribute {attr_name} does not exist in data"
    )
    attr = mesh.attribute(attr_name)
    keep = [transform.condition(value) for value in attr.data]

    match attr.element_type:
        case lagrange.AttributeElement.Facet:
            selected_facets = np.arange(mesh.num_facets, dtype=np.uint32)[keep]
            df.mesh = lagrange.extract_submesh(
                mesh,
                selected_facets=selected_facets,
                map_attributes=True,
            )
        case lagrange.AttributeElement.Vertex:
            if view.mark == Mark.Point:
                vertices_to_remove = np.arange(mesh.num_vertices, dtype=np.uint32)[
                    np.logical_not(keep)
                ]
                df.mesh.remove_vertices(vertices_to_remove)
            elif view.mark == Mark.Surface:
                selected_vertices = np.zeros(mesh.num_vertices, dtype=np.uint32)
                selected_vertices[keep] = 1
                selected_facets = np.all(selected_vertices[mesh.facets], axis=1)
                selected_facets = np.arange(mesh.num_facets, dtype=np.uint32)[
                    selected_facets
                ]
                df.mesh = lagrange.extract_submesh(
                    mesh,
                    selected_facets=selected_facets,
                    map_attributes=True,
                )
            else:
                # TODO: Add curve support.
                raise NotImplementedError(
                    "Filter transform does not support curve mark yet."
                )
        case _:
            raise RuntimeError(f"Unsupported element type: {attr.element_type}!")


def _interp_corner_values(corner_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Barycentrically combine per-corner attribute values.

    ``corner_values`` is ``(Nc, 3)`` for scalar attributes or ``(Nc, 3, C)`` for
    multi-channel ones, holding the three parent-triangle corner values for each
    output corner; ``weights`` is the matching ``(Nc, 3)`` barycentric weights.

    Floating-point attributes are interpolated; integer attributes (e.g. ids)
    are not meaningfully averaged, so the value of the dominant corner is taken.
    """
    cv = np.asarray(corner_values)
    if cv.shape[0] == 0:
        return np.ascontiguousarray(cv.reshape((0,) + cv.shape[2:]))
    if np.issubdtype(cv.dtype, np.integer) or cv.dtype == bool:
        pick = np.argmax(weights, axis=1)
        return np.ascontiguousarray(cv[np.arange(cv.shape[0]), pick])
    if cv.ndim == 2:
        out = np.einsum("nk,nk->n", weights, cv)
    else:
        out = np.einsum("nk,nkc->nc", weights, cv)
    return np.ascontiguousarray(out)


def _clip_mesh(
    mesh: lagrange.SurfaceMesh,
    point: np.ndarray,
    normal: np.ndarray,
) -> lagrange.SurfaceMesh:
    """Clip ``mesh`` against the plane through ``point`` with the given ``normal``.

    The half-space where ``dot(normal, x - point) >= 0`` is kept. Triangles
    straddling the plane are cut via Sutherland-Hodgman clipping (partial
    triangles are produced); the exposed cross-section is left open.

    The result is a triangle soup (one independent vertex per output corner) so
    that vertex, corner, and indexed attributes are all interpolated correctly at
    the cut. Per-facet attributes are copied to the child triangles.
    """
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n_len = np.linalg.norm(n)
    if n_len == 0.0:
        raise ValueError("Clip plane normal must be non-zero.")
    n = n / n_len
    p = np.asarray(point, dtype=np.float64).reshape(3)

    # Work on a triangulated copy so the caller's mesh is untouched and every
    # facet has exactly three corners.
    mesh = copy.deepcopy(mesh)
    if mesh.has_edges:
        mesh.clear_edges()
    if not mesh.is_triangle_mesh:
        lagrange.triangulate_polygonal_facets(mesh)

    vertices = mesh.vertices
    facets = mesh.facets  # (Nf, 3)

    # Signed distance to the plane; keep where sd >= 0.
    sd = (vertices - p) @ n
    inside = sd >= 0.0

    eye3 = np.eye(3, dtype=np.float64)
    sd_tri = sd[facets]  # (Nf, 3)
    inside_count = inside[facets].sum(axis=1)  # (Nf,)

    # Each output corner records its parent facet and barycentric weights over
    # the parent triangle's three corners. Output triangles are consecutive
    # triples of corners.
    out_facet: list[int] = []
    out_weights: list[np.ndarray] = []

    # Whole triangles fully inside: copy as-is (identity barycentric per corner).
    full_in = np.flatnonzero(inside_count == 3)
    if full_in.size:
        out_facet.extend(np.repeat(full_in, 3).tolist())
        out_weights.extend(np.tile(eye3, (full_in.size, 1)))

    # Straddling triangles: Sutherland-Hodgman clip against the half-space, then
    # fan-triangulate the resulting convex polygon.
    for f in np.flatnonzero((inside_count == 1) | (inside_count == 2)):
        d = sd_tri[f]
        poly: list[np.ndarray] = []
        for k in range(3):
            j = (k + 1) % 3
            cur_in = d[k] >= 0.0
            nxt_in = d[j] >= 0.0
            if nxt_in:
                if not cur_in:
                    t = d[k] / (d[k] - d[j])
                    poly.append((1.0 - t) * eye3[k] + t * eye3[j])
                poly.append(eye3[j])
            elif cur_in:
                t = d[k] / (d[k] - d[j])
                poly.append((1.0 - t) * eye3[k] + t * eye3[j])
        if len(poly) < 3:
            continue
        for i in range(1, len(poly) - 1):
            out_facet.extend((int(f), int(f), int(f)))
            out_weights.extend((poly[0], poly[i], poly[i + 1]))

    out_facet_arr = np.asarray(out_facet, dtype=np.int64)
    weights = np.asarray(out_weights, dtype=np.float64).reshape(-1, 3)
    num_corners = out_facet_arr.shape[0]

    out = lagrange.SurfaceMesh(mesh.dimension)
    if num_corners == 0:
        logger.warning("Clip transform removed the entire mesh.")
        return out

    corner_vtx = facets[out_facet_arr]  # (Nc, 3) parent vertex indices
    corner_idx = (3 * out_facet_arr)[:, None] + np.arange(
        3
    )  # (Nc, 3) parent corner ids
    parent_facet_per_tri = out_facet_arr[::3]  # (Nt,)

    out_positions = np.einsum("nk,nkc->nc", weights, vertices[corner_vtx])
    out.add_vertices(np.ascontiguousarray(out_positions))
    out.add_triangles(np.arange(num_corners, dtype=np.uint32).reshape(-1, 3))

    for attr_id in mesh.get_matching_attribute_ids():
        attr_name = mesh.get_attribute_name(attr_id)
        if mesh.attr_name_is_reserved(attr_name):
            continue  # positions (rebuilt above) and other internal attributes.

        if mesh.is_attribute_indexed(attr_id):
            attr = mesh.indexed_attribute(attr_id)
            corner_vals = attr.values.data[attr.indices.data[corner_idx]]
            out.create_attribute(
                attr_name,
                element=lagrange.AttributeElement.Vertex,
                usage=attr.usage,
                initial_values=_interp_corner_values(corner_vals, weights),
            )
            continue

        attr = mesh.attribute(attr_id)
        match attr.element_type:
            case lagrange.AttributeElement.Vertex:
                out.create_attribute(
                    attr_name,
                    element=lagrange.AttributeElement.Vertex,
                    usage=attr.usage,
                    initial_values=_interp_corner_values(
                        attr.data[corner_vtx], weights
                    ),
                )
            case lagrange.AttributeElement.Corner:
                out.create_attribute(
                    attr_name,
                    element=lagrange.AttributeElement.Vertex,
                    usage=attr.usage,
                    initial_values=_interp_corner_values(
                        attr.data[corner_idx], weights
                    ),
                )
            case lagrange.AttributeElement.Facet:
                out.create_attribute(
                    attr_name,
                    element=lagrange.AttributeElement.Facet,
                    usage=attr.usage,
                    initial_values=np.ascontiguousarray(
                        attr.data[parent_facet_per_tri]
                    ),
                )
            case _:
                logger.warning(
                    f"Clip transform: skipping attribute '{attr_name}' with "
                    f"unsupported element type {attr.element_type}."
                )

    return out


def _apply_clip_transform(view: View, transform: Clip):
    """Clip the data frame against the plane specified in the transform."""
    df = view.data_frame
    assert df is not None
    assert transform is not None
    mesh = df.mesh
    if mesh.num_vertices == 0:
        return
    point = np.asarray(transform.point, dtype=np.float64).reshape(3)
    normal = np.asarray(transform.normal, dtype=np.float64).reshape(3)
    df.mesh = _clip_mesh(mesh, point, normal)


def _apply_uv_mesh_transform(view: View, transform: UVMesh):
    """Extract the UV mesh from the original mesh."""
    df = view.data_frame
    assert df is not None
    assert transform is not None
    mesh = df.mesh
    if mesh.num_vertices == 0:
        return

    if isinstance(transform.uv, str):
        transform.uv = Attribute(name=transform.uv)

    if transform.uv is not None:
        uv_attr_name = transform.uv.name
        if transform.uv.scale is not None:
            logger.warning("Attribute scale is ignored when applying transform.")
    else:
        uv_ids = mesh.get_matching_attribute_ids(usage=lagrange.AttributeUsage.UV)
        assert len(uv_ids) > 0
        uv_attr_name = mesh.get_attribute_name(uv_ids[0])
        logger.info(f"Automatically detected UV attribute: {uv_attr_name}")
    assert mesh.has_attribute(uv_attr_name)
    if mesh.is_attribute_indexed(uv_attr_name):
        uv_attr = mesh.indexed_attribute(uv_attr_name)
        uv_values = uv_attr.values.data.copy()
        uv_values = np.hstack((uv_values, np.zeros((uv_values.shape[0], 1))))
        uv_indices = uv_attr.indices.data.copy()
        if mesh.is_regular:
            uv_indices = uv_indices.reshape(-1, mesh.vertex_per_facet)
        else:
            raise RuntimeError(
                "Hybrid mesh (e.g. mixture of triangles, quads and/or polygons) is not supported yet."
            )

        uv_mesh = lagrange.SurfaceMesh()
        uv_mesh.add_vertices(uv_values)
        uv_mesh.add_polygons(uv_indices)

        # Map attributes.
        for attr_id in mesh.get_matching_attribute_ids():
            attr_name = mesh.get_attribute_name(attr_id)
            if mesh.attr_name_is_reserved(attr_name):  # type: ignore
                continue

            if mesh.is_attribute_indexed(attr_id):  # type: ignore
                attr = mesh.indexed_attribute(attr_id)  # type: ignore
            else:
                attr = mesh.attribute(attr_id)  # type: ignore

            if attr.element_type != lagrange.AttributeElement.Vertex:
                uv_mesh.create_attribute_from(attr_name, mesh)
            else:
                lagrange.map_attribute_in_place(
                    mesh, attr_name, lagrange.AttributeElement.Corner
                )
                uv_mesh.create_attribute_from(attr_name, mesh)
                lagrange.map_attribute_in_place(
                    uv_mesh, attr_name, lagrange.AttributeElement.Vertex
                )
    else:
        uv_attr = mesh.attribute(uv_attr_name)
        assert uv_attr.element_type == lagrange.AttributeElement.Vertex
        uv_values = np.hstack((uv_attr.data, np.zeros((mesh.num_vertices, 1))))
        uv_mesh = copy.deepcopy(mesh)
        uv_mesh.vertices = uv_values
    df.mesh = uv_mesh

    # Because UV mesh is of very different scale than the original mesh. BBox must be updated.
    logger.debug("Updating view bbox due to UV mesh transform.")
    view.initialize_bbox()


def _apply_affine_transform(view: View, transform: Affine):
    df = view.data_frame
    assert df is not None
    assert transform is not None

    if np.shape(transform.matrix) == (4, 4):
        matrix = np.array(transform.matrix, order="F", dtype=np.float64)
    elif np.shape(transform.matrix) == (3, 3):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.array(transform.matrix, order="F", dtype=np.float64)
    else:
        raise RuntimeError(
            f"Invalid affine transformation matrix with shape {np.shape(transform.matrix)}."
        )
    view.global_transform = matrix @ view.global_transform

    # BBox must be updated after affine transform.
    logger.debug("Updating view bbox due to affine transform.")
    view.initialize_bbox()


def _apply_principal_axes_transform(view: View, transform: PrincipalAxes):
    df = view.data_frame
    assert df is not None
    assert transform is not None
    mesh = df.mesh
    if mesh.num_vertices == 0:
        return
    matrix = principal_axes_affine_matrix(
        mesh.vertices,
        np.asarray(transform.frame, dtype=np.float64),
        orthonormalize_frame=transform.orthonormalize_frame,
    )
    view.global_transform = matrix @ view.global_transform
    logger.debug("Updating view bbox due to principal-axes transform.")
    view.initialize_bbox()


def _apply_compute_transform(view: View, transform: Compute):
    df = view.data_frame
    assert df is not None
    assert transform is not None
    mesh = df.mesh

    if transform.x is not None:
        mesh.create_attribute(
            transform.x,
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=mesh.vertices[:, 0].copy(),
        )
    if transform.y is not None:
        mesh.create_attribute(
            transform.y,
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=mesh.vertices[:, 1].copy(),
        )
    if transform.z is not None:
        mesh.create_attribute(
            transform.z,
            element=lagrange.AttributeElement.Vertex,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=mesh.vertices[:, 2].copy(),
        )
    if transform.normal is not None:
        lagrange.compute_normal(mesh, output_attribute_name=transform.normal)
    if transform.vertex_normal is not None:
        lagrange.compute_vertex_normal(
            mesh, output_attribute_name=transform.vertex_normal
        )
    if transform.facet_normal is not None:
        lagrange.compute_facet_normal(
            mesh, output_attribute_name=transform.facet_normal
        )
    if transform.component is not None:
        lagrange.compute_components(mesh, output_attribute_name=transform.component)


def _apply_explode_transform(view: View, transform: Explode):
    df = view.data_frame
    assert df is not None
    assert transform is not None
    mesh = df.mesh
    assert mesh.has_attribute(transform.pieces)  # type: ignore
    if isinstance(transform.pieces, str):
        attr_name = transform.pieces
    elif isinstance(transform.pieces, Attribute):
        attr_name = transform.pieces.name
    pieces_attr = mesh.attribute(attr_name)
    assert pieces_attr.element_type == lagrange.AttributeElement.Facet
    piece_index = pieces_attr.data
    # Remove edge attribute to avoid warnings.
    if mesh.has_edges:
        mesh.clear_edges()
    pieces = lagrange.separate_by_facet_groups(mesh, piece_index, map_attributes=True)

    center = np.average(mesh.vertices, axis=0)
    offset_dirs = np.array([np.average(p.vertices, axis=0) - center for p in pieces])
    offset_dirs *= transform.magnitude

    for i in range(len(pieces)):
        pieces[i].vertices = pieces[i].vertices + offset_dirs[i]

    df.mesh = lagrange.combine_meshes(pieces)


def _apply_norm_transform(view: View, transform: Norm):
    df = view.data_frame
    assert df is not None
    assert transform is not None
    mesh = df.mesh
    if isinstance(transform.data, str):
        input_attr_name = transform.data
    elif isinstance(transform.data, Attribute):
        input_attr_name = transform.data.name
    else:
        raise RuntimeError("Invalid input vector data.")
    assert mesh.has_attribute(input_attr_name)  # type: ignore
    if mesh.is_attribute_indexed(input_attr_name):
        input_attr = mesh.indexed_attribute(input_attr_name)
        assert input_attr.num_channels > 1
        norm_data = np.linalg.norm(input_attr.values.data, axis=1, ord=transform.order)
        mesh.create_attribute(
            transform.norm_attr_name,
            element=input_attr.element_type,
            usage=input_attr.usage,
            initial_values=norm_data,
            initial_indices=input_attr.indices.data.copy(),
        )
    else:
        input_attr = mesh.attribute(input_attr_name)
        assert input_attr.num_channels > 1
        norm_data = np.linalg.norm(input_attr.data, axis=1, ord=transform.order)
        mesh.create_attribute(
            transform.norm_attr_name,
            element=input_attr.element_type,
            usage=input_attr.usage,
            initial_values=norm_data,
        )


def _apply_boundary_transform(view: View, transform: Boundary):
    df = view.data_frame
    assert df is not None
    assert transform is not None
    mesh = df.mesh

    unified_mesh = lagrange.unify_index_buffer(
        mesh, attribute_names=transform.attributes
    )
    unified_mesh.initialize_edges()

    is_boundary = np.array(
        [unified_mesh.is_boundary_edge(e) for e in range(unified_mesh.num_edges)]
    )
    bd_edge_indices = np.arange(unified_mesh.num_edges)[is_boundary == 1]
    bd_edges = np.array(
        [unified_mesh.get_edge_vertices(ei) for ei in bd_edge_indices], dtype=np.uint32
    )

    bd_mesh = lagrange.SurfaceMesh()
    bd_mesh.add_vertices(unified_mesh.vertices)
    bd_mesh.add_polygons(bd_edges)
    lagrange.remove_isolated_vertices(bd_mesh)

    df.mesh = bd_mesh


def _apply_streamline_transform(view: View, transform: Streamline):
    df = view.data_frame
    assert df is not None
    assert transform is not None

    if isinstance(transform.vec_field, str):
        vec_field_attr = transform.vec_field
    elif isinstance(transform.vec_field, Attribute):
        if transform.vec_field.scale is not None:
            logger.warning("Attribute scale is ignored when applying transform.")
        vec_field_attr = transform.vec_field.name
    else:
        raise RuntimeError("Streamline.vec_field must be a string or Attribute.")

    sl_mesh = _compute_streamlines(
        df.mesh,
        vec_field_attr,
        n=transform.n,
        cross_field=transform.cross_field,
        length=transform.length,
        seed=transform.seed,
        min_length=transform.min_length,
    )

    if transform.id_attr_name != "_hakowan_streamline_id" and sl_mesh.has_attribute(
        "_hakowan_streamline_id"
    ):
        sl_mesh.rename_attribute("_hakowan_streamline_id", transform.id_attr_name)

    df.mesh = sl_mesh

    logger.debug("Updating view bbox due to streamline transform.")
    view.initialize_bbox()


def apply_transform(view: View):
    """Apply a chain of transforms specified by view.transform to view.data_frame.
    Transforms are applied in the order specified by the chain.
    """

    def _apply(t: Transform | None):
        if t is None:
            return
        _apply(t._child)

        match t:
            case Filter():
                assert view.data_frame is not None
                _apply_filter_transform(view, t)
            case Clip():
                assert view.data_frame is not None
                _apply_clip_transform(view, t)
            case UVMesh():
                assert view.data_frame is not None
                _apply_uv_mesh_transform(view, t)
            case Affine():
                assert view.data_frame is not None
                _apply_affine_transform(view, t)
            case PrincipalAxes():
                assert view.data_frame is not None
                _apply_principal_axes_transform(view, t)
            case Compute():
                assert view.data_frame is not None
                _apply_compute_transform(view, t)
            case Explode():
                assert view.data_frame is not None
                _apply_explode_transform(view, t)
            case Norm():
                assert view.data_frame is not None
                _apply_norm_transform(view, t)
            case Boundary():
                assert view.data_frame is not None
                _apply_boundary_transform(view, t)
            case Streamline():
                assert view.data_frame is not None
                _apply_streamline_transform(view, t)
            case _:
                raise NotImplementedError(f"Unsupported transform: {type(t)}!")

    _apply(view.transform)
