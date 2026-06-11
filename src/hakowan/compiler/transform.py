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


def _flatten_indexed_attribute(out: lagrange.SurfaceMesh, aid: int) -> None:
    """Convert an indexed attribute to a per-vertex attribute in-place.

    For corner-indexed attributes, each vertex gets the average of its corners'
    values (correct for smooth data such as normals; exact when all corners of a
    vertex share the same value).
    """
    ia = out.indexed_attribute(aid)
    name = out.get_attribute_name(aid)
    usage = ia.usage  # save before delete invalidates ia
    corner_values = ia.values.data[ia.indices.data]  # (n_corners, ...) or (n_corners,)
    # out.facets is a flat corner→vertex array for both triangle and polygon meshes.
    corner_to_vtx = out.facets.ravel()
    n_verts = out.num_vertices
    scalar = corner_values.ndim == 1
    if scalar:
        vtx_values = np.zeros(n_verts, dtype=np.float64)
        counts = np.zeros(n_verts, dtype=np.float64)
        np.add.at(vtx_values, corner_to_vtx, corner_values)
        np.add.at(counts, corner_to_vtx, 1.0)
        vtx_values /= np.maximum(counts, 1.0)
    else:
        d = corner_values.shape[1]
        vtx_values = np.zeros((n_verts, d), dtype=np.float64)
        counts = np.zeros(n_verts, dtype=np.float64)
        np.add.at(vtx_values, corner_to_vtx, corner_values)
        np.add.at(counts, corner_to_vtx, 1.0)
        vtx_values /= np.maximum(counts, 1.0)[:, None]
    out.delete_attribute(aid)
    out.create_attribute(
        name,
        element=lagrange.AttributeElement.Vertex,
        usage=usage,
        initial_values=np.ascontiguousarray(vtx_values),
    )


def _clip_mesh(
    mesh: lagrange.SurfaceMesh,
    point: np.ndarray,
    normal: np.ndarray,
) -> lagrange.SurfaceMesh:
    """Clip ``mesh`` against the plane through ``point`` with the given ``normal``.

    The half-space where ``dot(normal, x - point) >= 0`` is kept. Uses
    ``lagrange.trim_by_isoline`` on the signed-distance field, which handles
    attribute interpolation and produces a proper connected mesh.
    """
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n_len = np.linalg.norm(n)
    if n_len == 0.0:
        raise ValueError("Clip plane normal must be non-zero.")
    n = n / n_len
    p = np.asarray(point, dtype=np.float64).reshape(3)

    mesh = copy.deepcopy(mesh)
    if mesh.has_edges:
        mesh.clear_edges()
    if not mesh.is_triangle_mesh:
        lagrange.triangulate_polygonal_facets(mesh)

    if mesh.num_vertices == 0:
        return mesh

    # Save facet attributes: trim_by_isoline resets them to zero, so we
    # recover them afterwards via nearest-input-centroid assignment.
    facet_attrs: dict[str, tuple[lagrange.AttributeUsage, np.ndarray]] = {}
    for faid in mesh.get_matching_attribute_ids():
        fname = mesh.get_attribute_name(faid)
        if mesh.attr_name_is_reserved(fname) or mesh.is_attribute_indexed(faid):
            continue
        fa = mesh.attribute(faid)
        if fa.element_type == lagrange.AttributeElement.Facet:
            facet_attrs[fname] = (fa.usage, fa.data.copy())
    if facet_attrs:
        in_centroid_id = lagrange.compute_facet_centroid(mesh)
        in_centroids = mesh.attribute(in_centroid_id).data.copy()

    _SD_ATTR = "_hakowan_clip_sd"
    sd = ((mesh.vertices - p) @ n).astype(np.float64)
    mesh.create_attribute(
        _SD_ATTR,
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=sd,
    )

    # keep_below=False keeps the side where sd >= isovalue (i.e. sd >= 0).
    out = lagrange.trim_by_isoline(mesh, _SD_ATTR, isovalue=0.0, keep_below=False)

    if out.num_vertices == 0:
        logger.warning("Clip transform removed the entire mesh.")
        return out

    if out.has_attribute(_SD_ATTR):
        out.delete_attribute(_SD_ATTR)

    # Recover facet attributes via nearest input centroid.
    # TODO(perf): the brute-force distance matrix below is O(n_out * n_in) in
    # both time and memory (it materializes an (n_out, n_in, 3) array), so it
    # will OOM on large meshes. Replace with a spatial index (e.g. a KD-tree /
    # lagrange nearest-facet query) or chunk the argmin over output facets.
    if facet_attrs:
        out_centroid_id = lagrange.compute_facet_centroid(out)
        out_centroids = out.attribute(out_centroid_id).data
        diff = out_centroids[:, None, :] - in_centroids[None, :, :]
        parent_idx = np.argmin((diff * diff).sum(axis=2), axis=1)
        for fname, (fusage, fdata) in facet_attrs.items():
            if out.has_attribute(fname):
                out.delete_attribute(fname)
            out.create_attribute(
                fname,
                element=lagrange.AttributeElement.Facet,
                usage=fusage,
                initial_values=np.ascontiguousarray(fdata[parent_idx]),
            )

    # Flatten indexed attributes to per-vertex (matches pre-refactor behaviour
    # and avoids surprises for downstream code that calls mesh.attribute()).
    for iaid in list(out.get_matching_attribute_ids()):
        if not out.is_attribute_indexed(iaid):
            continue
        ianame = out.get_attribute_name(iaid)
        if out.attr_name_is_reserved(ianame):
            continue
        _flatten_indexed_attribute(out, iaid)

    if not out.is_triangle_mesh:
        lagrange.triangulate_polygonal_facets(out)

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
    if isinstance(transform.pieces, str):
        attr_name = transform.pieces
    elif isinstance(transform.pieces, Attribute):
        attr_name = transform.pieces.name
    else:
        raise RuntimeError("Explode.pieces must be a string or Attribute.")
    assert mesh.has_attribute(attr_name)
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
