from .view import View
from ..grammar.mark import Mark
from ..grammar.dataframe import DataFrame
from ..grammar.scale import Attribute
from ..grammar.transform import (
    Affine,
    Boundary,
    Compute,
    Explode,
    Filter,
    Norm,
    Transform,
    UVMesh,
)
from ..common import logger

import copy
import lagrange
import numpy as np


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
    assert mesh.has_attribute(
        attr_name
    ), f"Attribute {attr_name} does not exist in data"
    attr = mesh.attribute(attr_name)
    keep = [transform.condition(value) for value in attr.data]

    match (attr.element_type):
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
                    f"Filter transform does not support curve mark yet."
                )
        case _:
            raise RuntimeError(f"Unsupported element type: {attr.element_type}!")


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


def apply_transform(view: View):
    """Apply a chain of transforms specified by view.transform to view.data_frame.
    Transforms are applied in the order specified by the chain.
    """

    def _apply(t: Transform | None):
        if t is None:
            return
        _apply(t._child)

        match (t):
            case Filter():
                assert view.data_frame is not None
                _apply_filter_transform(view, t)
            case UVMesh():
                assert view.data_frame is not None
                _apply_uv_mesh_transform(view, t)
            case Affine():
                assert view.data_frame is not None
                _apply_affine_transform(view, t)
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
            case _:
                raise NotImplementedError(f"Unsupported transform: {type(t)}!")

    _apply(view.transform)
