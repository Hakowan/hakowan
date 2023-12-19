from .view import View
from ..grammar.mark import Mark
from ..grammar.dataframe import DataFrame
from ..grammar.scale import Attribute
from ..grammar.transform import Transform, Filter, UVMesh, Affine, Compute, Explode
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
            assert view.mark == Mark.Point
            vertices_to_remove = np.arange(mesh.num_vertices, dtype=np.uint32)[
                np.logical_not(keep)
            ]
            df.mesh.remove_vertices(vertices_to_remove)
        case _:
            raise RuntimeError(f"Unsupported element type: {attr.element_type}!")


def _appply_uv_mesh_transform(view: View, transform: UVMesh):
    """Extract the UV mesh from the original mesh."""
    df = view.data_frame
    assert df is not None
    assert transform is not None
    mesh = df.mesh
    if mesh.num_vertices == 0:
        return

    if isinstance(transform.uv, str):
        transform.uv = Attribute(name=transform.uv)

    uv_attr_name = transform.uv.name
    if transform.uv.scale is not None:
        logger.warning("Attribute scale is ignored when applying transform.")
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
    mesh = df.mesh

    if np.shape(transform.matrix) == (4, 4):
        matrix = np.array(transform.matrix, order="F", dtype=np.float64)
    elif np.shape(transform.matrix) == (3, 3):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.array(transform.matrix, order="F", dtype=np.float64)
    lagrange.transform_mesh(mesh, matrix)

    # BBox must be updated after affine transform.
    logger.debug("Updating view bbox due to affine transform.")
    view.initialize_bbox()


def _apply_compute_transform(view: View, transform: Compute):
    df = view.data_frame
    assert df is not None
    assert transform is not None
    mesh = df.mesh

    if transform.component is not None:
        lagrange.compute_components(mesh, output_attribute_name=transform.component)


def _apply_explode_transform(view: View, transform: Explode):
    df = view.data_frame
    assert df is not None
    assert transform is not None
    mesh = df.mesh
    assert mesh.has_attribute(transform.pieces)
    pieces_attr = mesh.attribute(transform.pieces)
    assert pieces_attr.element_type == lagrange.AttributeElement.Facet
    piece_index = pieces_attr.data
    # Remove edge attribute to avoid warnings.
    mesh.clear_edges()
    pieces = lagrange.separate_by_facet_groups(mesh, piece_index, map_attributes=True)

    center = np.average(mesh.vertices, axis=0)
    offset_dirs = np.array([np.average(p.vertices, axis=0) - center for p in pieces])
    offset_dirs *= transform.magnitude

    for i in range(len(pieces)):
        pieces[i].vertices = pieces[i].vertices + offset_dirs[i]

    df.mesh = lagrange.combine_meshes(pieces)


def apply_transform(view: View):
    """Apply a chain of transforms specified by view.transform to view.data_frame.
    Transforms are applied in the order specified by the chain.
    """

    def _apply(t: Transform | None):
        if t is None:
            return
        match (t):
            case Filter():
                assert view.data_frame is not None
                _apply_filter_transform(view, t)
            case UVMesh():
                assert view.data_frame is not None
                _appply_uv_mesh_transform(view, t)
            case Affine():
                assert view.data_frame is not None
                _apply_affine_transform(view, t)
            case Compute():
                assert view.data_frame is not None
                _apply_compute_transform(view, t)
            case Explode():
                assert view.data_frame is not None
                _apply_explode_transform(view, t)
            case _:
                raise NotImplementedError(f"Unsupported transform: {type(t)}!")

        _apply(t._child)

    _apply(view.transform)
