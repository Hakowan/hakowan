"""Adapters from common Python data containers to Hakowan data frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import lagrange
import numpy as np
import numpy.typing as npt

from .dataframe import DataFrame

PositionColumns: TypeAlias = str | Sequence[str] | None

_EMBEDDABLE_ATTRIBUTE_USAGES = frozenset(
    {
        lagrange.AttributeUsage.Vector,
        lagrange.AttributeUsage.Position,
        lagrange.AttributeUsage.Normal,
        lagrange.AttributeUsage.Tangent,
        lagrange.AttributeUsage.Bitangent,
    }
)


def _module_name(value: object) -> str:
    return type(value).__module__.split(".", 1)[0]


def _positions(values: Any, *, label: str = "positions") -> npt.NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError(f"{label} must have shape (n, 2) or (n, 3); got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one point")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{label} must be numeric; got dtype {array.dtype}")
    result = np.asarray(array, dtype=np.float64)
    if result.shape[1] == 2:
        result = np.column_stack((result, np.zeros(result.shape[0], dtype=np.float64)))
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite coordinates")
    return np.ascontiguousarray(result)


def _position_names(
    columns: Sequence[Any], positions: PositionColumns
) -> tuple[str, ...]:
    names = tuple(str(column) for column in columns)
    if positions is not None:
        requested = (positions,) if isinstance(positions, str) else tuple(positions)
        if len(requested) not in (2, 3):
            raise ValueError("positions must name two or three columns")
        missing = [name for name in requested if name not in names]
        if missing:
            raise ValueError(f"Position columns do not exist: {missing}")
        return requested
    for candidate in (("x", "y", "z"), ("X", "Y", "Z"), ("x", "y"), ("X", "Y")):
        if all(name in names for name in candidate):
            return candidate
    raise ValueError(
        "Could not infer position columns; provide positions=('x', 'y', 'z')"
    )


def _usage(values: npt.NDArray[Any]) -> lagrange.AttributeUsage:
    channels = 1 if values.ndim == 1 else values.shape[1]
    return (
        lagrange.AttributeUsage.Scalar
        if channels == 1
        else lagrange.AttributeUsage.Vector
    )


def _add_attribute(
    mesh: lagrange.SurfaceMesh,
    name: str,
    values: Any,
    element: lagrange.AttributeElement,
    count: int,
) -> None:
    array = np.asarray(values)
    if array.ndim == 0 or array.shape[0] != count:
        return
    if array.ndim > 2 or not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        return
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if np.issubdtype(array.dtype, np.bool_):
        array = array.astype(np.uint8)
    mesh.create_attribute(
        str(name),
        element=element,
        usage=_usage(array),
        initial_values=np.array(array, copy=True, order="C"),
    )


def _point_mesh(points: Any, attributes: Mapping[str, Any]) -> lagrange.SurfaceMesh:
    vertices = _positions(points)
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(vertices)
    for name, values in attributes.items():
        _add_attribute(
            mesh,
            name,
            values,
            lagrange.AttributeElement.Vertex,
            mesh.num_vertices,
        )
    return mesh


def _embed_2d_attribute_values(values: Any) -> np.ndarray:
    array = np.array(values, copy=True, order="C")
    if array.ndim != 2 or array.shape[1] != 2:
        return array
    result = np.zeros((array.shape[0], 3), dtype=array.dtype)
    result[:, :2] = array
    return result


def _copy_mesh_attributes(
    source: lagrange.SurfaceMesh,
    target: lagrange.SurfaceMesh,
) -> None:
    for attribute_id in source.get_matching_attribute_ids():
        name = source.get_attribute_name(attribute_id)
        if name is None:
            continue
        if source.is_attribute_indexed(name):
            attribute = source.indexed_attribute(name)
            values = attribute.values.data
            if attribute.usage in _EMBEDDABLE_ATTRIBUTE_USAGES:
                values = _embed_2d_attribute_values(values)
            else:
                values = np.array(values, copy=True, order="C")
            target.create_attribute(
                name,
                element=attribute.element_type,
                usage=attribute.usage,
                initial_values=values,
                initial_indices=np.array(attribute.indices.data, copy=True, order="C"),
            )
        else:
            regular_attribute = source.attribute(name)
            values = regular_attribute.data
            if regular_attribute.usage in _EMBEDDABLE_ATTRIBUTE_USAGES:
                values = _embed_2d_attribute_values(values)
            else:
                values = np.array(values, copy=True, order="C")
            target.create_attribute(
                name,
                element=regular_attribute.element_type,
                usage=regular_attribute.usage,
                initial_values=values,
            )


def _embed_2d_mesh(mesh: lagrange.SurfaceMesh) -> lagrange.SurfaceMesh:
    """Embed a 2D Lagrange mesh in the XY plane for Hakowan's 3D pipeline."""
    if mesh.dimension != 2:
        return mesh

    embedded = lagrange.SurfaceMesh(dimension=3)
    vertices = np.zeros((mesh.num_vertices, 3), dtype=mesh.vertices.dtype)
    vertices[:, :2] = mesh.vertices
    embedded.add_vertices(vertices)
    if mesh.num_facets and mesh.is_triangle_mesh:
        embedded.add_triangles(np.ascontiguousarray(mesh.facets, dtype=np.uint32))
    elif mesh.num_facets:
        for facet_id in range(mesh.num_facets):
            facet = np.array(
                mesh.get_facet_vertices(facet_id),
                dtype=np.uint32,
                copy=True,
                order="C",
            )
            embedded.add_polygon(facet)
    if mesh.has_edges:
        embedded.initialize_edges()
    _copy_mesh_attributes(mesh, embedded)
    return embedded


def _add_mapping_attribute(
    mesh: lagrange.SurfaceMesh,
    name: str,
    values: Any,
    element: lagrange.AttributeElement,
    count: int,
    *,
    embed_2d: bool,
) -> None:
    array = np.asarray(values)
    if array.ndim == 0 or array.shape[0] != count:
        raise ValueError(
            f"{element} attribute {name!r} must have {count} values; "
            f"got shape {array.shape}"
        )
    if array.ndim > 2:
        raise ValueError(
            f"{element} attribute {name!r} must be one- or two-dimensional"
        )
    if not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"{element} attribute {name!r} must be numeric")
    if array.ndim == 2 and array.shape[1] == 0:
        raise ValueError(f"{element} attribute {name!r} must have at least one channel")
    if embed_2d:
        array = _embed_2d_attribute_values(array)
    _add_attribute(mesh, name, array, element, count)


def _mesh_mapping(data: Mapping[str, Any]) -> lagrange.SurfaceMesh:
    """Build a mesh from explicit vertex/facet arrays and optional attributes."""
    allowed = {"vertices", "facets", "vertex_attributes", "facet_attributes"}
    missing = {"vertices", "facets"} - data.keys()
    unknown = data.keys() - allowed
    if missing:
        raise ValueError(f"Mesh mapping is missing required keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Mesh mapping has unsupported keys: {sorted(unknown)}")

    raw_vertices = np.asarray(data["vertices"])
    embed_2d = raw_vertices.ndim == 2 and raw_vertices.shape[1] == 2
    vertices = _positions(raw_vertices, label="mesh vertices")
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(vertices)

    facets = np.asarray(data["facets"])
    if facets.size == 0:
        if facets.ndim > 2 or (facets.ndim == 2 and facets.shape[0] > 0):
            raise ValueError(
                f"faces must have shape (m, k), k >= 3; got {facets.shape}"
            )
    else:
        if facets.ndim != 2 or facets.shape[1] < 3:
            raise ValueError(
                f"faces must have shape (m, k), k >= 3; got {facets.shape}"
            )
        if not np.issubdtype(facets.dtype, np.integer):
            raise TypeError("mesh facets must contain integer vertex indices")
        if np.any(facets < 0) or np.any(facets >= mesh.num_vertices):
            raise ValueError("mesh facet indices must refer to existing vertices")
    _add_facets(mesh, facets)

    for key, element, count in (
        ("vertex_attributes", lagrange.AttributeElement.Vertex, mesh.num_vertices),
        ("facet_attributes", lagrange.AttributeElement.Facet, mesh.num_facets),
    ):
        attributes = data.get(key, {})
        if not isinstance(attributes, Mapping):
            raise TypeError(f"{key} must be a mapping of names to values")
        for name, values in attributes.items():
            _add_mapping_attribute(
                mesh,
                str(name),
                values,
                element,
                count,
                embed_2d=embed_2d,
            )
    return mesh


def _table_mesh(table: Any, positions: PositionColumns) -> lagrange.SurfaceMesh:
    names = _position_names(tuple(table.columns), positions)
    points = np.column_stack([np.asarray(table[name]) for name in names])
    attributes = {
        str(name): np.asarray(table[name])
        for name in table.columns
        if str(name) not in names
    }
    return _point_mesh(points, attributes)


def _xarray_mesh(dataset: Any, positions: PositionColumns) -> lagrange.SurfaceMesh:
    available = tuple(
        dict.fromkeys((*dataset.coords.keys(), *dataset.data_vars.keys()))
    )
    names = _position_names(available, positions)
    coordinate_arrays = [np.asarray(dataset[name].values) for name in names]
    if any(array.ndim != 1 for array in coordinate_arrays):
        raise ValueError("xarray position variables must be one-dimensional")
    points = np.column_stack(coordinate_arrays)
    count = points.shape[0]
    attributes: dict[str, Any] = {}
    for name in available:
        if str(name) in names:
            continue
        values = np.asarray(dataset[name].values)
        if values.ndim in (1, 2) and values.shape[0] == count:
            attributes[str(name)] = values
    return _point_mesh(points, attributes)


def _add_facets(mesh: lagrange.SurfaceMesh, facets: Any) -> None:
    array = np.asarray(facets)
    if array.size == 0:
        return
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"faces must have shape (m, k), k >= 3; got {array.shape}")
    indices = np.ascontiguousarray(array, dtype=np.uint32)
    if indices.shape[1] == 3:
        mesh.add_triangles(indices)
    else:
        for facet in indices:
            mesh.add_polygon(np.ascontiguousarray(facet, dtype=np.uint32))


def _packed_faces(values: Any) -> list[list[int]]:
    packed = np.asarray(values, dtype=np.int64).reshape(-1)
    result: list[list[int]] = []
    offset = 0
    while offset < packed.size:
        size = int(packed[offset])
        if size < 3 or offset + size >= packed.size:
            raise ValueError("Invalid PyVista packed face array")
        result.append(packed[offset + 1 : offset + size + 1].tolist())
        offset += size + 1
    return result


def _pyvista_mesh(data: Any) -> lagrange.SurfaceMesh:
    faces = getattr(data, "faces", None)
    if faces is None:
        data = data.extract_surface()
        faces = data.faces
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(_positions(data.points, label="PyVista points"))
    faces = _packed_faces(faces)
    for face in faces:
        mesh.add_polygon(np.ascontiguousarray(face, dtype=np.uint32))
    for name, values in data.point_data.items():
        _add_attribute(
            mesh, name, values, lagrange.AttributeElement.Vertex, mesh.num_vertices
        )
    for name, values in data.cell_data.items():
        _add_attribute(
            mesh, name, values, lagrange.AttributeElement.Facet, mesh.num_facets
        )
    return mesh


def _trimesh_mesh(data: Any) -> lagrange.SurfaceMesh:
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(_positions(data.vertices, label="Trimesh vertices"))
    _add_facets(mesh, data.faces)
    for name, values in data.vertex_attributes.items():
        _add_attribute(
            mesh, name, values, lagrange.AttributeElement.Vertex, mesh.num_vertices
        )
    for name, values in data.face_attributes.items():
        _add_attribute(
            mesh, name, values, lagrange.AttributeElement.Facet, mesh.num_facets
        )
    return mesh


def _mesh_file(path: Path) -> lagrange.SurfaceMesh:
    return lagrange.io.load_mesh(path, quiet=True, stitch_vertices=True)


def to_dataframe(
    data: object,
    *,
    positions: PositionColumns = None,
    roi_box: npt.ArrayLike | None = None,
) -> DataFrame:
    """Convert supported geometry or tabular data to a Hakowan DataFrame.

    Paths, Lagrange meshes, and mappings with ``vertices`` and ``facets`` retain
    topology. Two-dimensional Lagrange meshes and mapping vertices are embedded
    in the XY plane. Numeric arrays become 2D/3D point clouds.
    Pandas and xarray inputs use inferred or explicit position columns; numeric fields become
    vertex attributes. PyVista and Trimesh adapters preserve supported
    point/vertex and cell/facet attributes without importing optional libraries
    inside Hakowan.

    Args:
        data: Mesh path, Lagrange mesh, mapping of mesh arrays, point array,
            table, PyVista dataset, Trimesh object, or existing Hakowan DataFrame.
        positions: Two or three position column names for pandas/xarray input.
        roi_box: Optional axis-aligned region of interest.

    Returns:
        A DataFrame backed by a Lagrange SurfaceMesh.

    Raises:
        TypeError: If the input type or option combination is unsupported.
        ValueError: If positions, topology, or numeric values are invalid.

    """
    if isinstance(data, DataFrame):
        if positions is not None:
            raise TypeError("positions cannot be used with an existing DataFrame")
        mesh = _embed_2d_mesh(data.mesh)
        if roi_box is None:
            if mesh is data.mesh:
                return data
            return DataFrame(mesh=mesh, roi_box=data.roi_box, source=data.source)
        return DataFrame(mesh=mesh, roi_box=roi_box, source=data.source)
    if isinstance(data, (str, Path)):
        if positions is not None:
            raise TypeError("positions cannot be used with a mesh file")
        source = Path(data)
        return DataFrame(
            mesh=_embed_2d_mesh(_mesh_file(source)),
            roi_box=roi_box,
            source=source,
        )
    if isinstance(data, lagrange.SurfaceMesh):
        if positions is not None:
            raise TypeError("positions cannot be used with a SurfaceMesh")
        return DataFrame(mesh=_embed_2d_mesh(data), roi_box=roi_box)
    module = _module_name(data)
    if module == "pandas":
        return DataFrame(mesh=_table_mesh(data, positions), roi_box=roi_box)
    if module == "xarray":
        return DataFrame(mesh=_xarray_mesh(data, positions), roi_box=roi_box)
    if isinstance(data, Mapping):
        if positions is not None:
            raise TypeError("positions cannot be used with a mesh mapping")
        return DataFrame(mesh=_mesh_mapping(data), roi_box=roi_box)
    if isinstance(data, np.ndarray) or (
        isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray))
    ):
        if positions is not None:
            raise TypeError("positions cannot be used with a position array")
        return DataFrame(mesh=_point_mesh(data, {}), roi_box=roi_box)

    if module == "pyvista" and hasattr(data, "points"):
        if positions is not None:
            raise TypeError("positions cannot be used with PyVista geometry")
        return DataFrame(mesh=_pyvista_mesh(data), roi_box=roi_box)
    if module == "trimesh" and hasattr(data, "vertices") and hasattr(data, "faces"):
        if positions is not None:
            raise TypeError("positions cannot be used with Trimesh geometry")
        return DataFrame(mesh=_trimesh_mesh(data), roi_box=roi_box)
    raise TypeError(f"Unsupported data type: {type(data)!r}")


__all__ = ["PositionColumns", "to_dataframe"]
