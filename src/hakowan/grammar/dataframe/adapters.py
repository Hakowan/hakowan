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
    mesh = lagrange.io.load_mesh(path, quiet=True, stitch_vertices=True)
    if mesh.num_facets == 0:
        # Lagrange's vertex stitching converts point-cloud vertex attributes to
        # empty indexed attributes because there are no corners to reindex.
        return lagrange.io.load_mesh(path, quiet=True, stitch_vertices=False)
    return mesh


def to_dataframe(
    data: object,
    *,
    positions: PositionColumns = None,
    roi_box: npt.ArrayLike | None = None,
) -> DataFrame:
    """Convert supported geometry or tabular data to a Hakowan DataFrame.

    Paths and Lagrange meshes retain topology. Numeric arrays become 2D/3D
    point clouds. Pandas and xarray inputs use inferred or explicit position
    columns; numeric fields become vertex attributes. PyVista and Trimesh
    adapters preserve supported point/vertex and cell/facet attributes without
    importing optional libraries inside Hakowan.

    Args:
        data: Mesh path, Lagrange mesh, point array, table, PyVista dataset,
            Trimesh object, or existing Hakowan DataFrame.
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
        if roi_box is None:
            return data
        return DataFrame(mesh=data.mesh, roi_box=roi_box, source=data.source)
    if isinstance(data, (str, Path)):
        if positions is not None:
            raise TypeError("positions cannot be used with a mesh file")
        source = Path(data)
        return DataFrame(
            mesh=_mesh_file(source),
            roi_box=roi_box,
            source=source,
        )
    if isinstance(data, lagrange.SurfaceMesh):
        if positions is not None:
            raise TypeError("positions cannot be used with a SurfaceMesh")
        return DataFrame(mesh=data, roi_box=roi_box)
    if isinstance(data, np.ndarray) or (
        isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray))
    ):
        if positions is not None:
            raise TypeError("positions cannot be used with a position array")
        return DataFrame(mesh=_point_mesh(data, {}), roi_box=roi_box)

    module = _module_name(data)
    if module == "pandas":
        return DataFrame(mesh=_table_mesh(data, positions), roi_box=roi_box)
    if module == "xarray":
        return DataFrame(mesh=_xarray_mesh(data, positions), roi_box=roi_box)
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
