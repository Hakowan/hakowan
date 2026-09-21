"""Structured, JSON-safe summaries of Hakowan data inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import lagrange
import numpy as np

from .grammar.dataframe import DataFrameLike, PositionColumns, to_dataframe


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _json_number(value: Any) -> int | float | None:
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, (int, np.integer)):
        return int(value)
    result = float(value)
    return result if np.isfinite(result) else None


def _json_vector(value: Any) -> int | float | list[int | float | None] | None:
    array = np.asarray(value)
    if array.ndim == 0 or array.size == 1:
        return _json_number(array.reshape(-1)[0])
    return [_json_number(item) for item in array.reshape(-1)]


@dataclass(frozen=True, slots=True)
class AttributeSummary:
    """Machine-readable metadata and statistics for one mesh attribute."""

    name: str
    element: str
    usage: str
    indexed: bool
    channels: int
    dtype: str
    element_count: int
    value_count: int
    finite_count: int
    nonfinite_count: int
    minimum: int | float | list[int | float | None] | None
    maximum: int | float | list[int | float | None] | None
    quantiles: dict[str, int | float | list[int | float | None] | None]

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a representation accepted by :func:`json.dumps`."""
        return asdict(self)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class DataSummary:
    """Machine-readable geometry, topology, and attribute summary."""

    source: str | None
    dimension: int
    vertex_count: int
    facet_count: int
    corner_count: int
    edge_count: int | None
    edges_initialized: bool
    bounds: list[list[float]] | None
    attributes: tuple[AttributeSummary, ...]

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a representation accepted by :func:`json.dumps`."""
        result = asdict(self)
        result["attributes"] = [attribute.to_dict() for attribute in self.attributes]
        return result  # type: ignore[return-value]

    def attribute(self, name: str) -> AttributeSummary:
        """Look up an attribute by name.

        Raises:
            KeyError: If the input has no attribute named ``name``.
        """
        for attribute in self.attributes:
            if attribute.name == name:
                return attribute
        raise KeyError(name)


def _load_data(
    data: DataFrameLike, positions: PositionColumns = None
) -> tuple[lagrange.SurfaceMesh, str | None]:
    frame = to_dataframe(data, positions=positions)
    return frame.mesh, str(frame.source) if frame.source is not None else None


def _attribute_values(
    mesh: lagrange.SurfaceMesh, name: str
) -> tuple[Any, np.ndarray, np.ndarray, bool]:
    indexed = mesh.is_attribute_indexed(name)
    if indexed:
        attribute = mesh.indexed_attribute(name)
        values = np.asarray(attribute.values.data)
        indices = np.asarray(attribute.indices.data)
    else:
        attribute = mesh.attribute(name)
        values = np.asarray(attribute.data)
        indices = np.empty(0, dtype=np.uint32)
    return attribute, values, indices, indexed


def _summarize_attribute(mesh: lagrange.SurfaceMesh, name: str) -> AttributeSummary:
    attribute, values, indices, indexed = _attribute_values(mesh, name)
    rows = values.reshape(values.shape[0], -1) if values.ndim > 1 else values.reshape(-1, 1)
    numeric = np.issubdtype(rows.dtype, np.number)

    if numeric and rows.size:
        finite_rows = np.all(np.isfinite(rows), axis=1)
        finite_values = rows[finite_rows]
    else:
        finite_rows = np.ones(rows.shape[0], dtype=bool)
        finite_values = rows

    if numeric and finite_values.size:
        minimum = _json_vector(np.min(finite_values, axis=0))
        maximum = _json_vector(np.max(finite_values, axis=0))
        quantiles = {
            label: _json_vector(np.quantile(finite_values, q, axis=0))
            for label, q in (
                ("0", 0.0),
                ("0.25", 0.25),
                ("0.5", 0.5),
                ("0.75", 0.75),
                ("1", 1.0),
            )
        }
    else:
        minimum = maximum = None
        quantiles = {}

    element_count = int(indices.size) if indexed else int(rows.shape[0])
    return AttributeSummary(
        name=name,
        element=attribute.element_type.name.lower(),
        usage=attribute.usage.name.lower(),
        indexed=indexed,
        channels=int(attribute.num_channels),
        dtype=str(values.dtype),
        element_count=element_count,
        value_count=int(rows.shape[0]),
        finite_count=int(np.count_nonzero(finite_rows)),
        nonfinite_count=int(finite_rows.size - np.count_nonzero(finite_rows)),
        minimum=minimum,
        maximum=maximum,
        quantiles=quantiles,
    )


def inspect(
    data: DataFrameLike, *, positions: PositionColumns = None
) -> DataSummary:
    """Inspect supported geometry or tabular data without changing it.

    The returned dataclasses contain only JSON-safe metadata. Attribute statistics
    are computed from unique values for indexed attributes; ``element_count``
    reports the index count while ``value_count`` reports the unique-value count.

    Args:
        data: Any input accepted by :func:`hkw.dataframe.to_dataframe`.
        positions: Optional position column names for pandas and xarray inputs.

    Returns:
        Geometry, topology, attribute-domain, and numeric-range metadata.
    """
    mesh, source = _load_data(data, positions)
    if mesh.num_vertices:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        bounds = [np.min(vertices, axis=0).tolist(), np.max(vertices, axis=0).tolist()]
    else:
        bounds = None

    names = sorted(
        mesh.get_attribute_name(attribute_id)
        for attribute_id in mesh.get_matching_attribute_ids()
    )
    attributes = tuple(_summarize_attribute(mesh, name) for name in names)
    return DataSummary(
        source=source,
        dimension=int(mesh.dimension),
        vertex_count=int(mesh.num_vertices),
        facet_count=int(mesh.num_facets),
        corner_count=int(mesh.num_corners),
        edge_count=int(mesh.num_edges) if mesh.has_edges else None,
        edges_initialized=bool(mesh.has_edges),
        bounds=bounds,
        attributes=attributes,
    )


__all__ = ["AttributeSummary", "DataSummary", "inspect"]
