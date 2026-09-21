"""Small deterministic scientific datasets used by the LLM benchmark."""

from __future__ import annotations

from functools import lru_cache

import lagrange
import numpy as np


def _attribute(mesh: lagrange.SurfaceMesh, name: str, values, *, usage) -> None:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    mesh.create_attribute(
        name,
        element=lagrange.AttributeElement.Vertex,
        usage=usage,
        initial_values=np.array(array, copy=True, order="C"),
    )


def _surface(vertices: list[list[float]], facets: list[list[int]]) -> lagrange.SurfaceMesh:
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.asarray(vertices, dtype=np.float64))
    mesh.add_triangles(np.asarray(facets, dtype=np.uint32))
    return mesh


def _scalar_surface() -> lagrange.SurfaceMesh:
    mesh = _surface(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0.2], [-1, 1, 0.1]],
        [[0, 1, 2], [0, 2, 3]],
    )
    _attribute(mesh, "temperature", [10, 20, 40, 30], usage=lagrange.AttributeUsage.Scalar)
    _attribute(mesh, "pressure", [0.1, 1, 10, 100], usage=lagrange.AttributeUsage.Scalar)
    _attribute(mesh, "constant", [2, 2, 2, 2], usage=lagrange.AttributeUsage.Scalar)
    _attribute(
        mesh,
        "velocity",
        [[1, 0, 0], [0, 2, 0], [-1, 0, 0], [0, -2, 0]],
        usage=lagrange.AttributeUsage.Vector,
    )
    return mesh


def _signed_surface() -> lagrange.SurfaceMesh:
    mesh = _surface(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
        [[0, 1, 2], [0, 2, 3]],
    )
    _attribute(mesh, "signed", [-1, 0, 1, 2], usage=lagrange.AttributeUsage.Scalar)
    _attribute(mesh, "quality", [1, np.nan, 3, 4], usage=lagrange.AttributeUsage.Scalar)
    return mesh


def _point_cloud() -> lagrange.SurfaceMesh:
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(
        np.asarray(
            [[-1, 0, 0], [-0.5, 0.5, 0], [0, 0, 0], [0.5, -0.5, 0], [1, 0, 0]],
            dtype=np.float64,
        )
    )
    _attribute(mesh, "pressure", [1, 2, 4, 8, 16], usage=lagrange.AttributeUsage.Scalar)
    _attribute(
        mesh,
        "velocity",
        [[1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0], [-1, 0, 0]],
        usage=lagrange.AttributeUsage.Vector,
    )
    return mesh


def _disconnected() -> lagrange.SurfaceMesh:
    mesh = _surface(
        [
            [-2, 0, 0],
            [-1, 0, 0],
            [-1.5, 1, 0],
            [1, 0, 0],
            [2, 0, 0],
            [1.5, 1, 0],
        ],
        [[0, 1, 2], [3, 4, 5]],
    )
    _attribute(mesh, "region", [0, 0, 0, 1, 1, 1], usage=lagrange.AttributeUsage.Scalar)
    return mesh


def _elongated() -> lagrange.SurfaceMesh:
    mesh = _surface(
        [[-50, -0.2, 0], [50, -0.2, 0], [50, 0.2, 0], [-50, 0.2, 0]],
        [[0, 1, 2], [0, 2, 3]],
    )
    _attribute(mesh, "stress", [0, 0.25, 1, 0.75], usage=lagrange.AttributeUsage.Scalar)
    return mesh


_BUILDERS = {
    "scalar_surface": _scalar_surface,
    "signed_surface": _signed_surface,
    "point_cloud": _point_cloud,
    "disconnected": _disconnected,
    "elongated": _elongated,
}


@lru_cache(maxsize=None)
def dataset(name: str) -> lagrange.SurfaceMesh:
    """Return one cached deterministic benchmark mesh by name."""
    try:
        return _BUILDERS[name]()
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark dataset {name!r}") from exc




def case_resolver(dataset_name: str):
    """Return a resolver callable bound to one benchmark dataset."""
    mesh = dataset(dataset_name)

    def resolve(identifier: str):
        if identifier != "data":
            raise KeyError(identifier)
        return mesh

    return resolve


def names() -> tuple[str, ...]:
    """Return available generated dataset names."""
    return tuple(_BUILDERS)
