"""Internal geometry-plus-attributes data frame representation."""

from dataclasses import dataclass
import lagrange
from pathlib import Path
import numpy.typing as npt


@dataclass(slots=True)
class DataFrame:
    """Store spatial geometry and element attributes on a SurfaceMesh.

    Facet-free meshes represent point clouds; meshes with facets represent
    surfaces. Attributes act as typed columns defined on vertices, edges,
    facets, corners, or indexed values.

    Attributes:
        mesh: Lagrange geometry and attribute storage.
        roi_box: Optional axis-aligned region used for framing and normalization.
        source: Original mesh path for canonical serialization, or ``None`` for
            in-memory data.

    """

    mesh: lagrange.SurfaceMesh
    roi_box: npt.ArrayLike | None = None
    source: Path | None = None


DataFrameLike = object
"""Runtime data accepted by :func:`to_dataframe`.

Supported values are mesh paths, Lagrange meshes, Hakowan data frames, numeric
point arrays, pandas DataFrames, xarray Datasets, PyVista datasets, and Trimesh
objects. Optional third-party packages are detected without importing them.
"""
