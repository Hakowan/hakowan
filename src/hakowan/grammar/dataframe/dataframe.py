from dataclasses import dataclass
import lagrange
from pathlib import Path
import numpy.typing as npt


@dataclass(slots=True)
class DataFrame:
    """DataFrame represents data that are stored on a 3D surface.

    A DataFrame contains a reference to a SurfaceMesh object, which defines the 3D geometry where
    data are stored. The mesh object also contains a set of attributes, which can be thought of as
    columns in traditional table-based data representation. Each attribute defines data values
    associated with mesh vertices, edges, facets, etc.

    Attributes:
        mesh: A SurfaceMesh object that defines the 3D geometry where data are stored.
        roi_box: A box defining the region of interest. If None, the entire mesh is considered.
        source: Original mesh path when loaded from a file. Used by canonical
            specification serialization; None for in-memory meshes.

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
