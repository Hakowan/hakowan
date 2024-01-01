from dataclasses import dataclass
import lagrange
from typing import TypeAlias
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
    """

    mesh: lagrange.SurfaceMesh
    roi_box: npt.ArrayLike | None = None


DataFrameLike: TypeAlias = str | Path | lagrange.SurfaceMesh | DataFrame
"""Type alias for objects that can be converted to a DataFrame.

* A string or a Path object is interpreted as a path to a file that contains a mesh object. A
`DataFrame` object will be created with the loaded mesh.
* A SurfaceMesh object will create a DataFrame object with the mesh object.
* A DataFrame object will be unchanged.
"""
