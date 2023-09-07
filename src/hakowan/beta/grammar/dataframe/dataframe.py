from dataclasses import dataclass

import lagrange


@dataclass(kw_only=True)
class DataFrame:
    """DataFrame represents data that are stored on a 3D surface.

    A DataFrame contains a reference to a SurfaceMesh object, which defines the 3D geometry where
    data are stored. The mesh object also contains a set of attributes, which can be thought of as
    columns in traditional table-based data representation. Each attribute defines data values
    associated with mesh vertices, edges, facets, etc.
    """

    mesh: lagrange.SurfaceMesh
