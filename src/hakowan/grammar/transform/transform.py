from dataclasses import dataclass, field
from typing import Callable, Optional
import copy
import numpy as np
import numpy.typing as npt

from ..scale import AttributeLike

__all__ = [
    "Transform",
    "Filter",
    "Clip",
    "UVMesh",
    "Affine",
    "PrincipalAxes",
    "Normalize",
    "Compute",
    "Explode",
    "Norm",
    "Boundary",
    "Streamline",
]


def _default_condition(x) -> bool:  # module-level so it is picklable
    return True


@dataclass(kw_only=True, slots=True)
class Transform:
    """Transform is the base class of all transforms."""

    _child: Optional["Transform"] = None

    def __imul__(self, other: "Transform") -> "Transform":
        """In place update by applying another transform after the current transform.

        Args:
            other: The transform to apply after the current transform.
        """
        # Because transform may be used in multiple places in the layer graph, and it may have a
        # child in the future, it must be deep copied to avoid undesired side effects.
        if self._child is None:
            self._child = copy.deepcopy(other)
        else:
            t = self._child
            while t._child is not None:
                t = t._child
            t._child = copy.deepcopy(other)
        return self

    def __mul__(self, other: "Transform") -> "Transform":
        """Apply another transform, `other`, after the current transform.

        Args:
            other: The other transform.

        Returns: A new transform that is the composition of the current transform and `other`.
        """
        r = copy.deepcopy(self)
        r *= other
        return r


@dataclass(slots=True)
class Filter(Transform):
    """Filter data based on a condition.

    Attributes:
        data: The attribute to filter on. If None, the vertex position is used.
        condition: A callable that takes a single argument, the value of the attribute, and returns
            a boolean indicating whether the data should be kept.
    """

    data: AttributeLike | None = None
    condition: Callable = field(default=_default_condition)


@dataclass(slots=True)
class Clip(Transform):
    """Clip the mesh against a plane, keeping only the half-space the normal points into.

    Unlike :class:`Filter`, which keeps or drops whole facets, ``Clip`` slices
    through triangles: facets straddling the plane are cut so that only the part
    on the positive side of the plane is kept (partial triangles are produced).
    The exposed cross-section is left open (it is not capped).

    The plane is defined in the data/object coordinate space (the same space the
    raw mesh lives in, before any layer-level :class:`Affine` transform), keeping
    its meaning consistent with :class:`Filter`.

    Attributes:
        point: A point lying on the clipping plane.
        normal: The plane normal. The half-space where
            ``dot(normal, x - point) >= 0`` is kept; the rest is clipped away.
    """

    point: npt.ArrayLike = field(default_factory=lambda: np.zeros(3))
    normal: npt.ArrayLike = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))


@dataclass(slots=True)
class UVMesh(Transform):
    """Extract UV mesh from data.

    Attributes:
        uv: The attribute defining the UV coordinates. If None, automatically deetect the UV
            attribute from the data.
    """

    uv: AttributeLike | None = None


@dataclass(slots=True)
class Affine(Transform):
    """Apply affine transformation to data.

    Attributes:
        matrix: The 4x4 affine matrix to apply.
    """

    matrix: npt.ArrayLike


@dataclass(slots=True)
class PrincipalAxes(Transform):
    """Align PCA principal directions of vertex positions with a target orthonormal frame.

    Covariance is computed from the current data-frame vertex positions. Principal
    axes are ordered by descending eigenvalue (largest variance first). The rotation and
    translation match those directions to the columns of ``frame``: column 0 is the
    direction for the largest-variance axis, column 1 for the second, column 2 for the third.

    The resulting affine is pre-composed with any prior global transform on the layer, so
    earlier ``Affine`` transforms (translate / rotate / scale) are preserved and applied
    before this PCA-based alignment.

    Attributes:
        frame: 3x3 matrix whose columns are the target orthonormal axes (see above).
        orthonormalize_frame: If True (default), orthonormalize ``frame`` with QR so mildly
            skewed inputs still yield a proper rotation.
    """

    frame: npt.ArrayLike = field(default_factory=lambda: np.eye(3))
    orthonormalize_frame: bool = True


@dataclass(slots=True)
class Normalize(Transform):
    """Recenter and uniformly scale the mesh to fit a unit box centered at the origin.

    Wraps :func:`lagrange.normalize_mesh`: vertex positions are translated so the
    bounding-box center sits at the origin and uniformly scaled so the bounding-box
    diagonal is 2 (i.e. the geometry fits inside the unit sphere). Use it to bring
    meshes from unrelated coordinate systems to a comparable on-screen size — for
    example when laying several meshes side by side with :meth:`Layer.juxtapose`.

    Unlike a layer-level :class:`Affine`, this mutates the data-frame vertices in
    place, so it normalizes the geometry as it currently stands (after any earlier
    mesh-mutating transforms) and ignores prior global affine transforms — matching
    how :class:`PrincipalAxes` reads object-space positions.

    Attributes:
        normalize_normals: Re-normalize normal attributes to unit length. Default True.
        normalize_tangents_bitangents: Re-normalize tangent/bitangent attributes to
            unit length. Default True.
    """

    normalize_normals: bool = True
    normalize_tangents_bitangents: bool = True


@dataclass(slots=True, kw_only=True)
class Compute(Transform):
    """Compute new attributes from the current data frame.

    Attributes:
        x: Extract the x coordinate as an attribute.
        y: Extract the y coordinate as an attribute.
        z: Extract the z coordinate as an attribute.
        normal: Compute the normal vector field as an attribute.
        vertex_normal: Compute the vertex normal vector field as an attribute.
        facet_normal: Compute the facet normal vector field as an attribute.
        component: Compute connected component ids.
    """

    x: str | None = None
    y: str | None = None
    z: str | None = None
    normal: str | None = None
    vertex_normal: str | None = None
    facet_normal: str | None = None
    component: str | None = None


@dataclass(slots=True)
class Explode(Transform):
    """Explode data into multiple pieces.

    Attributes:
        pieces: The attribute defining the pieces.
        magnitude: The magnitude of the displacement.
    """

    pieces: AttributeLike
    magnitude: float = 1


@dataclass(slots=True)
class Norm(Transform):
    """Compute the row-wise norm of a given vector attribute.

    Attributes:
        data: The vector attribute to compute the norm on.
        norm_attr_name: The name of the output norm attribute.
        order: The order of the norm. Default is 2, which is the L2 norm.
    """

    data: AttributeLike
    norm_attr_name: str
    order: int = 2


@dataclass(slots=True)
class Boundary(Transform):
    """Compute the boundary of a mesh.

    Attributes:
        attributes: The attributes to take into account when computing the boundary.
            i.e. discontinuities in these attributes will be considered as boundaries.
    """

    attributes: list[str] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class Streamline(Transform):
    """Replace the mesh with surface streamlines traced from a per-facet vector or
    cross field.

    The output is a vertex-only mesh whose 2-vertex polygonal faces encode line
    segments along the streamlines, suitable for the ``curve`` mark.  A per-vertex
    ``int32`` attribute named by ``id_attr_name`` identifies which streamline
    each point belongs to.

    Attributes:
        vec_field: The per-facet vector field attribute name.  Vertex- or corner-
            domain attributes are averaged to per-facet first.
        n: Number of blue-noise seed faces to sample.  Default 50.
        cross_field: Treat the field as 4-RoSy cross field.  Default True.
        length: Maximum object-space length per half-trace (measured on the
            data-frame mesh, before any layer-level affine transforms).  Tracing
            stops once the accumulated length exceeds this value.  ``None`` means
            no limit (trace until mesh boundary).  Default None.
        seed: RNG seed passed to blue-noise sampling.  Default 0.
        min_length: Discard streamlines shorter than this many sample points.
            Default 3.
        id_attr_name: Name of the per-vertex streamline-id attribute on the
            output mesh.  Default ``_hakowan_streamline_id``.
    """

    vec_field: AttributeLike
    n: int = 50
    cross_field: bool = True
    length: float | None = None
    seed: int = 0
    min_length: int = 3
    id_attr_name: str = "_hakowan_streamline_id"
