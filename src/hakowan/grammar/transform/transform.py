from dataclasses import dataclass
from typing import Callable, Optional
import copy
import numpy.typing as npt

from ..scale import Attribute, AttributeLike


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


@dataclass(slots=True, kw_only = True)
class Filter(Transform):
    """Filter data based on a condition.

    Attributes:
        data: The attribute to filter on. If None, the vertex position is used.
        condition: A callable that takes a single argument, the value of the attribute, and returns
            a boolean indicating whether the data should be kept.
    """

    data: AttributeLike | None = None
    condition: Callable


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
