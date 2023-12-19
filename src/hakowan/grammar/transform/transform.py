from dataclasses import dataclass
from typing import Self, Callable, Optional
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


@dataclass(slots=True)
class Filter(Transform):
    """Filter data based on a condition.

    Attributes:
        data: The attribute to filter on.
        condition: A callable that takes a single argument, the value of the attribute, and returns
            a boolean indicating whether the data should be kept.
    """

    data: AttributeLike
    condition: Callable


@dataclass(slots=True)
class UVMesh(Transform):
    """Extract UV mesh from data."""

    uv: AttributeLike


@dataclass(slots=True)
class Affine(Transform):
    """Apply affine transformation to data.

    Attributes:
        matrix: The 4x4 affine matrix to apply.
    """

    matrix: npt.ArrayLike


@dataclass(slots=True)
class Compute(Transform):
    """Compute new attributes from the current data frame.

    Attributes:
        component: Compute connected component ids.
    """

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
