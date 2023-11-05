from dataclasses import dataclass
from typing import Self, Callable
import copy

from ..scale import Attribute, AttributeLike


@dataclass(kw_only=True, slots=True)
class Transform:
    _child: Self | None = None

    def __imul__(self, other: Self):
        """ Apply another transform, `other`, after the current transform."""
        # Because transform may be used in multiple places in the layer graph, and it may have a
        # child in the future, it must be deep copied to avoid undesired side effects.
        if self._child is None:
            self._child = copy.deepcopy(other)
        else:
            t = self._child
            while t._child is not None:
                t = t._child
            t._child = copy.deepcopy(other)


@dataclass(kw_only=True, slots=True)
class Filter(Transform):
    """ Filter data based on a condition.

    Attributes:
        data: The attribute to filter on.
        condition: A callable that takes a single argument, the value of the attribute, and returns
            a boolean indicating whether the data should be kept.
    """

    data: AttributeLike
    condition: Callable
