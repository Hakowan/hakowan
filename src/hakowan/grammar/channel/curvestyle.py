from dataclasses import dataclass
from typing import Literal

from ..scale import AttributeLike


@dataclass
class CurveStyle:
    """Curve style base class."""

    pass


@dataclass
class Bend(CurveStyle):
    """Curve bending style.

    Attributes:
        direction (AttributeLike): The attribute used to encode the bending direction.
        bend_type (Literal["n", "r", "s"]): The type of bending.
            ``"n"`` = normal bend, ``"r"`` = ribbon bend, ``"s"`` = smooth bend.
            The default value is ``"n"``.
    """

    direction: AttributeLike
    bend_type: Literal["n", "r", "s"] = "n"
