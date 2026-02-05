from dataclasses import dataclass

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
        bend_type (str): The type of bending (options are 's', 'r', 'n'). The default value is 'n'.
    """

    direction: AttributeLike
    bend_type: str = "n"
