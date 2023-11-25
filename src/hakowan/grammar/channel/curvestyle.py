from dataclasses import dataclass

from ..scale import AttributeLike


@dataclass
class CurveStyle:
    pass


@dataclass
class Bend(CurveStyle):
    direction: AttributeLike
    bend_type: str = "n"
