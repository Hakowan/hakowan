from dataclasses import dataclass

from ....common.color import ColorLike


@dataclass(slots=True)
class Medium:
    """Medium represents a volumetric material inside/outside of a shape.

    Attributes:
        albedo (ColorLike) : The albedo of the medium.
        scale (float): The scale of the medium.
    """

    albedo: ColorLike = 0.75
    scale: float = 1.0
