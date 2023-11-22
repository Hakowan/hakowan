from dataclasses import dataclass

from ....common.color import ColorLike

@dataclass(slots=True)
class Medium:
    albedo: ColorLike = 0.75
    scale: float = 1.0
