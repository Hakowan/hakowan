from ..common.color import Color

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import numpy.typing as npt


@dataclass(kw_only=True, slots=True)
class Emitter:
    pass


@dataclass(kw_only=True, slots=True)
class Point(Emitter):
    intensity: Color | float
    position: npt.NDArray


@dataclass(kw_only=True, slots=True)
class Envmap(Emitter):
    filename: Path = field(
        default_factory=lambda: Path(__file__).parents[1] / "envmaps" / "museum.exr"
    )
    scale: float = 1.0
    up: npt.NDArray = field(default_factory=lambda: np.array([0, 1, 0]))
    rotation: float = 180.0
