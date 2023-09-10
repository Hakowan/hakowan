from dataclasses import dataclass, field
import numpy.typing as npt
from pathlib import Path

from ..common import Color


@dataclass(kw_only=True)
class Emitter:
    pass


@dataclass(kw_only=True)
class Point(Emitter):
    intensity: Color | float
    position: npt.NDArray[float]


@dataclass(kw_only=True)
class Envmap(Emitter):
    filename: Path = field(
        default_factory=lambda: Path(__path__).parents[1] / "envmap" / "envmap.exr"
    )
    scale: float = 1.0
