from dataclasses import dataclass, field
import numpy.typing as npt
from pathlib import Path

from ..common.color import Color


@dataclass(kw_only=True)
class Emitter:
    pass


@dataclass(kw_only=True)
class Point(Emitter):
    intensity: Color | float
    position: npt.NDArray


@dataclass(kw_only=True)
class Envmap(Emitter):
    filename: Path = field(
        default_factory=lambda: Path(__file__).parents[2] / "envmaps" / "museum.exr"
    )
    scale: float = 1.0
