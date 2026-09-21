"""Point, directional, and environment emitter settings."""

from ..common.color import ColorLike

from dataclasses import dataclass, field
from pathlib import Path
import numpy.typing as npt


@dataclass(kw_only=True, slots=True)
class Emitter:
    """Emitter dataclass contains lighting-related settings."""

    pass


@dataclass(kw_only=True, slots=True)
class Point(Emitter):
    """Point light source.

    Attributes:
        intensity: Light intensity.
        position: Light position.

    """

    intensity: ColorLike | float = 1.0
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    color: ColorLike | None = None

@dataclass(kw_only=True, slots=True)
class Directional(Emitter):
    """Directional light with rays traveling along ``direction``."""

    direction: list[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])
    intensity: float = 1.0
    color: ColorLike = "white"


@dataclass(kw_only=True, slots=True)
class Envmap(Emitter):
    """Environment light (i.e. image-based lighting).

    Attributes:
        filename: Path to the environment light image file.
        scale: Scaling factor to be applied to the environment light.
        up: Up vector of the environment light.
        rotation: Rotation angle of the environment light around the up direction.

    """

    filename: Path = field(
        default_factory=lambda: Path(__file__).parents[1] / "envmaps" / "museum.exr"
    )
    scale: float = 1.0
    up: npt.ArrayLike = field(default_factory=lambda: [0, 1, 0])
    rotation: float = 180.0
