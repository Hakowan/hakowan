"""Point, directional, and environment emitter settings."""

from ..common.color import ColorLike
from ..common.to_color import to_color

from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(kw_only=True, slots=True)
class Emitter:
    """Emitter dataclass contains lighting-related settings."""

    pass


@dataclass(kw_only=True, slots=True)
class Point(Emitter):
    """Point light source.

    Attributes:
        intensity: Numeric strength, or a legacy color when ``color`` is unset.
        position: Three finite world-space coordinates.
        color: Optional light color; requires numeric ``intensity``.

    """

    intensity: ColorLike | float = 1.0
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    color: ColorLike | None = None

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("Point position must contain three finite values.")
        if self.color is not None:
            to_color(self.color)
            if not isinstance(self.intensity, Real):
                raise TypeError("Point intensity must be numeric when color is set")
        elif not isinstance(self.intensity, Real):
            to_color(self.intensity)
        if isinstance(self.intensity, Real) and (
            not np.isfinite(float(self.intensity)) or float(self.intensity) < 0.0
        ):
            raise ValueError("Point intensity must be finite and non-negative.")


@dataclass(kw_only=True, slots=True)
class Directional(Emitter):
    """Directional light with rays traveling along ``direction``."""

    direction: list[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])
    intensity: float = 1.0
    color: ColorLike = "white"

    def __post_init__(self) -> None:
        direction = np.asarray(self.direction, dtype=np.float64)
        if (
            direction.shape != (3,)
            or not np.all(np.isfinite(direction))
            or np.linalg.norm(direction) <= 1e-12
        ):
            raise ValueError("Directional direction must be a finite non-zero vector.")
        if not isinstance(self.intensity, Real):
            raise TypeError("Directional intensity must be numeric.")
        if not np.isfinite(float(self.intensity)) or float(self.intensity) < 0.0:
            raise ValueError("Directional intensity must be finite and non-negative.")
        to_color(self.color)


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

    def __post_init__(self) -> None:
        up = np.asarray(self.up, dtype=np.float64)
        if (
            up.shape != (3,)
            or not np.all(np.isfinite(up))
            or np.linalg.norm(up) <= 1e-12
        ):
            raise ValueError("Environment up must be a finite non-zero vector.")
        if not isinstance(self.scale, Real):
            raise TypeError("Environment scale must be numeric.")
        if not np.isfinite(float(self.scale)) or float(self.scale) < 0.0:
            raise ValueError("Environment scale must be finite and non-negative.")
