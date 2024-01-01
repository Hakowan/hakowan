from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt


@dataclass(kw_only=True, slots=True)
class Sensor:
    location: npt.NDArray = field(default_factory=lambda: np.array([0, 0, 5]))
    target: npt.NDArray = field(default_factory=lambda: np.array([0, 0, 0]))
    up: npt.NDArray = field(default_factory=lambda: np.array([0, 1, 0]))
    near_clip: float = 1e-2
    far_clip: float = 1e4


@dataclass(kw_only=True, slots=True)
class Perspective(Sensor):
    fov: float = 28.8415  # degrees
    fov_axis: str = "smaller"


@dataclass(kw_only=True, slots=True)
class Orthographic(Sensor):
    pass


@dataclass(kw_only=True, slots=True)
class ThinLens(Perspective):
    aperture_radius: float = 0.1
    focus_distance: float = 0.0
