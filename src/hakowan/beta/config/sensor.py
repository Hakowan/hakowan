from dataclasses import dataclass, field
import numpy.typing as npt


@dataclass(kw_only=True)
class Sensor:
    location: npt.NDarray = field(default_factory=lambda: np.array([0, 0, 5]))
    target: npt.NDArray = field(default_factory=lambda: np.array([0, 0, 0]))
    up: npt.NDArray = field(default_factory=lambda: np.array([0, 1, 0]))
    near_clip: float = 1e-2
    far_clip: float = 1e4


@dataclass(kw_only=True)
class Perspective(Sensor):
    fov: float = 28.8415  # degrees
    fov_axis: str = "x"


@dataclass(kw_only=True)
class Orthographic(Sensor):
    pass


@dataclass(kw_only=True)
class ThinLens(Perspective):
    aparture_radius: float = 0.1
    focus_distance: float = 0.0
