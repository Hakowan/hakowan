from dataclasses import dataclass, field


@dataclass(kw_only=True, slots=True)
class Sensor:
    """Sensor dataclass contains camera-related settings.

    Attributes:
        location: Camera location in world space.
        target: Camera look-at location in world space.
        up: Camera up vector in world space.
        near_clip: Near clipping plane distance.
        far_clip: Far clipping plane distance.
    """

    location: list = field(default_factory=lambda: [0, 0, 5])
    target: list = field(default_factory=lambda: [0, 0, 0])
    up: list = field(default_factory=lambda: [0, 1, 0])
    near_clip: float = 1e-2
    far_clip: float = 1e4


@dataclass(kw_only=True, slots=True)
class Perspective(Sensor):
    """Perspective camera dataclass.

    Attributes:
        fov: Field of view in degrees.
        fov_axis: Axis to which fov is applied. Can be "x" or "y" or "diagonal" or "smaller" or "larger".
    """

    fov: float = 28.8415  # degrees
    fov_axis: str = "smaller"


@dataclass(kw_only=True, slots=True)
class Orthographic(Sensor):
    """Orthographic camera dataclass."""

    pass


@dataclass(kw_only=True, slots=True)
class ThinLens(Perspective):
    """Thin lens camera dataclass.

    Attributes:
        aperture_radius: Radius of the aperture in world space.
        focus_distance: Distance to the focal plane in world space.
    """

    aperture_radius: float = 0.1
    focus_distance: float = 0.0
