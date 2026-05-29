"""Translate ``Config.sensor`` to a glTF camera node."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from ...common import logger
from ...setup import Config
from ...setup.sensor import Orthographic, Perspective, ThinLens

from .builder import GLTFBuilder
from .utils import look_at


def _aspect_ratio(config: Config) -> float:
    return float(config.film.width) / float(config.film.height)


def _yfov_radians(sensor: Perspective, aspect: float) -> float:
    """Convert hakowan's fov + fov_axis to glTF's y-axis fov (radians)."""
    fov_rad = math.radians(sensor.fov)
    half = fov_rad / 2.0

    axis: str = sensor.fov_axis
    if axis == "y":
        return fov_rad
    if axis == "x":
        return 2.0 * math.atan(math.tan(half) / aspect)
    if axis == "diagonal":
        # Treat fov as the diagonal extent and split per-axis.
        diag_half_tan = math.tan(half)
        # diag^2 = x^2 + y^2 and aspect = x/y → y_tan = diag/sqrt(1 + a^2)
        y_tan = diag_half_tan / math.sqrt(1.0 + aspect * aspect)
        return 2.0 * math.atan(y_tan)
    if axis == "smaller":
        if aspect >= 1.0:
            # height is the smaller side
            return fov_rad
        return 2.0 * math.atan(math.tan(half) / aspect)
    if axis == "larger":
        if aspect >= 1.0:
            return 2.0 * math.atan(math.tan(half) / aspect)
        return fov_rad
    logger.warning(f"WebGL backend: unknown fov_axis '{axis}'; treating as 'y'.")
    return fov_rad


def add_camera(
    builder: GLTFBuilder, config: Config
) -> tuple[int, dict[str, list[float]]]:
    """Register a camera node and return ``(node_index, initial_view_dict)``.

    ``initial_view_dict`` carries the eye/target/up vectors that the HTML
    viewer uses to position the OrbitControls camera at load time, since
    glTF doesn't standardise an OrbitControls target.
    """
    sensor = config.sensor
    eye = np.asarray(sensor.location, dtype=np.float64).reshape(3)
    target = np.asarray(sensor.target, dtype=np.float64).reshape(3)
    up = np.asarray(sensor.up, dtype=np.float64).reshape(3)
    world_matrix = look_at(eye, target, up)
    aspect = _aspect_ratio(config)

    if isinstance(sensor, Orthographic):
        # Approximate ortho extents from the sensor->target distance and a
        # nominal 1.0 world-unit height (the global transform fits everything
        # into a unit sphere, so this is reasonable).
        ymag = 1.0
        xmag = ymag * aspect
        node_idx = builder.add_orthographic_camera(
            xmag=xmag,
            ymag=ymag,
            znear=float(sensor.near_clip),
            zfar=float(sensor.far_clip),
            world_transform_4x4=world_matrix,
        )
    else:
        if isinstance(sensor, ThinLens):
            logger.warning(
                "WebGL backend: ThinLens depth-of-field not supported; "
                "rendering as standard perspective."
            )
        perspective = sensor if isinstance(sensor, Perspective) else Perspective()
        yfov = _yfov_radians(perspective, aspect)
        zfar = float(perspective.far_clip)
        node_idx = builder.add_perspective_camera(
            yfov=yfov,
            aspect_ratio=aspect,
            znear=float(perspective.near_clip),
            zfar=zfar,
            world_transform_4x4=world_matrix,
        )

    initial_view = {
        "eye": eye.tolist(),
        "target": target.tolist(),
        "up": up.tolist(),
    }
    return node_idx, initial_view
