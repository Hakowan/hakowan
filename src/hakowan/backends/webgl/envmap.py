"""Encode a hakowan Envmap emitter for the viewer HTML.

We base64-embed the source file (typically ``.exr``) and emit a small JSON
descriptor that the viewer JS reads. Three.js loads it with ``EXRLoader`` or
``RGBELoader`` depending on the file extension, sets it as
``scene.environment`` (for PBR reflections) and optionally as
``scene.background``.

The envmap orientation matches Mitsuba's convention:
``to_world = align_y_to(up) @ rotate_y(rotation_deg)``. We bake the composed
3x3 matrix into the descriptor so the viewer can apply it via
``scene.environmentRotation`` without re-implementing the maths.
"""

from __future__ import annotations

import base64
import math
from pathlib import Path
from typing import Any

import numpy as np

from ...common import logger
from ...setup import Config
from ...setup.emitter import Envmap


def envmap_descriptor(config: Config) -> dict[str, Any] | None:
    """Pick the first ``Envmap`` emitter in ``config`` and encode it.

    Returns a JSON-serialisable dict like::

        {
            "format": "exr" | "hdr",
            "uri": "data:application/octet-stream;base64,...",
            "scale": 1.0,
            "rotation": 180.0,
            "background": True,
        }

    or ``None`` when no Envmap is present (the viewer falls back to its
    built-in 3-point lighting).
    """
    envmap: Envmap | None = None
    for emitter in config.emitters:
        if isinstance(emitter, Envmap):
            envmap = emitter
            break
    if envmap is None:
        return None

    path = Path(envmap.filename)
    if not path.is_file():
        logger.warning(
            f"WebGL backend: envmap file '{path}' not found; "
            "falling back to default lighting."
        )
        return None

    suffix = path.suffix.lower()
    if suffix == ".exr":
        fmt = "exr"
    elif suffix in (".hdr", ".rgbe"):
        fmt = "hdr"
    else:
        logger.warning(
            f"WebGL backend: envmap format '{suffix}' not supported "
            "(use .exr or .hdr); falling back to default lighting."
        )
        return None

    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    rotation_matrix = _build_rotation_matrix(envmap.rotation, envmap.up)
    return {
        "format": fmt,
        "uri": f"data:application/octet-stream;base64,{b64}",
        "scale": float(envmap.scale),
        # Row-major 3x3 — the viewer reads it into a Matrix4 and converts to
        # an Euler for scene.environmentRotation.
        "rotation_matrix": rotation_matrix.flatten().tolist(),
        "background": False,
    }


def _rotate_y(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _align_y_to(up: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping +Y onto the given (unit) ``up`` vector.

    Mirrors ``backends/mitsuba/utils.rotation`` (Rodrigues' formula), but
    short-circuits the parallel / anti-parallel degenerate cases.
    """
    y = np.array([0.0, 1.0, 0.0])
    n = float(np.linalg.norm(up))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    u = up / n
    cos_a = float(np.dot(y, u))
    if cos_a > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)
    if cos_a < -1.0 + 1e-9:
        # Y and up are anti-parallel — rotate 180° around any axis ⟂ Y.
        return np.diag([1.0, -1.0, -1.0]).astype(np.float64)
    axis = np.cross(y, u)
    sin_a = float(np.linalg.norm(axis))
    axis = axis / sin_a
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + sin_a * K + (1.0 - cos_a) * (K @ K)


def _build_rotation_matrix(rotation_deg: float, up: list[float]) -> np.ndarray:
    """Compose ``align_y_to(up) @ rotate_y(rotation_deg)`` — same as Mitsuba's
    ``Envmap.to_world``.
    """
    up_arr = np.asarray(up, dtype=np.float64)
    return _align_y_to(up_arr) @ _rotate_y(math.radians(float(rotation_deg)))
