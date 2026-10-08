"""Camera matrix and coordinate-frame helpers for the CLI."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..setup import Config


def compute_camera_matrix(config: Config) -> dict[str, Any]:
    """Compute the world-to-camera view matrix and camera intrinsics.

    Convention:
      - Camera space: right-handed, camera looks down **-Z**, Y is up.
      - Pixel space: (0, 0) at top-left, u increases right, v increases down.
      - Back-projection of pixel (u, v) to a camera-space ray direction:
            d_cam = normalize([(u - cx)/f, -(v - cy)/f, -1])
        then rotate to world space with the 3x3 upper-left block of view_matrix^-1
        (= view_matrix[:3,:3].T).

    Args:
        config: hakowan Config whose sensor/film are already fully configured.

    Returns:
        dict with keys:
          view_matrix      – 4x4 float64 world-to-camera transform
          intrinsic_matrix – 3x3 float64 K matrix (fx=fy, cx, cy)
          camera_location  – (3,) world-space eye position
          camera_target    – (3,) world-space look-at point
          camera_up        – (3,) orthogonalised up vector in world space
          fov_deg          – scalar FOV applied to the shorter image dimension
          width, height    – image dimensions (int32)
    """
    sensor = config.sensor
    eye = np.asarray(sensor.location, dtype=np.float64)
    target = np.asarray(getattr(sensor, "target", [0.0, 0.0, 0.0]), dtype=np.float64)
    up_hint = np.asarray(getattr(sensor, "up", [0.0, 1.0, 0.0]), dtype=np.float64)

    # Orthonormal camera basis
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)  # guaranteed orthogonal to both

    # 4x4 view matrix  (world → camera, camera looks down -Z)
    # [ right  | -dot(right, eye) ]
    # [ up     | -dot(up, eye)    ]
    # [ -fwd   |  dot(fwd, eye)   ]
    # [ 0 0 0  |  1               ]
    view = np.array(
        [
            [right[0], right[1], right[2], -np.dot(right, eye)],
            [up[0], up[1], up[2], -np.dot(up, eye)],
            [-fwd[0], -fwd[1], -fwd[2], np.dot(fwd, eye)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Intrinsic matrix K  (FOV applies to the shorter image dimension)
    width = config.film.width
    height = config.film.height
    fov_deg = float(getattr(sensor, "fov", 28.8415))
    f = (min(width, height) / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    return {
        "view_matrix": view,
        "intrinsic_matrix": K,
        "camera_location": eye,
        "camera_target": target,
        "camera_up": up,
        "fov_deg": np.float64(fov_deg),
        "width": np.int32(width),
        "height": np.int32(height),
    }


def _orient_pca_target_frame(z_up: bool) -> npt.NDArray[np.float64]:
    """Columns: world directions for (major, mid, minor) PCA axes (right-handed)."""
    if z_up:
        # major -> +Z, mid -> +X, minor -> +Y
        return np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64
        )
    # major -> +Y, mid -> +Z, minor -> +X (Y-up)
    return np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )


def save_camera_matrix(data: dict[str, Any], output_path: Path) -> None:
    """Save camera matrix data produced by compute_camera_matrix.

    Supported formats (determined by file extension):
      .npz  – NumPy compressed archive (default / recommended)
      .json – JSON for interoperability with non-Python tools
    """
    suffix = output_path.suffix.lower()
    if suffix == ".json":

        def _to_python(v: Any) -> Any:
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            return v

        output_path.write_text(
            json.dumps({k: _to_python(v) for k, v in data.items()}, indent=2)
        )
    else:
        # Ensure the file has the .npz extension that numpy requires.
        out = (
            output_path
            if output_path.suffix.lower() == ".npz"
            else output_path.with_suffix(".npz")
        )
        np.savez(str(out), **data)
