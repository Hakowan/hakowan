"""Small helpers shared across the WebGL backend."""

from __future__ import annotations

import base64
import numpy as np
import numpy.typing as npt


def np_to_bytes(arr: np.ndarray) -> bytes:
    """Return contiguous little-endian bytes for a numpy array."""
    return np.ascontiguousarray(arr).tobytes(order="C")


def glb_to_data_uri(glb_bytes: bytes) -> str:
    """Encode GLB bytes as a base64 data URI suitable for GLTFLoader.parse."""
    b64 = base64.b64encode(glb_bytes).decode("ascii")
    return f"data:model/gltf-binary;base64,{b64}"


def look_at(eye: npt.ArrayLike, target: npt.ArrayLike, up: npt.ArrayLike) -> np.ndarray:
    """Return the camera's world transform (4x4) for an eye/target/up triple.

    Uses the glTF/three.js convention: the camera looks down its local -Z axis,
    with +Y up. The returned matrix is the inverse of the view matrix.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-12:
        forward = np.array([0.0, 0.0, -1.0])
    else:
        forward = forward / forward_norm
    # Camera local -Z points toward target, so local +Z points away from target.
    z_axis = -forward

    x_axis = np.cross(up, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-12:
        # up parallel to forward; pick an arbitrary perpendicular.
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(z_axis @ fallback) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(fallback, z_axis)
        x_norm = np.linalg.norm(x_axis)
    x_axis = x_axis / x_norm

    y_axis = np.cross(z_axis, x_axis)

    m = np.eye(4, dtype=np.float64)
    m[:3, 0] = x_axis
    m[:3, 1] = y_axis
    m[:3, 2] = z_axis
    m[:3, 3] = eye
    return m


def gltf_matrix(m_4x4: np.ndarray) -> list[float]:
    """Flatten a 4x4 row-major numpy matrix into glTF column-major list."""
    arr = np.asarray(m_4x4, dtype=np.float64).reshape(4, 4)
    return arr.T.flatten().tolist()
