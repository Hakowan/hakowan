import numpy as np
import numpy.typing as npt


def rotation(from_vector: npt.NDArray, to_vector: npt.NDArray):
    """Rotation matrix that aligns ``from_vector`` onto ``to_vector``.

    Both arguments are treated as directions and normalized internally, so
    callers may pass un-normalized vectors (e.g. point-cloud normals imported
    from a PCD file, which are frequently not unit length). Skipping this
    normalization turns the Rodrigues terms below into a huge scale/shear
    instead of a rotation, producing giant degenerate shapes that make
    Mitsuba's BVH pathologically slow (effectively a hang).
    """
    from_vector = np.asarray(from_vector, dtype=np.float64)
    to_vector = np.asarray(to_vector, dtype=np.float64)
    from_norm = np.linalg.norm(from_vector)
    to_norm = np.linalg.norm(to_vector)
    if from_norm < 1e-9 or to_norm < 1e-9:
        return np.eye(4)
    from_vector = from_vector / from_norm
    to_vector = to_vector / to_norm

    axis = np.cross(from_vector, to_vector)
    sin_a = np.linalg.norm(axis)
    cos_a = np.dot(from_vector, to_vector)
    eye3 = np.eye(3)
    if sin_a < 1e-9:
        if cos_a > 0:
            # Parallel: no rotation needed.
            return np.eye(4)
        # Antiparallel: rotate 180 degrees about any axis perpendicular to
        # ``from_vector`` so the shape faces the opposite direction.
        perp = np.cross(from_vector, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(from_vector, np.array([0.0, 1.0, 0.0]))
        v = perp / np.linalg.norm(perp)
        A = np.eye(4, dtype=np.float64)
        A[:3, :3] = -eye3 + 2.0 * np.outer(v, v)
        return A

    v = np.array(axis / sin_a, dtype=np.float64)
    H = np.outer(v, v)
    S = np.cross(eye3, v)
    M = eye3 * cos_a + S * sin_a + H * (1 - cos_a)

    A = np.eye(4, dtype=np.float64)
    A[:3, :3] = M
    return A
