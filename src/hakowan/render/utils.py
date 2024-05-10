import numpy as np
import numpy.typing as npt


def rotation(from_vector: npt.NDArray, to_vector: npt.NDArray):
    axis = np.cross(from_vector, to_vector)
    sin_a = np.linalg.norm(axis)
    cos_a = np.dot(from_vector, to_vector)
    if sin_a < 1e-9:
        return np.eye(4)
    else:
        v = np.array(axis / sin_a, dtype=np.float64)
        I = np.eye(3)
        H = np.outer(v, v)
        S = np.cross(I, v)
        M = I * cos_a + S * sin_a + H * (1 - cos_a)

        A = np.eye(4, dtype=np.float64)
        A[:3, :3] = M
        return A
