import numpy as np
import numpy.typing as npt


def filter_zero_length_vectors(
    base: npt.NDArray,
    tip: npt.NDArray,
    base_size,
    tip_size,
    ctrl_pts_1: npt.NDArray | None,
    ctrl_pts_2: npt.NDArray | None,
):
    """Drop vector-field arrows whose tip coincides with their base.

    A zero-length vector has no well-defined direction, so it cannot be
    rendered as an arrow/segment. Returns all inputs filtered in order;
    ``ctrl_pts_1``/``ctrl_pts_2`` may be ``None`` and pass through unchanged.
    """
    mask = np.linalg.norm(tip - base, axis=1) > 0
    if np.all(mask):
        return base, tip, base_size, tip_size, ctrl_pts_1, ctrl_pts_2
    return (
        base[mask],
        tip[mask],
        np.asarray(base_size)[mask],
        np.asarray(tip_size)[mask],
        None if ctrl_pts_1 is None else ctrl_pts_1[mask],
        None if ctrl_pts_2 is None else ctrl_pts_2[mask],
    )
