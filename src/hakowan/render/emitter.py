from ..setup.emitter import Emitter, Point, Envmap
from .spectrum import generate_spectrum_config

from typing import Any
import numpy.typing as npt
import numpy as np
import mitsuba as mi


def rotation(from_vector: npt.NDArray, to_vector: npt.NDArray):
    axis = np.cross(from_vector, to_vector)
    angle = np.degrees(np.arccos(np.dot(from_vector, to_vector)))
    return mi.ScalarTransform4f.rotate(axis, angle)  # type: ignore


def generate_emitter_config(emitter: Emitter) -> dict:
    """Generate a Mitsuba emitter description dict from a Emitter."""

    mi_config: dict[str, Any] = {}

    match emitter:
        case Point():
            mi_config["type"] = "point"
            mi_config["position"] = emitter.position.tolist()
            mi_config["intensity"] = generate_spectrum_config(emitter.intensity)
        case Envmap():
            mi_config["type"] = "envmap"
            mi_config["filename"] = str(emitter.filename)
            mi_config["scale"] = emitter.scale
            mi_config["to_world"] = rotation(
                np.array([0, 1, 0]), emitter.up
            ) @ mi.ScalarTransform4f.rotate( # type: ignore
                [0, 1, 0],
                emitter.rotation,
            )
        case _:
            raise NotImplementedError(f"Unknown emitter type: {type(emitter)}")

    return mi_config
