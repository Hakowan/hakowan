from ..setup.emitter import Emitter, Point, Envmap
from .spectrum import generate_spectrum_config
from .utils import rotation

from typing import Any
import numpy.typing as npt
import numpy as np
import mitsuba as mi


def generate_emitter_config(emitter: Emitter) -> dict:
    """Generate a Mitsuba emitter description dict from a Emitter."""

    mi_config: dict[str, Any] = {}

    match emitter:
        case Point():
            mi_config["type"] = "point"
            mi_config["position"] = list(emitter.position)
            mi_config["intensity"] = generate_spectrum_config(emitter.intensity)
        case Envmap():
            mi_config["type"] = "envmap"
            mi_config["filename"] = str(emitter.filename)
            mi_config["scale"] = emitter.scale
            mi_config["to_world"] = mi.ScalarTransform4f(  # type: ignore
                rotation(np.array([0, 1, 0]), np.array(emitter.up))
            ) @ mi.ScalarTransform4f.rotate(  # type: ignore
                [0, 1, 0],
                emitter.rotation,
            )
        case _:
            raise NotImplementedError(f"Unknown emitter type: {type(emitter)}")

    return mi_config
