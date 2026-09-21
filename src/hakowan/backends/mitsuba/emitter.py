from ...setup.emitter import Directional, Emitter, Envmap, Point
from .spectrum import generate_spectrum_config
from ...common.to_color import to_color
from .utils import rotation

from typing import Any
import numpy as np
import mitsuba as mi


def generate_emitter_config(emitter: Emitter) -> dict:
    """Generate a Mitsuba emitter description dict from a Emitter."""

    mi_config: dict[str, Any] = {}

    match emitter:
        case Point():
            mi_config["type"] = "point"
            mi_config["position"] = list(emitter.position)
            if emitter.color is None:
                intensity = (
                    float(emitter.intensity)
                    if isinstance(emitter.intensity, (int, float))
                    else to_color(emitter.intensity)
                )
            else:
                if not isinstance(emitter.intensity, (int, float)):
                    raise TypeError("Point intensity must be numeric when color is set")
                intensity = to_color(emitter.color) * float(emitter.intensity)
            mi_config["intensity"] = generate_spectrum_config(intensity)
        case Directional():
            mi_config["type"] = "directional"
            mi_config["direction"] = list(emitter.direction)
            irradiance = to_color(emitter.color) * emitter.intensity
            mi_config["irradiance"] = generate_spectrum_config(irradiance)
        case Envmap():
            mi_config["type"] = "envmap"
            mi_config["filename"] = str(emitter.filename)
            mi_config["scale"] = emitter.scale
            mi_config["to_world"] = mi.ScalarTransform4f(  # type: ignore
                rotation(np.array([0, 1, 0]), np.array(emitter.up))
            ) @ mi.ScalarTransform4f().rotate(  # type: ignore
                [0, 1, 0],
                emitter.rotation,
            )
        case _:
            raise NotImplementedError(f"Unknown emitter type: {type(emitter)}")

    return mi_config
