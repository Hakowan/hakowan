from ..config.emitter import Emitter, Point, Envmap
from .spectrum import generate_spectrum_config

from typing import Any


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
        case _:
            raise NotImplementedError(f"Unknown emitter type: {type(emitter)}")

    return mi_config
