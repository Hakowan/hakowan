from ..common.color import Color

from typing import Any


def generate_spectrum_config(spectrum: Color | float) -> dict:
    """Generate a Mitsuba spectrum description dict from a Spectrum."""
    mi_config: dict[str, Any] = {}
    match spectrum:
        case float():
            mi_config["type"] = "rgb"
            mi_config["value"] = [spectrum] * 3
        case Color():
            mi_config["type"] = "rgb"
            mi_config["value"] = spectrum.data.tolist()
        case _:
            raise NotImplementedError(f"Unknown spectrum type: {type(spectrum)}")

    return mi_config
