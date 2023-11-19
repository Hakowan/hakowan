from .color import generate_color_config
from ..compiler import View
from ..grammar.channel.material import Dielectric

from typing import Any
from numpy.linalg import norm


def generate_medium_config(view: View) -> dict[str, Any]:
    assert view.material_channel is not None
    assert isinstance(view.material_channel, Dielectric)
    assert view.material_channel.medium is not None
    albedo = view.material_channel.medium.albedo
    match albedo:
        case float() | int():
            albedo = float(albedo)
        case str() | list() | tuple():
            albedo = generate_color_config(albedo)
        case _:
            raise NotImplementedError(f"Unsupported albedo type: {type(albedo)}")

    assert view.bbox is not None
    # TODO: this is just a heuristic.
    bbox_diag = 5 * norm(view.bbox[0] - view.bbox[1])
    return {
        "type": "homogeneous",
        "albedo": albedo,
        "scale": bbox_diag,
    }
