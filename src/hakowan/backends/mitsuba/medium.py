from .color import generate_color_config
from ...compiler import View
from ...grammar.channel.material import Dielectric

from typing import Any
from numpy.linalg import norm


def generate_medium_config(view: View) -> dict[str, Any]:
    assert view.material_channel is not None
    assert isinstance(view.material_channel, Dielectric)
    assert view.material_channel.medium is not None
    albedo = view.material_channel.medium.albedo
    scale = view.material_channel.medium.scale
    albedo_config: Any
    match albedo:
        case float() | int():
            albedo_config = float(albedo)
        case str() | list() | tuple():
            albedo_config = generate_color_config(albedo)
        case _:
            raise NotImplementedError(f"Unsupported albedo type: {type(albedo)}")

    assert view.bbox is not None
    bbox_diag = scale * norm(view.bbox[0] - view.bbox[1])
    return {
        "type": "homogeneous",
        "albedo": albedo_config,
        "scale": bbox_diag,
    }
