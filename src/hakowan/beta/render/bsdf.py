from .texture import generate_texture_config

from ..compiler import View
from ..grammar.channel import Diffuse, Conductor, RoughConductor
from ..grammar.texture import Texture

from typing import Any


def generate_diffuse_bsdf_config(mat: Diffuse):
    mi_config = {
        "type": "diffuse",
        "reflectance": generate_texture_config(mat.reflectance),
    }
    return mi_config


def generate_conductor_bsdf_config(mat: Conductor):
    mi_config = {
        "type": "conductor",
        "material": mat.material,
    }
    return mi_config


def generate_rough_conductor_bsdf_config(mat: RoughConductor):
    mi_config: dict[str, Any] = {
        "type": "roughconductor",
        "material": mat.material,
        "distribution": mat.distribution,
    }
    match mat.alpha:
        case float():
            mi_config["alpha"] = mat.alpha
        case Texture():
            mi_config["alpha"] = (generate_texture_config(mat.alpha),)
    return mi_config


def generate_bsdf_config(view: View):
    assert view.material_channel is not None
    match view.material_channel:
        case Diffuse():
            return generate_diffuse_bsdf_config(view.material_channel)
        case RoughConductor():
            return generate_rough_conductor_bsdf_config(view.material_channel)
        case Conductor():
            return generate_conductor_bsdf_config(view.material_channel)
        case _:
            raise NotImplementedError(
                f"Unknown material type: {type(view.material_channel)}"
            )
