from .texture import generate_texture_config

from ..compiler import View
from ..grammar.channel import (
    Diffuse,
    Conductor,
    RoughConductor,
    Plastic,
    RoughPlastic,
    Principled,
)
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


def generate_float_or_texture_config(tex: float | Texture):
    match tex:
        case float():
            return tex
        case Texture():
            return generate_texture_config(tex)


def generate_rough_conductor_bsdf_config(mat: RoughConductor):
    mi_config: dict[str, Any] = {
        "type": "roughconductor",
        "material": mat.material,
        "distribution": mat.distribution,
        "alpha": generate_float_or_texture_config(mat.alpha),
    }
    return mi_config


def generate_plastic_bsdf_config(mat: Plastic):
    mi_config: dict[str, Any] = {
        "type": "plastic",
        "diffuse_reflectance": generate_float_or_texture_config(
            mat.diffuse_reflectance
        ),
        "specular_reflectance": generate_float_or_texture_config(
            mat.specular_reflectance
        ),
    }
    return mi_config


def generate_rough_plastic_bsdf_config(mat: RoughPlastic):
    mi_config: dict[str, Any] = {
        "type": "roughplastic",
        "diffuse_reflectance": generate_float_or_texture_config(
            mat.diffuse_reflectance
        ),
        "specular_reflectance": generate_float_or_texture_config(
            mat.specular_reflectance
        ),
        "distribution": mat.distribution,
        "alpha": mat.alpha,
    }
    return mi_config


def generate_principled_bsdf_config(mat: Principled):
    mi_config: dict[str, Any] = {
        "type": "principled",
        "base_color": generate_float_or_texture_config(mat.color),
        "roughness": generate_float_or_texture_config(mat.roughness),
        "metallic": generate_float_or_texture_config(mat.metallic),
    }
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
        case RoughPlastic():
            return generate_rough_plastic_bsdf_config(view.material_channel)
        case Plastic():
            return generate_plastic_bsdf_config(view.material_channel)
        case Principled():
            return generate_principled_bsdf_config(view.material_channel)
        case _:
            raise NotImplementedError(
                f"Unknown material type: {type(view.material_channel)}"
            )
