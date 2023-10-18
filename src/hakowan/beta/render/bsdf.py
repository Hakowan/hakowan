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
import lagrange


def generate_float_or_texture_config(mesh: lagrange.SurfaceMesh, tex: float | Texture, is_color: bool = False):
    match tex:
        case float():
            return tex
        case Texture():
            return generate_texture_config(mesh, tex, is_color)


def generate_diffuse_bsdf_config(mesh: lagrange.SurfaceMesh, mat: Diffuse):
    mi_config = {
        "type": "diffuse",
        "reflectance": generate_float_or_texture_config(mesh, mat.reflectance, True),
    }
    return mi_config


def generate_conductor_bsdf_config(mesh: lagrange.SurfaceMesh, mat: Conductor):
    mi_config = {
        "type": "conductor",
        "material": mat.material,
    }
    return mi_config


def generate_rough_conductor_bsdf_config(mesh: lagrange.SurfaceMesh, mat: RoughConductor):
    mi_config: dict[str, Any] = {
        "type": "roughconductor",
        "material": mat.material,
        "distribution": mat.distribution,
        "alpha": generate_float_or_texture_config(mesh, mat.alpha),
    }
    return mi_config


def generate_plastic_bsdf_config(mesh: lagrange.SurfaceMesh, mat: Plastic):
    mi_config: dict[str, Any] = {
        "type": "plastic",
        "diffuse_reflectance": generate_float_or_texture_config(
            mesh, mat.diffuse_reflectance, True
        ),
        "specular_reflectance": generate_float_or_texture_config(
            mesh, mat.specular_reflectance
        ),
    }
    return mi_config


def generate_rough_plastic_bsdf_config(mesh: lagrange.SurfaceMesh, mat: RoughPlastic):
    mi_config: dict[str, Any] = {
        "type": "roughplastic",
        "diffuse_reflectance": generate_float_or_texture_config(
            mesh, mat.diffuse_reflectance, True
        ),
        "specular_reflectance": generate_float_or_texture_config(
            mesh, mat.specular_reflectance
        ),
        "distribution": mat.distribution,
        "alpha": mat.alpha,
    }
    return mi_config


def generate_principled_bsdf_config(mesh: lagrange.SurfaceMesh, mat: Principled):
    mi_config: dict[str, Any] = {
        "type": "principled",
        "base_color": generate_float_or_texture_config(mesh, mat.color, True),
        "roughness": generate_float_or_texture_config(mesh, mat.roughness),
        "metallic": generate_float_or_texture_config(mesh, mat.metallic),
    }
    return mi_config


def generate_bsdf_config(view: View):
    assert view.data_frame is not None
    assert view.material_channel is not None
    mesh = view.data_frame.mesh
    match view.material_channel:
        case Diffuse():
            return generate_diffuse_bsdf_config(mesh, view.material_channel)
        case RoughConductor():
            return generate_rough_conductor_bsdf_config(mesh, view.material_channel)
        case Conductor():
            return generate_conductor_bsdf_config(mesh, view.material_channel)
        case RoughPlastic():
            return generate_rough_plastic_bsdf_config(mesh, view.material_channel)
        case Plastic():
            return generate_plastic_bsdf_config(mesh, view.material_channel)
        case Principled():
            return generate_principled_bsdf_config(mesh, view.material_channel)
        case _:
            raise NotImplementedError(
                f"Unknown material type: {type(view.material_channel)}"
            )
