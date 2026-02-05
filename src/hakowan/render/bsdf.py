from .texture import generate_texture_config
from .color import generate_color_config

from ..compiler import View
from ..grammar.channel.material import (
    Conductor,
    Dielectric,
    Diffuse,
    Hair,
    Material,
    Plastic,
    Principled,
    RoughConductor,
    RoughDielectric,
    RoughPlastic,
    ThinDielectric,
    ThinPrincipled,
)
from ..grammar.channel import BumpMap, NormalMap
from ..grammar.texture import Texture
from ..common.color import ColorLike

from typing import Any
import lagrange


def generate_float_color_texture_config(
    mesh: lagrange.SurfaceMesh,
    tex: ColorLike | Texture,
    is_color: bool = False,
    is_primitive: bool = False,
):
    match tex:
        case float() | int():
            return float(tex)
        case str() | list() | tuple():
            assert is_color
            return generate_color_config(tex)
        case Texture():
            return generate_texture_config(mesh, tex, is_color, is_primitive)
        case _:
            raise NotImplementedError(f"Unsupported type: {type(tex)}")


def generate_diffuse_bsdf_config(
    mesh: lagrange.SurfaceMesh, mat: Diffuse, is_primitive
):
    reflectance = generate_float_color_texture_config(
        mesh, mat.reflectance, True, is_primitive
    )
    mi_config: dict[str, Any]
    if is_primitive and "colors" in reflectance:
        mi_config = {
            f"bsdf_{i:06}": {
                "type": "diffuse",
                "reflectance": generate_color_config(color),
            }
            for i, color in enumerate(reflectance["colors"])
        }
    else:
        mi_config = {"type": "diffuse", "reflectance": reflectance}
    return mi_config


def generate_conductor_bsdf_config(mesh: lagrange.SurfaceMesh, mat: Conductor):
    mi_config = {
        "type": "conductor",
        "material": mat.material,
    }
    return mi_config


def generate_rough_conductor_bsdf_config(
    mesh: lagrange.SurfaceMesh, mat: RoughConductor
):
    mi_config: dict[str, Any] = {
        "type": "roughconductor",
        "material": mat.material,
        "distribution": mat.distribution,
        "alpha": generate_float_color_texture_config(mesh, mat.alpha),
    }
    return mi_config


def generate_plastic_bsdf_config(mesh: lagrange.SurfaceMesh, mat: Plastic):
    mi_config: dict[str, Any] = {
        "type": "plastic",
        "diffuse_reflectance": generate_float_color_texture_config(
            mesh, mat.diffuse_reflectance, True
        ),
        "specular_reflectance": generate_float_color_texture_config(
            mesh, mat.specular_reflectance
        ),
    }
    return mi_config


def generate_rough_plastic_bsdf_config(mesh: lagrange.SurfaceMesh, mat: RoughPlastic):
    mi_config: dict[str, Any] = {
        "type": "roughplastic",
        "diffuse_reflectance": generate_float_color_texture_config(
            mesh, mat.diffuse_reflectance, True
        ),
        "specular_reflectance": generate_float_color_texture_config(
            mesh, mat.specular_reflectance
        ),
        "distribution": mat.distribution,
        "alpha": mat.alpha,
    }
    return mi_config


def generate_principled_bsdf_config(
    mesh: lagrange.SurfaceMesh, mat: Principled, is_primitive: bool, thin: bool = False
):
    # Extract color, roughness, metallic
    colors = generate_float_color_texture_config(mesh, mat.color, True, is_primitive)
    roughness = generate_float_color_texture_config(
        mesh, mat.roughness, False, is_primitive
    )
    metallic = generate_float_color_texture_config(
        mesh, mat.metallic, False, is_primitive
    )

    n: int | None = None

    # Check size and generate getters
    if isinstance(colors, dict) and "colors" in colors:
        n = len(colors["colors"])
        get_color = lambda i: generate_color_config(colors["colors"][i])
    else:
        get_color = lambda i: colors

    if isinstance(roughness, dict) and "values" in roughness:
        if n is None:
            n = len(roughness["values"])
        else:
            assert n == len(roughness["values"])
        get_roughness = lambda i: roughness["values"][i]
    else:
        get_roughness = lambda i: roughness

    if isinstance(metallic, dict) and "values" in metallic:
        if n is None:
            n = len(metallic["values"])
        else:
            assert n == len(metallic["values"])
        get_metallic = lambda i: metallic["values"][i]
    else:
        get_metallic = lambda i: metallic

    mat_name = "principled" if not thin else "principledthin"
    base_config: dict[str, Any] = {
        "anisotropic": mat.anisotropic,
        "spec_trans": mat.spec_trans,
        "eta": mat.eta,
        "spec_tint": mat.spec_tint,
        "sheen": mat.sheen,
        "sheen_tint": mat.sheen_tint,
        "flatness": generate_color_config(mat.flatness),
    }
    if thin:
        assert isinstance(mat, ThinPrincipled)
        base_config["diff_trans"] = mat.diff_trans
    if n is None:
        mi_config: dict[str, Any] = {
            "type": mat_name,
            "base_color": generate_float_color_texture_config(mesh, mat.color, True),
            "roughness": generate_float_color_texture_config(mesh, mat.roughness),
            "metallic": generate_float_color_texture_config(mesh, mat.metallic),
        } | base_config
    else:
        mi_config = {
            f"bsdf_{i:06}": {
                "type": mat_name,
                "base_color": get_color(i),
                "roughness": get_roughness(i),
                "metallic": get_metallic(i),
            }
            | base_config
            for i in range(n)
        }
    return mi_config


def generate_dielectric_bsdf_config(mesh: lagrange.SurfaceMesh, mat: Dielectric):
    return {
        "type": "dielectric",
        "int_ior": mat.int_ior,
        "ext_ior": mat.ext_ior,
        "specular_reflectance": mat.specular_reflectance,
        "specular_transmittance": mat.specular_transmittance,
    }


def generate_thin_dielectric_bsdf_config(mesh: lagrange.SurfaceMesh, mat: Dielectric):
    return {
        "type": "thindielectric",
        "int_ior": mat.int_ior,
        "ext_ior": mat.ext_ior,
        "specular_reflectance": mat.specular_reflectance,
    }


def generate_rough_dielectric_bsdf_config(
    mesh: lagrange.SurfaceMesh, mat: RoughDielectric
):
    mi_config: dict[str, Any] = {
        "type": "roughdielectric",
        "int_ior": mat.int_ior,
        "ext_ior": mat.ext_ior,
        "distribution": mat.distribution,
        "alpha": generate_float_color_texture_config(mesh, mat.alpha),
        "specular_reflectance": mat.specular_reflectance,
        "specular_transmittance": mat.specular_transmittance,
    }
    return mi_config


def generate_hair_bsdf_config(mesh: lagrange.SurfaceMesh, mat: Hair):
    mi_config: dict[str, Any] = {
        "type": "hair",
        "eumelanin": mat.eumelanin,
        "pheomelanin": mat.pheomelanin,
        "longitudinal_roughness": 0.05,
        "azimuthal_roughness": 0.3,
    }
    return mi_config


def make_material_two_sided(mi_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "twosided",
        "material": mi_config,
    }


def add_bump_map(
    mi_config: dict[str, Any],
    mesh: lagrange.SurfaceMesh,
    bump_map: BumpMap,
    is_primitive: bool,
) -> dict[str, Any]:
    assert "type" in mi_config, "Bump map can only be applied over a single BSDF"
    return {
        "type": "bumpmap",
        "bump_texture": generate_float_color_texture_config(mesh, bump_map.texture),
        "scale": bump_map.scale,
        "bsdf": mi_config,
    }


def add_normal_map(
    mi_config: dict[str, Any],
    mesh: lagrange.SurfaceMesh,
    normal_map: NormalMap,
    is_primitive: bool,
) -> dict[str, Any]:
    assert "type" in mi_config, "Normal map can only be applied over a single BSDF"
    return {
        "type": "normalmap",
        "normalmap": generate_float_color_texture_config(mesh, normal_map.texture),
        "bsdf": mi_config,
    }


def generate_bsdf_config(view: View, is_primitive=False) -> dict[str, Any]:
    assert view.data_frame is not None
    assert view.material_channel is not None
    mesh = view.data_frame.mesh
    material_config: dict[str, Any] = {}
    match view.material_channel:
        case Diffuse():
            material_config = generate_diffuse_bsdf_config(
                mesh, view.material_channel, is_primitive
            )
        case RoughConductor():
            material_config = generate_rough_conductor_bsdf_config(
                mesh, view.material_channel
            )
        case Conductor():
            material_config = generate_conductor_bsdf_config(
                mesh, view.material_channel
            )
        case RoughPlastic():
            material_config = generate_rough_plastic_bsdf_config(
                mesh, view.material_channel
            )
        case Plastic():
            material_config = generate_plastic_bsdf_config(mesh, view.material_channel)
        case Principled():
            material_config = generate_principled_bsdf_config(
                mesh, view.material_channel, is_primitive
            )
        case ThinPrincipled():
            material_config = generate_principled_bsdf_config(
                mesh, view.material_channel, is_primitive, thin=True
            )
        case RoughDielectric():
            material_config = generate_rough_dielectric_bsdf_config(
                mesh, view.material_channel
            )
        case ThinDielectric():
            material_config = generate_thin_dielectric_bsdf_config(
                mesh, view.material_channel
            )
        case Dielectric():
            material_config = generate_dielectric_bsdf_config(
                mesh, view.material_channel
            )
        case Hair():
            assert not is_primitive
            material_config = generate_hair_bsdf_config(mesh, view.material_channel)
        case _:
            raise NotImplementedError(
                f"Unknown material type: {type(view.material_channel)}"
            )
    if view.material_channel.two_sided:
        material_config = make_material_two_sided(material_config)

    if view.bump_map is not None:
        material_config = add_bump_map(
            material_config, mesh, view.bump_map, is_primitive
        )

    if view.normal_map is not None:
        material_config = add_normal_map(
            material_config, mesh, view.normal_map, is_primitive
        )

    return material_config
