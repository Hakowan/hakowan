from dataclasses import dataclass

from .medium import Medium
from ..channel import Channel
from ...texture import Texture, TextureLike
from ....common.color import ColorLike


@dataclass(kw_only=True, slots=True)
class Material(Channel):
    """Material base class.

    Attributes:
        two_sided: Whether to render both sides of the surface (default: False).
        bump_map: Bump map texture (default: None).
    """

    two_sided: bool = False
    bump_map: Texture | None = None


@dataclass(slots=True)
class Diffuse(Material):
    """Diffuse material.

    Attributes:
        reflectance: Diffuse reflectance (i.e. base color) texture (default: 0.5).
    """

    reflectance: TextureLike = 0.5


@dataclass(slots=True)
class Conductor(Material):
    """Conductor material.

    Attributes:
        material: Conductor material name based on [Mitsuba preset](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#conductor-ior-list).
    """

    material: str


@dataclass(slots=True)
class RoughConductor(Conductor):
    """Rough conductor material.

    Attributes:
        distribution: Microfacet distribution (default: "beckmann").
        alpha: Roughness value (default: 0.1).
    """
    distribution: str = "beckmann"
    alpha: Texture | float = 0.1


@dataclass(slots=True)
class Plastic(Material):
    """Plastic material.

    Attributes:
        diffuse_reflectance: Diffuse reflectance (i.e. base color) texture (default: 0.5).
        specular_reflectance: Specular reflectance texture (default: 1.0).
    """
    diffuse_reflectance: TextureLike = 0.5
    specular_reflectance: Texture | float = 1.0


@dataclass(slots=True)
class RoughPlastic(Plastic):
    """Rough plastic material.

    Attributes:
        distribution: Microfacet distribution (default: "beckmann").
        alpha: Roughness value (default: 0.1).
    """
    distribution: str = "beckmann"
    alpha: float = 0.1


@dataclass(slots=True)
class Principled(Material):
    """Principled material.

    Attributes:
        color: Base color texture (default: 0.5).
        roughness: Roughness texture (default: 0.5).
        metallic: Metallic texture (default: 0.0).
    """
    color: TextureLike = 0.5
    roughness: Texture | float = 0.5
    metallic: Texture | float = 0.0


@dataclass(slots=True)
class Dielectric(Material):
    """Dielectric material.

    Attributes:
        int_ior: Interior index of refraction (default: "bk7").
        ext_ior: Exterior index of refraction (default: "air").
        medium: Medium (default: None).
    """
    int_ior: str | float = "bk7"
    ext_ior: str | float = "air"
    medium: Medium | None = None


@dataclass(slots=True)
class ThinDielectric(Dielectric):
    """Thin dielectric material."""
    pass


@dataclass(slots=True)
class RoughDielectric(Dielectric):
    """Rough dielectric material.

    Attributes:
        distribution: Microfacet distribution (default: "beckmann").
        alpha: Roughness value (default: 0.1).
    """
    distribution: str = "beckmann"
    alpha: Texture | float = 0.1


@dataclass(slots=True)
class Hair(Material):
    eumelanin: float = 1.3
    pheomelanin: float = 0.2
