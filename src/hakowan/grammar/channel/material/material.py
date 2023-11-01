from dataclasses import dataclass

from ..channel import Channel
from ...texture import Texture
from ....common.color import ColorLike


@dataclass(kw_only=True, slots=True)
class Material(Channel):
    pass


@dataclass(kw_only=True, slots=True)
class Diffuse(Material):
    reflectance: Texture | ColorLike = 0.5


@dataclass(kw_only=True, slots=True)
class Conductor(Material):
    material: str

@dataclass(kw_only=True, slots=True)
class RoughConductor(Conductor):
    distribution: str = "beckmann"
    alpha: Texture | float = 0.1

@dataclass(kw_only=True, slots=True)
class Plastic(Material):
    diffuse_reflectance: Texture | ColorLike = 0.5
    specular_reflectance: Texture | float = 1.0

@dataclass(kw_only=True, slots=True)
class RoughPlastic(Plastic):
    distribution: str = "beckmann"
    alpha: float = 0.1

@dataclass(kw_only=True, slots=True)
class Principled(Material):
    color: Texture | ColorLike = 0.5
    roughness: Texture | float = 0.5
    metallic: Texture | float = 0.0
