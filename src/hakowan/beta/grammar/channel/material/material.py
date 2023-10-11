from dataclasses import dataclass

from ..channel import Channel
from ...texture import Texture


@dataclass(kw_only=True)
class Material(Channel):
    pass


@dataclass(kw_only=True)
class Diffuse(Material):
    reflectance: Texture


@dataclass(kw_only=True)
class Conductor(Material):
    material: str

@dataclass(kw_only=True)
class RoughConductor(Conductor):
    distribution: str = "beckmann"
    alpha: Texture | float = 0.1

@dataclass(kw_only=True)
class Plastic(Material):
    diffuse_reflectance: Texture | float = 0.5
    specular_reflectance: Texture | float = 1.0

@dataclass(kw_only=True)
class RoughPlastic(Plastic):
    distribution: str = "beckmann"
    alpha: float = 0.1
