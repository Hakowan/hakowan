from dataclasses import dataclass

from ..channel import Channel
from ...texture import Texture
from ....common.color import ColorLike


@dataclass(kw_only=True, slots=True)
class Material(Channel):
    two_sided: bool = False
    bump_map: Texture | None = None


@dataclass(slots=True)
class Diffuse(Material):
    reflectance: Texture | ColorLike = 0.5


@dataclass(slots=True)
class Conductor(Material):
    material: str


@dataclass(slots=True)
class RoughConductor(Conductor):
    distribution: str = "beckmann"
    alpha: Texture | float = 0.1


@dataclass(slots=True)
class Plastic(Material):
    diffuse_reflectance: Texture | ColorLike = 0.5
    specular_reflectance: Texture | float = 1.0


@dataclass(slots=True)
class RoughPlastic(Plastic):
    distribution: str = "beckmann"
    alpha: float = 0.1


@dataclass(slots=True)
class Principled(Material):
    color: Texture | ColorLike = 0.5
    roughness: Texture | float = 0.5
    metallic: Texture | float = 0.0
