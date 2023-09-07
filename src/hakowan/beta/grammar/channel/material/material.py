from dataclasses import dataclass

from ..channel import Channel
from ...texture import Texture


@dataclass(kw_only=True)
class Material(Channel):
    pass


@dataclass(kw_only=True)
class Diffuse(Material):
    reflectance: Texture
