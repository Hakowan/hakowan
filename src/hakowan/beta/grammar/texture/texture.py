from dataclasses import dataclass
from typing import Union
from pathlib import Path

from ...common.color import Color


@dataclass(kw_only=True)
class Texture:
    _active_attribute: list[str]


@dataclass(kw_only=True)
class ScalarField(Texture):
    data: str
    colormap: str
    min_value: float
    max_value: float


@dataclass(kw_only=True)
class Uniform(Texture):
    color: Union[float, str, Color]


@dataclass(kw_only=True)
class Image(Texture):
    uv: str
    filename: Path


@dataclass(kw_only=True)
class CheckerBoard(Texture):
    uv: str
    color1: Texture
    color2: Texture


@dataclass(kw_only=True)
class Isocontour(Texture):
    data: str
    color1: Texture
    color2: Texture
    _uv: str
