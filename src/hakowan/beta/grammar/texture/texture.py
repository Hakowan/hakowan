from dataclasses import dataclass, field
from pathlib import Path

from ...common.color import Color
from ..scale import Attribute


@dataclass(kw_only=True)
class Texture:
    pass


@dataclass(kw_only=True)
class ScalarField(Texture):
    data: Attribute
    colormap: str = "viridis"
    _min_value: float | None = None
    _max_value: float | None = None


@dataclass(kw_only=True)
class Uniform(Texture):
    color: float | str | Color


@dataclass(kw_only=True)
class Image(Texture):
    uv: Attribute
    filename: Path


@dataclass(kw_only=True)
class CheckerBoard(Texture):
    uv: Attribute
    texture1: Texture
    texture2: Texture


@dataclass(kw_only=True)
class Isocontour(Texture):
    data: Attribute
    texture1: Texture
    texture2: Texture
    _uv: Attribute | None = None
