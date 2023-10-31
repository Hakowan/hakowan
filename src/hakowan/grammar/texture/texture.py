from dataclasses import dataclass, field
from pathlib import Path

from ...common.color import ColorLike
from ..scale import Attribute


@dataclass(kw_only=True, slots=True)
class Texture:
    _uv: Attribute | None = None


@dataclass(kw_only=True, slots=True)
class ScalarField(Texture):
    data: Attribute
    colormap: str = "viridis"
    domain: tuple[float, float] | None = None
    range: tuple[float, float] | None = None


@dataclass(kw_only=True, slots=True)
class Uniform(Texture):
    color: ColorLike


@dataclass(kw_only=True, slots=True)
class Image(Texture):
    uv: Attribute | None = None
    filename: Path


@dataclass(kw_only=True, slots=True)
class CheckerBoard(Texture):
    uv: Attribute | None = None
    texture1: Texture
    texture2: Texture


@dataclass(kw_only=True, slots=True)
class Isocontour(Texture):
    data: Attribute
    ratio: float = 0.1
    texture1: Texture
    texture2: Texture