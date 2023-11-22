from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

from ...common.color import ColorLike
from ..scale import Attribute, AttributeLike


@dataclass(kw_only=True, slots=True)
class Texture:
    _uv: Attribute | None = None


TextureLike: TypeAlias = ColorLike | Texture


@dataclass(slots=True)
class ScalarField(Texture):
    """The scalar field texture converts an attribute to a either a gradient field or color field.

    Attributes:
        data (AttributeLike): The attribute to convert to a color field.
        colormap (str | list[ColorLike]): The name of the colormap to use or a list colors.
        domain (tuple[float, float]): The domain of the attribute to map to the colormap.
        range (tuple[float, float]): The range of the colormap to map the attribute to.
    """

    data: AttributeLike
    colormap: str | list[ColorLike] = "viridis"
    domain: tuple[float, float] | None = None
    range: tuple[float, float] | None = None


@dataclass(slots=True)
class Uniform(Texture):
    color: ColorLike


@dataclass(slots=True)
class Image(Texture):
    filename: Path
    uv: AttributeLike | None = None


@dataclass(slots=True)
class Checkerboard(Texture):
    uv: AttributeLike | None = None
    texture1: TextureLike = 0.4
    texture2: TextureLike = 0.2


@dataclass(slots=True)
class Isocontour(Texture):
    data: AttributeLike
    ratio: float = 0.1
    texture1: TextureLike = 0.4
    texture2: TextureLike = 0.2
