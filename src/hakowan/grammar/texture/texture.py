from dataclasses import dataclass, field
from pathlib import Path

from ...common.color import ColorLike
from ..scale import Attribute, AttributeLike


@dataclass(kw_only=True, slots=True)
class Texture:
    _uv: Attribute | None = None


@dataclass(kw_only=True, slots=True)
class ScalarField(Texture):
    """ The scalar field texture converts an attribute to a either a gradient field or color field.

    Attributes:
        data (AttributeLike): The attribute to convert to a color field.
        colormap (str): The name of the colormap to use.
        domain (tuple[float, float]): The domain of the attribute to map to the colormap.
        range (tuple[float, float]): The range of the colormap to map the attribute to.
    """
    data: AttributeLike
    colormap: str = "viridis"
    domain: tuple[float, float] | None = None
    range: tuple[float, float] | None = None


@dataclass(kw_only=True, slots=True)
class Uniform(Texture):
    color: ColorLike


@dataclass(kw_only=True, slots=True)
class Image(Texture):
    uv: AttributeLike | None = None
    filename: Path


@dataclass(kw_only=True, slots=True)
class CheckerBoard(Texture):
    uv: AttributeLike | None = None
    texture1: Texture
    texture2: Texture


@dataclass(kw_only=True, slots=True)
class Isocontour(Texture):
    data: AttributeLike
    ratio: float = 0.1
    texture1: Texture
    texture2: Texture
