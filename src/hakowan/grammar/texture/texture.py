from dataclasses import dataclass, field
from os import PathLike
from typing import TypeAlias

from ...common.color import ColorLike
from ..scale import Attribute, AttributeLike


@dataclass(kw_only=True, slots=True)
class Texture:
    """Texture provides the mapping from raw data to visual properties such as color."""

    _uv: Attribute | None = None


TextureLike: TypeAlias = ColorLike | Texture
"""TextureLike is a type alias for a texture or color."""


@dataclass(slots=True)
class Uniform(Texture):
    """The uniform texture provides a constant color field.

    Attributes:
        color (ColorLike): The color of the uniform texture.
    """

    color: ColorLike


@dataclass(slots=True)
class Image(Texture):
    """The image texture uses a texture image to provide a color field.

    Attributes:
        filename (PathLike): The path to the texture image.
        uv (AttributeLike): The attribute to use as the texture coordinates.
    """

    filename: PathLike
    uv: AttributeLike | None = None


@dataclass(slots=True)
class Checkerboard(Texture):
    """The checkerboard texture provides a checkerboard color/value field.

    Attributes:
        uv (AttributeLike): The attribute to use as the texture coordinates.
        texture1 (TextureLike): The texture to use for the first color/value.
        texture2 (TextureLike): The texture to use for the second color/value.
        size (int): The size of the checkerboard (e.g. 8 means 8x8 checkerboard).
    """

    uv: AttributeLike | None = None
    texture1: TextureLike = 0.8
    texture2: TextureLike = 0.2
    size: int = 8


@dataclass(slots=True)
class Isocontour(Texture):
    """The isocontour texture provides a color/value field based on the isocontours of an attribute.

    Attributes:
        data (AttributeLike): The attribute used to generate the isocontours.
        ratio (float): The ratio of the isocontour thickness to non-isocontour thickness.
        texture1 (TextureLike): The texture to use for the isocontour regions.
        texture2 (TextureLike): The texture to use for the non-isocontour regions.
        num_contours (int): The number of isocontours to generate within a unit distance.
    """

    data: AttributeLike
    ratio: float = 0.1
    texture1: TextureLike = 0.4
    texture2: TextureLike = 0.2
    num_contours: int = 8


@dataclass(slots=True)
class ScalarField(Texture):
    """The scalar field texture converts an attribute to a either a value field or color field.

    Attributes:
        data (AttributeLike): The attribute to convert to a color field.
        colormap (str | list[ColorLike]): The name of the colormap to use or a list colors.
        domain (tuple[float, float]): The domain of the attribute to map to the colormap.
        range (tuple[float, float]): The range of the colormap to map the attribute to.
        categories (bool): Whether the attribute represents categorical data (i.e. discrete values).
    """

    data: AttributeLike
    colormap: str | list[ColorLike] = "viridis"
    domain: tuple[float, float] | None = None
    range: tuple[float, float] | None = None
    categories: bool = False
