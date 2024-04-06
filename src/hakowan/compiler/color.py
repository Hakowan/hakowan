from .utils import unique_name
from ..grammar.dataframe import DataFrame
from ..grammar.scale import Attribute
from ..grammar.texture import Texture, ScalarField, Checkerboard, Isocontour
from ..common.colormap.named_colormaps import named_colormaps
from ..common.to_color import to_color
from ..common.colormap.colormap import ColorMap
from typing import Callable

import lagrange
import numpy as np


def apply_colormap(df: DataFrame, tex: Texture):
    """Apply a colormap to the given texture and its sub-textures.

    The output color field is stored in Attribute._internal_color_field of the corresponding texture.

    Args:
        df: The data frame
        tex: The texture to apply the colormap on.

    Returns:
        A list of active attributes used by the texture.
    """
    _apply_colormap(df, tex)


def _apply_colormap_scalar_field(df: DataFrame, tex: ScalarField):
    assert isinstance(tex.data, Attribute)
    assert tex.data._internal_name is not None
    mesh = df.mesh
    attr_name = tex.data._internal_name
    assert mesh.has_attribute(attr_name)

    def attr_to_color(colormap: Callable, categories: bool = False):
        nonlocal mesh
        nonlocal attr_name
        nonlocal tex

        unique_values: list[float] = []

        def get_color(value: float):
            if not categories:
                return colormap(value).data
            else:
                assert len(unique_values) > 0
                assert isinstance(colormap, ColorMap)
                num_colors = colormap.num_colors()
                return colormap(
                    unique_values.index(value) % num_colors / (num_colors - 1)
                ).data

        if mesh.is_attribute_indexed(attr_name):
            attr = mesh.indexed_attribute(attr_name)
            value_attr = attr.values
            index_attr = attr.indices
            if categories:
                unique_values = np.unique(value_attr.data).tolist()

            color_data = np.array([get_color(x) for x in value_attr.data])
            color_attr_name = unique_name(mesh, "vertex_color")

            mesh.create_attribute(
                color_attr_name,
                element=attr.element_type,
                usage=lagrange.AttributeUsage.Color,
                initial_values=color_data,
                initial_indices=index_attr.data.copy(),
            )
        else:
            attr = mesh.attribute(attr_name)
            if categories:
                unique_values = np.unique(attr.data).tolist()
            color_data = np.array([get_color(x) for x in attr.data])

            if attr.element_type == lagrange.AttributeElement.Facet:
                color_attr_name = unique_name(mesh, "face_color")
            else:
                color_attr_name = unique_name(mesh, "vertex_color")

            mesh.create_attribute(
                color_attr_name,
                element=attr.element_type,
                usage=lagrange.AttributeUsage.Color,
                initial_values=color_data,
            )
        assert isinstance(tex.data, Attribute)
        tex.data._internal_color_field = color_attr_name

    if tex.colormap == "identity":
        # Assuming attribute is already storing color data.
        attr_to_color(lambda x: to_color(x.tolist()))
    elif isinstance(tex.colormap, str):
        assert tex.colormap in named_colormaps
        colormap = named_colormaps[tex.colormap]
        attr_to_color(colormap, tex.categories)
    elif isinstance(tex.colormap, list):
        colors = np.array([to_color(c).data for c in tex.colormap])
        colormap = ColorMap(colors)
        attr_to_color(colormap, tex.categories)


def _apply_colormap(df: DataFrame, tex: Texture):
    match tex:
        case ScalarField():
            _apply_colormap_scalar_field(df, tex)
        case Checkerboard() | Isocontour():
            assert isinstance(tex.texture1, Texture)
            assert isinstance(tex.texture2, Texture)
            apply_colormap(df, tex.texture1)
            apply_colormap(df, tex.texture2)
