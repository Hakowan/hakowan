from .utils import unique_name
from ..grammar.dataframe import DataFrame
from ..grammar.texture import Texture, ScalarField, CheckerBoard, Isocontour
from ..common.colormap.named_colormaps import named_colormaps

import lagrange
import numpy as np


def apply_colormap(df: DataFrame, tex: Texture):
    """Apply a colormap to the given texture and its sub-textures.

    :param df: The data frame
    :param tex: The texture to apply the colormap on.

    The output color field is stored in Attribute._internal_color_field of the corresponding texture.
    """
    _apply_colormap(df, tex)


def _apply_colormap_scalar_field(df: DataFrame, tex: ScalarField):
    assert tex.data._internal_name is not None
    mesh = df.mesh
    attr_name = tex.data._internal_name
    assert mesh.has_attribute(attr_name)

    if tex.colormap == "identity":
        # Assuming attribute is already storing color data.
        attr = mesh.attribute(attr_name)
        assert attr.num_channels == 3
        assert attr.usage == lagrange.AttributeUsage.Color
        tex.data._internal_color_field = attr_name
    elif tex.colormap in named_colormaps:
        colormap = named_colormaps[tex.colormap]
        if mesh.is_attribute_indexed(attr_name):
            attr = mesh.indexed_attribute(attr_name)
            value_attr = attr.values
            index_attr = attr.indices

            assert value_attr.num_channels == 1
            color_data = np.array([colormap(x).data for x in value_attr.data])
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
            assert attr.num_channels == 1
            color_data = np.array([colormap(x).data for x in attr.data])

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

        tex.data._internal_color_field = color_attr_name


def _apply_colormap(df: DataFrame, tex: Texture):
    match tex:
        case ScalarField():
            _apply_colormap_scalar_field(df, tex)
        case CheckerBoard() | Isocontour():
            apply_colormap(df, tex.texture1)
            apply_colormap(df, tex.texture2)
