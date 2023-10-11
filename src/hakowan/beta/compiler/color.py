from .utils import unique_name
from ..grammar.dataframe import DataFrame
from ..grammar.texture import Texture, ScalarField, CheckerBoard, Isocontour
from ..common.colormap.named_colormaps import named_colormaps

import lagrange
import numpy as np


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

        tex.data._internal_name = color_attr_name

def _apply_colormap(df: DataFrame, tex:Texture):
    match tex:
        case ScalarField():
            _apply_colormap_scalar_field(df, tex)
        case CheckerBoard() | Isocontour():
            _apply_colormap(df, tex.texture1)
            _apply_colormap(df, tex.texture2)

def apply_colormap(df: DataFrame, tex: Texture):
    _apply_colormap(df, tex)

