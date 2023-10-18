from .view import View
from .attribute import compute_attribute_minmax, compute_scaled_attribute
from ..grammar.dataframe import DataFrame
from ..grammar.texture import (
    Texture,
    ScalarField,
    Uniform,
    Image,
    CheckerBoard,
    Isocontour,
)
from ..grammar.scale import Attribute, Clip, Normalize, Scale

import lagrange
import math
import numpy as np


def apply_texture(
    df: DataFrame, tex: Texture, uv: Attribute | None = None
) -> list[Attribute]:
    """Apply scale to attributes used in the texture.

    :param df:  The data frame, which will be modified in place.
    :param tex: The texture to process.
    :param uv:  The attribute used as UV coordinates.

    :return: A list of active attributes used by the texture.
    """
    return _apply_texture(df, tex, uv)


def _apply_scalar_field(df: DataFrame, tex: ScalarField):
    if tex.domain is not None:
        # Add a clip scale as the last scale to the attribute.
        clip_scale = Clip(domain=tex.domain)
        if tex.data.scale is not None:
            s: Scale = tex.data.scale
            while s._child is not None:
                s = s._child
            s._child = clip_scale
        else:
            tex.data.scale = clip_scale

    # Add a normalize scale as the first scale to the attribute.
    domain_min, domain_max = tex.domain if tex.domain is not None else (None, None)
    range_min, range_max = tex.range if tex.range is not None else (0, 1)
    normalize_scale = Normalize(
        bbox_min=range_min,
        bbox_max=range_max,
        domain_min=domain_min,
        domain_max=domain_max,
    )
    normalize_scale._child = tex.data.scale
    tex.data.scale = normalize_scale

    # Compute scaled attribute
    compute_scaled_attribute(df, tex.data)

    return [tex.data]


def _apply_image(df: DataFrame, tex: Image, uv: Attribute | None = None):
    assert tex.filename.exists()
    if uv is None:
        compute_scaled_attribute(df, tex.uv)
        tex._uv = tex.uv
    else:
        assert (
            uv.name == tex.uv.name and uv.scale == tex.uv.scale
        ), "Conflicting UV detected"
        tex._uv = uv
    return [tex._uv]


def _apply_checker_board(df: DataFrame, tex: CheckerBoard, uv: Attribute | None = None):
    if uv is None:
        compute_scaled_attribute(df, tex.uv)
        tex._uv = tex.uv
    else:
        assert (
            uv.name == tex.uv.name and uv.scale == tex.uv.scale
        ), "Conflicting UV detected"
        tex._uv = uv

    active_attrs_1 = apply_texture(df, tex.texture1, tex._uv)
    active_attrs_2 = apply_texture(df, tex.texture2, tex._uv)
    return [tex._uv] + active_attrs_1 + active_attrs_2


def _apply_isocontour(df: DataFrame, tex: Isocontour):
    if tex._uv is not None:
        return []
    compute_scaled_attribute(df, tex.data)

    def generate_uv_values(attr_values: lagrange.Attribute):
        assert attr_values.num_channels == 1
        uv_values = np.repeat(attr_values.data, 2).reshape((-1, 2)).astype(np.float32)  # type: ignore
        uv_values[:, 1] += (1 - tex.ratio) / 2
        return uv_values

    # Generate UV.
    mesh = df.mesh
    assert tex.data._internal_name is not None
    attr_name: str = tex.data._internal_name
    assert mesh.has_attribute(attr_name)
    uv_name = "_hakowan_uv"
    if mesh.is_attribute_indexed(attr_name):
        attr = mesh.indexed_attribute(attr_name)
        attr_values = attr.values
        attr_indices = attr.indices
        uv_values = generate_uv_values(attr_values)
        mesh.create_attribute(
            uv_name,
            element=lagrange.AttributeElement.Indexed,
            usage=lagrange.AttributeUsage.UV,
            initial_values=uv_values,
            initial_indices=attr_indices,
        )
    else:
        attr = mesh.attribute(attr_name)
        match attr.element_type:
            case lagrange.AttributeElement.Vertex:
                uv_values = generate_uv_values(attr)
                mesh.create_attribute(
                    uv_name,
                    element=lagrange.AttributeElement.Indexed,
                    usage=lagrange.AttributeUsage.UV,
                    initial_values=uv_values,
                    initial_indices=mesh.facets,
                )
            case lagrange.AttributeElement.Corner:
                uv_values = generate_uv_values(attr)
                mesh.create_attribute(
                    uv_name,
                    element=lagrange.AttributeElement.Indexed,
                    usage=lagrange.AttributeUsage.UV,
                    initial_values=uv_values,
                    initial_indices=np.arange(mesh.num_corners, dtype=np.uint32),
                )
            case _:
                raise NotImplementedError(
                    f"Isocontour does not support attribute element type {attr.element_type}."
                )

    tex._uv = Attribute(name=uv_name, _internal_name=uv_name)
    active_attrs_1 = apply_texture(df, tex.texture1, tex._uv)
    active_attrs_2 = apply_texture(df, tex.texture2, tex._uv)
    return [tex._uv] + active_attrs_1 + active_attrs_2


def _apply_texture(
    df: DataFrame, tex: Texture, uv: Attribute | None = None
) -> list[Attribute]:
    match tex:
        case ScalarField():
            return _apply_scalar_field(df, tex)
        case Uniform():
            # Nothing to do with uniform texture.
            return []
        case Image():
            return _apply_image(df, tex, uv)
        case CheckerBoard():
            return _apply_checker_board(df, tex, uv)
        case Isocontour():
            assert uv is None, "Isocontour texture is incompatible with UV."
            return _apply_isocontour(df, tex)
        case _:
            raise NotImplementedError(f"Texture type {type(tex)} is not supported")
