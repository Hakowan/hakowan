from .view import View
from .attribute import compute_attribute_minmax, compute_scaled_attribute
from .utils import get_default_uv, unique_name
from ..grammar.dataframe import DataFrame
from ..grammar.texture import (
    Texture,
    ScalarField,
    Uniform,
    Image,
    Checkerboard,
    Isocontour,
)
from ..grammar.scale import Attribute, Clip, Normalize, Scale

import lagrange
import math
import numpy as np
from pathlib import Path


def apply_texture(
    df: DataFrame, tex: Texture, uv: Attribute | None = None
) -> list[Attribute]:
    """Apply scale to attributes used in the texture.

    :param df:  The data frame, which will be modified in place.
    :param tex: The texture to process.
    :param uv:  The attribute used as UV coordinates.

    :return: A list of active attributes used by the texture.
    """
    r = _apply_texture(df, tex, uv)
    for attr in r:
        assert attr is not None
    return r


def _apply_scalar_field(df: DataFrame, tex: ScalarField):
    if isinstance(tex.data, str):
        tex.data = Attribute(name=tex.data)
    assert isinstance(tex.data, Attribute)

    if tex.domain is not None:
        # Add a clip scale as the last scale to the attribute.
        clip_scale = Clip(domain=tex.domain)
        if tex.data.scale is not None:
            assert isinstance(tex.data.scale, Scale)
            s: Scale = tex.data.scale
            while s._child is not None:
                s = s._child
            s._child = clip_scale
        else:
            tex.data.scale = clip_scale

    if tex.colormap != "identity":
        # Add a normalize scale as the first scale to the attribute if no scale is provided.
        if tex.data.scale is None:
            domain_min, domain_max = tex.domain if tex.domain is not None else (None, None)
            range_min, range_max = tex.range if tex.range is not None else (0, 1)
            normalize_scale = Normalize(
                range_min=range_min,
                range_max=range_max,
                domain_min=domain_min,
                domain_max=domain_max,
            )
            tex.data.scale = normalize_scale

    # Compute scaled attribute
    compute_scaled_attribute(df, tex.data)

    return [tex.data]


def _apply_image(df: DataFrame, tex: Image, uv: Attribute | None = None):
    filename = Path(tex.filename)
    assert filename.exists()
    if uv is None:
        if tex.uv is None:
            assert df.mesh is not None
            tex.uv = Attribute(name=get_default_uv(df.mesh))
        elif isinstance(tex.uv, str):
            assert df.mesh is not None
            assert df.mesh.has_attribute(tex.uv)
            tex.uv = Attribute(name=tex.uv)
        assert isinstance(tex.uv, Attribute)
        compute_scaled_attribute(df, tex.uv)
        tex._uv = tex.uv
    elif tex.uv is not None:
        if isinstance(tex.uv, str):
            assert df.mesh is not None
            assert df.mesh.has_attribute(tex.uv)
            tex.uv = Attribute(name=tex.uv)
        assert isinstance(tex.uv, Attribute)
        assert (
            uv.name == tex.uv.name and uv.scale == tex.uv.scale
        ), "Conflicting UV detected"
        tex._uv = uv
    else:
        assert df.mesh is not None
        assert df.mesh.has_attribute(uv.name)
        assert uv._internal_name is not None
        assert df.mesh.has_attribute(uv._internal_name)
        tex.uv = uv
        tex._uv = uv
    return [tex._uv]


def _apply_checker_board(df: DataFrame, tex: Checkerboard, uv: Attribute | None = None):
    if uv is None:
        if tex.uv is None:
            assert df.mesh is not None
            tex.uv = Attribute(name=get_default_uv(df.mesh))
        elif isinstance(tex.uv, str):
            assert df.mesh is not None
            assert df.mesh.has_attribute(tex.uv)
            tex.uv = Attribute(name=tex.uv)
        assert isinstance(tex.uv, Attribute)
        compute_scaled_attribute(df, tex.uv)
        tex._uv = tex.uv
    elif tex.uv is not None:
        if isinstance(tex.uv, str):
            assert df.mesh is not None
            assert df.mesh.has_attribute(tex.uv)
            tex.uv = Attribute(name=tex.uv)
        assert isinstance(tex.uv, Attribute)
        assert (
            uv.name == tex.uv.name and uv.scale == tex.uv.scale
        ), "Conflicting UV detected"
        tex._uv = uv
    else:
        assert df.mesh is not None
        assert df.mesh.has_attribute(uv.name)
        assert uv._internal_name is not None
        assert df.mesh.has_attribute(uv._internal_name)
        tex.uv = uv
        tex._uv = uv

    if not isinstance(tex.texture1, Texture):
        tex.texture1 = Uniform(color=tex.texture1)
    if not isinstance(tex.texture2, Texture):
        tex.texture2 = Uniform(color=tex.texture2)
    active_attrs_1 = apply_texture(df, tex.texture1, tex._uv)
    active_attrs_2 = apply_texture(df, tex.texture2, tex._uv)
    return [tex._uv] + active_attrs_1 + active_attrs_2


def _apply_isocontour(df: DataFrame, tex: Isocontour, uv: Attribute | None = None):
    if tex._uv is not None:
        # This texture is already processed.
        return []

    if isinstance(tex.data, str):
        tex.data = Attribute(name=tex.data)
    assert isinstance(tex.data, Attribute)

    if not isinstance(tex.texture1, Texture):
        tex.texture1 = Uniform(color=tex.texture1)
    if not isinstance(tex.texture2, Texture):
        tex.texture2 = Uniform(color=tex.texture2)

    compute_scaled_attribute(df, tex.data)

    def generate_uv_values(attr_values: lagrange.Attribute):
        assert attr_values.num_channels == 1
        s = tex.num_contours
        assert s > 0
        uv_values = np.repeat(attr_values.data * s, 2).reshape((-1, 2)).astype(np.float32)  # type: ignore
        uv_values[:, 1] += (1 - tex.ratio) / 2
        return uv_values

    # Generate UV.
    mesh = df.mesh
    assert tex.data._internal_name is not None
    attr_name: str = tex.data._internal_name
    assert mesh.has_attribute(attr_name)
    uv_name = f"_hakowan_isocontour_uv_generated_from_{tex.data.name}"

    if uv is not None:
        assert uv.name == uv_name, "Conflicting UV detected"
        tex._uv = uv
        active_attrs_1 = apply_texture(df, tex.texture1, tex._uv)
        active_attrs_2 = apply_texture(df, tex.texture2, tex._uv)
        return active_attrs_1 + active_attrs_2


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
        case Checkerboard():
            return _apply_checker_board(df, tex, uv)
        case Isocontour():
            return _apply_isocontour(df, tex, uv)
        case _:
            raise NotImplementedError(f"Texture type {type(tex)} is not supported")
