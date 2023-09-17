from .view import View
from .attribute import update_scale, compute_attribute_minmax, compute_scaled_attribute
from ..common.color import Color
from ..common.named_colors import css_colors
from ..grammar.dataframe import DataFrame
from ..grammar.texture import (
    Texture,
    ScalarField,
    Uniform,
    Image,
    CheckerBoard,
    Isocontour,
)
from ..grammar.scale import Clip, Normalize, Scale

import numpy as np


def update_texture(df: DataFrame, tex: Texture):
    _update_texture(df, tex)


def apply_texture(df: DataFrame, tex: Texture):
    _apply_texture(df, tex)


def _update_uniform(df: DataFrame, tex: Uniform):
    match tex.color:
        case float():
            tex.color = Color(tex.color, tex.color, tex.color)
        case ["#", *_]:
            tex.color = Color.from_hex(tex.color)
        case str():
            assert tex.color in css_colors, f"Unknown color name {tex.color}"
            tex.color = css_colors[tex.color]
        case Color():
            pass


def _update_scalar_field(df: DataFrame, tex: ScalarField):
    attr = tex.data
    # TODO ensure attribute is a scalar field.

    if attr.scale is not None:
        update_scale(df, attr.name, attr.scale)

    if tex.domain is not None:
        # Use user-specified domain of unscaled data if available.
        tex._value_min, tex._value_max = tex.domain
    else:
        # Compute domain from all data using this texture.
        # Note that value_min and value_max are computed on the __unscaled__ data.
        value_min, value_max = compute_attribute_minmax(df, attr.name)
        tex._value_min = (
            value_min if tex._value_min is None else min(tex._value_min, value_min)
        )
        tex._value_max = (
            value_max if tex._value_max is None else max(tex._value_max, value_max)
        )


def _update_texture(df: DataFrame, tex: Texture):
    match tex:
        case ScalarField():
            _update_scalar_field(df, tex)
        case Uniform():
            _update_uniform(df, tex)
        case Image():
            pass
        case CheckerBoard():
            pass
        case Isocontour():
            pass
        case _:
            raise NotImplementedError(f"Texture type {type(tex)} is not supported")


def _apply_scalar_field(df: DataFrame, tex: ScalarField):
    assert tex._value_min is not None
    assert tex._value_max is not None

    # Add a clip scale as the first scale to the attribute.
    clip_scale = Clip(domain=(tex._value_min, tex._value_max))
    clip_scale._child = tex.data.scale
    tex.data.scale = clip_scale

    # Add a normalize scale as the last scale to the attribute.
    normalize_scale = Normalize(
        bbox_min=np.array([0], dtype=np.float32),
        bbox_max=np.array([1], dtype=np.float32),
        _value_min=tex._value_min,
        _value_max=tex._value_max,
    )
    s: Scale = tex.data.scale
    while s._child is not None:
        s = s._child
    s._child = normalize_scale

    # Compute scaled attribute
    compute_scaled_attribute(df, tex.data)


def _apply_texture(df: DataFrame, tex: Texture):
    match tex:
        case ScalarField():
            _apply_scalar_field(df, tex)
        case Uniform():
            # Nothing to do with uniform texture.
            pass
        case Image():
            pass
        case CheckerBoard():
            pass
        case Isocontour():
            pass
        case _:
            raise NotImplementedError(f"Texture type {type(tex)} is not supported")
