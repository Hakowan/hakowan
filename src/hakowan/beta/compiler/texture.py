from .view import View
from .attribute import compute_attribute_minmax, compute_scaled_attribute
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


def apply_texture(df: DataFrame, tex: Texture):
    _apply_texture(df, tex)


def _apply_scalar_field(df: DataFrame, tex: ScalarField):
    if tex.domain is not None:
        # Add a clip scale as the first scale to the attribute.
        clip_scale = Clip(domain=tex.domain)
        clip_scale._child = tex.data.scale
        tex.data.scale = clip_scale

    # Add a normalize scale as the last scale to the attribute.
    domain_min, domain_max = tex.domain if tex.domain is not None else (None, None)
    range_min, range_max = tex.range if tex.range is not None else (0, 1)
    normalize_scale = Normalize(
        bbox_min=np.array([range_min], dtype=np.float32),
        bbox_max=np.array([range_max], dtype=np.float32),
        domain_min=domain_min,
        domain_max=domain_max,
    )
    if tex.data.scale is not None:
        s: Scale = tex.data.scale
        while s._child is not None:
            s = s._child
        s._child = normalize_scale
    else:
        tex.data.scale = normalize_scale

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
