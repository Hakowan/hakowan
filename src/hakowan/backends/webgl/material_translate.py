"""Translate hakowan ``Material`` → glTF ``pbrMetallicRoughness`` dict.

MVP scope: Diffuse with Uniform reflectance is mapped exactly; everything
else falls back to a gray diffuse with a logged warning. Higher-fidelity
mappings (Principled, ScalarField textures, image textures) land in a
follow-up phase.
"""

from __future__ import annotations

from typing import Any

from ...common import logger
from ...common.to_color import to_color
from ...compiler import View
from ...grammar.channel.material import (
    Diffuse,
    Material,
    Plastic,
    Principled,
)
from ...grammar.texture import Uniform


_DEFAULT_BASE_COLOR: list[float] = [0.5, 0.5, 0.5, 1.0]


def _srgb_to_linear(c: float) -> float:
    """Inverse of the standard sRGB transfer function (single channel)."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _color_to_rgba(color_like: Any) -> list[float]:
    color = to_color(color_like)
    # hakowan colors come in sRGB (CSS names, hex). glTF baseColorFactor is
    # specified in linear space, so convert.
    return [
        _srgb_to_linear(float(color.red)),
        _srgb_to_linear(float(color.green)),
        _srgb_to_linear(float(color.blue)),
        1.0,
    ]


def _reflectance_to_base_color(reflectance: Any) -> list[float]:
    if isinstance(reflectance, Uniform):
        return _color_to_rgba(reflectance.color)
    if isinstance(reflectance, (float, int, str, tuple, list)):
        return _color_to_rgba(reflectance)
    logger.warning(
        f"WebGL backend: reflectance type {type(reflectance).__name__} not yet "
        "supported; falling back to gray."
    )
    return list(_DEFAULT_BASE_COLOR)


def translate_material(view: View) -> tuple[dict[str, Any], bool]:
    """Return ``(pbr_dict, double_sided)`` for the view's material channel."""
    mat = view.material_channel
    if mat is None:
        return ({"baseColorFactor": list(_DEFAULT_BASE_COLOR)}, False)

    double_sided = bool(getattr(mat, "two_sided", False))

    if isinstance(mat, Diffuse):
        return (
            {
                "baseColorFactor": _reflectance_to_base_color(mat.reflectance),
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            double_sided,
        )

    if isinstance(mat, Principled):
        roughness = mat.roughness if isinstance(mat.roughness, float) else 0.5
        metallic = mat.metallic if isinstance(mat.metallic, float) else 0.0
        return (
            {
                "baseColorFactor": _reflectance_to_base_color(mat.color),
                "metallicFactor": float(metallic),
                "roughnessFactor": float(roughness),
            },
            double_sided,
        )

    if isinstance(mat, Plastic):
        return (
            {
                "baseColorFactor": _reflectance_to_base_color(mat.diffuse_reflectance),
                "metallicFactor": 0.0,
                "roughnessFactor": 0.3,
            },
            double_sided,
        )

    logger.warning(
        f"WebGL backend: material type {type(mat).__name__} is not yet "
        "supported; falling back to gray diffuse."
    )
    return (
        {
            "baseColorFactor": list(_DEFAULT_BASE_COLOR),
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        },
        double_sided,
    )
