from ..common.color import Color
from ..common.named_colors import css_colors
from ..grammar.texture import (
    Texture,
    ScalarField,
    Uniform,
    Image,
    CheckerBoard,
    Isocontour,
)

from typing import Any


def generate_texture_config(tex: Texture) -> dict:
    return _generate_texture_config(tex)


def generate_uniform_config(tex: Uniform) -> dict:
    mi_config: dict[str, Any] = {
        "type": "rgb",
    }
    match tex.color:
        case float():
            mi_config["value"] = [tex.color] * 3
        case str():
            assert tex.color in css_colors
            mi_config["value"] = css_colors[tex.color]
        case Color():
            mi_config["value"] = tex.color.data.tolist()

    return mi_config


def generate_image_config(tex: Image) -> dict:
    mi_config: dict[str, Any] = {
        "type": "bitmap",
        "filename": tex.filename.resolve(),
    }
    return mi_config


def generate_checker_board_config(tex: CheckerBoard) -> dict:
    mi_config: dict[str, Any] = {
        "type": "checkerboard",
        "color0": generate_texture_config(tex.texture1),
        "color1": generate_texture_config(tex.texture2),
    }
    return mi_config


def generate_isocontour_config(tex: Isocontour) -> dict:
    mi_config: dict[str, Any] = {
        "type": "checkerboard",
        "color0": generate_texture_config(tex.texture1),
        "color1": generate_texture_config(tex.texture2),
    }
    return mi_config


def generate_scalar_field_config(tex: ScalarField) -> dict:
    mi_config: dict[str, Any] = {
        "type": "mesh_attribute",
        "name": tex.data._internal_name,
    }
    return mi_config


def _generate_texture_config(tex: Texture) -> dict:
    match tex:
        case Uniform():
            return generate_uniform_config(tex)
        case Image():
            return generate_image_config(tex)
        case CheckerBoard():
            return generate_checker_board_config(tex)
        case Isocontour():
            return generate_isocontour_config(tex)
        case ScalarField():
            return generate_scalar_field_config(tex)
        case _:
            raise NotImplementedError(f"Unknown texture type: {type(tex)}")
