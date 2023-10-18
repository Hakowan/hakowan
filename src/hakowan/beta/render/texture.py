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


def generate_texture_config(tex: Texture, is_color: bool = False) -> dict:
    """ Generate a texture configuration for Mitsuba.

    :param tex: The texture to generate the configuration for.
    :param is_color: Whether the texture is used as a color or not.

    :return: The Mitsuba configuration for the texture.
    """
    return _generate_texture_config(tex, is_color)


def generate_uniform_config(tex: Uniform) -> dict:
    mi_config: dict[str, Any] = {
        "type": "rgb",
    }
    match tex.color:
        case float():
            mi_config["value"] = tex.color
        case str():
            assert tex.color in css_colors
            mi_config["value"] = css_colors[tex.color].data.tolist()
        case Color():
            mi_config["value"] = tex.color.data.tolist()

    return mi_config


def generate_image_config(tex: Image) -> dict:
    mi_config: dict[str, Any] = {
        "type": "bitmap",
        "filename": tex.filename.resolve(),
    }
    return mi_config


def generate_checker_board_config(tex: CheckerBoard, is_color: bool) -> dict:
    mi_config: dict[str, Any] = {
        "type": "checkerboard",
        "color0": generate_texture_config(tex.texture1, is_color),
        "color1": generate_texture_config(tex.texture2, is_color),
    }
    return mi_config


def generate_isocontour_config(tex: Isocontour, is_color: bool) -> dict:
    mi_config: dict[str, Any] = {
        "type": "checkerboard",
        "color0": generate_texture_config(tex.texture1, is_color),
        "color1": generate_texture_config(tex.texture2, is_color),
    }
    return mi_config


def generate_scalar_field_config(tex: ScalarField, is_color: bool) -> dict:
    assert tex.data._internal_name is not None
    if is_color:
        name = tex.data._internal_color_field
        assert name in ["vertex_color", "face_color"]
    else:
        if tex.data._internal_name.startswith("vertex_"):
            name = f"vertex_{tex.data._internal_name}"
        elif tex.data._internal_name.startswith("face_"):
            name = f"face_{tex.data._internal_name}"
        else:
            name = tex.data._internal_name
        if name.endswith("_0"):
            name = name[:-2]
    mi_config: dict[str, Any] = {
        "type": "mesh_attribute",
        "name": name,
    }
    return mi_config


def _generate_texture_config(tex: Texture, is_color: bool) -> dict:
    match tex:
        case Uniform():
            return generate_uniform_config(tex)
        case Image():
            return generate_image_config(tex)
        case CheckerBoard():
            return generate_checker_board_config(tex, is_color)
        case Isocontour():
            return generate_isocontour_config(tex, is_color)
        case ScalarField():
            return generate_scalar_field_config(tex, is_color)
        case _:
            raise NotImplementedError(f"Unknown texture type: {type(tex)}")
