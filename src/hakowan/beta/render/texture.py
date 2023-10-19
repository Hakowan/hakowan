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
import lagrange


def generate_texture_config(
    mesh: lagrange.SurfaceMesh, tex: Texture, is_color: bool = False, is_primitive=False
) -> dict:
    """Generate a texture configuration for Mitsuba.

    :param mesh: The mesh that contains the texture data.
    :param tex: The texture to generate the configuration for.
    :param is_color: Whether the texture is used as a color or not.
    :param is_primitive: Whether the texture is used on primitives or mesh.

    :return: The Mitsuba configuration for the texture.
    """
    return _generate_texture_config(mesh, tex, is_color, is_primitive)


def generate_uniform_config(tex: Uniform) -> dict:
    mi_config: dict[str, Any] = {
        "type": "rgb",
    }
    match tex.color:
        case float():
            mi_config["value"] = tex.color
        case str():
            if tex.color.startswith("#"):
                c = Color.from_hex(tex.color)
                mi_config["value"] = c.data.tolist()
            else:
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


def generate_checker_board_config(
    mesh: lagrange.SurfaceMesh, tex: CheckerBoard, is_color: bool
) -> dict:
    mi_config: dict[str, Any] = {
        "type": "checkerboard",
        "color0": generate_texture_config(mesh, tex.texture1, is_color),
        "color1": generate_texture_config(mesh, tex.texture2, is_color),
    }
    return mi_config


def generate_isocontour_config(
    mesh: lagrange.SurfaceMesh, tex: Isocontour, is_color: bool
) -> dict:
    mi_config: dict[str, Any] = {
        "type": "checkerboard",
        "color0": generate_texture_config(mesh, tex.texture1, is_color),
        "color1": generate_texture_config(mesh, tex.texture2, is_color),
    }
    return mi_config


def generate_scalar_field_config(
    mesh: lagrange.SurfaceMesh, tex: ScalarField, is_color: bool, is_primitive: bool
) -> dict:
    assert tex.data._internal_name is not None
    if is_primitive:
        assert is_color
        name = tex.data._internal_color_field
        assert name in ["vertex_color", "face_color"]
        colors = mesh.attribute(name).data
        return {"colors": colors.tolist()}
    elif is_color:
        name = tex.data._internal_color_field
        assert name in ["vertex_color", "face_color"]
    else:
        assert mesh.has_attribute(tex.data._internal_name)
        attr = mesh.attribute(tex.data._internal_name)
        match attr.element_type:
            case lagrange.AttributeElement.Vertex:
                name = f"vertex_{tex.data._internal_name}"
            case lagrange.AttributeElement.Facet:
                name = f"face_{tex.data._internal_name}"
            case _:
                raise NotImplementedError(
                    f"Unsupported attribute element type: {attr.element_type}"
                )

    mi_config: dict[str, Any] = {
        "type": "mesh_attribute",
        "name": name,
    }
    return mi_config


def _generate_texture_config(
    mesh: lagrange.SurfaceMesh, tex: Texture, is_color: bool, is_primitive=False
) -> dict:
    match tex:
        case Uniform():
            return generate_uniform_config(tex)
        case Image():
            return generate_image_config(tex)
        case CheckerBoard():
            return generate_checker_board_config(mesh, tex, is_color)
        case Isocontour():
            assert not is_primitive
            return generate_isocontour_config(mesh, tex, is_color)
        case ScalarField():
            return generate_scalar_field_config(mesh, tex, is_color, is_primitive)
        case _:
            raise NotImplementedError(f"Unknown texture type: {type(tex)}")
