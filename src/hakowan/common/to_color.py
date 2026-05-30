from .color import Color, ColorLike
from .named_colors import css_colors


def to_color(data: ColorLike) -> Color:
    """Convert a color-like value to a :class:`Color` object.

    Args:
        data: A float/int (gray), CSS color name, hex string (``"#rrggbb"``),
            or an ``(r, g, b)`` / ``(r, g, b, a)`` tuple.

    Returns:
        The corresponding :class:`Color` instance.

    Raises:
        ValueError: If *data* cannot be converted to a color.
    """
    match data:
        case float() | int():
            return Color(float(data), float(data), float(data))
        case str():
            if data.startswith("#"):
                return Color.from_hex(data)
            elif data.lower() in css_colors:
                return css_colors[data.lower()]
            else:
                raise ValueError(f"Unknown color name: {data}")
        case (r, g, b):
            return Color(r, g, b)
        case (r, g, b, a):
            return Color(r, g, b)
        case _:
            raise ValueError(f"Cannot convert {data} to color")
