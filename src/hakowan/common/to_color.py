from .color import Color, ColorLike
from .named_colors import css_colors


def to_color(data: ColorLike):
    match data:
        case float() | int():
            return Color(float(data), float(data), float(data))
        case str():
            if data.startswith("#"):
                return Color.from_hex(data)
            elif data.lower() in css_colors:
                return css_colors[data.lower()]
        case (r, g, b):
            return Color(r, g, b)
        case _:
            raise ValueError(f"Cannot convert {data} to color")
