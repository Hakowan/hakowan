from ...common.color import ColorLike
from ...common.to_color import to_color


def generate_color_config(value: ColorLike) -> dict:
    c = to_color(value)
    return {"type": "rgb", "value": c.data.tolist()}
