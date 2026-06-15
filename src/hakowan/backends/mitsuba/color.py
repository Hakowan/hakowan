from ...common.color import ColorLike, srgb_to_linear
from ...common.to_color import to_color


def generate_color_config(value: ColorLike) -> dict:
    c = to_color(value)
    # Mitsuba's ``rgb`` spectrum interprets values as linear RGB, but color
    # names / hex / user floats follow the sRGB convention. Decode so renders
    # match the WebGL backend and are physically correct.
    linear = [srgb_to_linear(float(x)) for x in c.data.tolist()]
    return {"type": "rgb", "value": linear}
