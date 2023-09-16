from .view import View
from .attribute import apply_scale
from ..grammar.dataframe import DataFrame
from ..grammar.scale import Attribute
from ..grammar.channel import Position, Normal, Size, Diffuse


def _process_position_channel(data: DataFrame, channel: Position):
    attr: Attribute = channel.data
    if attr.scale is not None:
        apply_scale(data, attr)

def process_channels(view: View):
    for channel in view.channels:
        match channel:
            case Position():
                if view._position_channel is not None:
                    view._position_channel = channel
                    assert view.data is not None
                    _process_position_channel(view.data, channel)
            case Normal():
                pass
            case Size():
                pass
            case Diffuse():
                pass
            case _:
                raise NotImplementedError(f"Channel type {type(channel)} is not supported")
