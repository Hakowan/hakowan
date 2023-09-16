from .view import View
from .attribute import apply_scale, update_scale
from ..grammar.dataframe import DataFrame
from ..grammar.scale import Attribute
from ..grammar.channel import Position, Normal, Size, Diffuse


def _generate_default_position_channel(df: DataFrame):
    mesh = df.mesh
    assert mesh is not None

    attr = Attribute(name=mesh.attr_name_vertex_to_position)
    position_channel = Position(data=attr)
    return position_channel


def _preprocess_position_channel(df: DataFrame, channel: Position):
    attr: Attribute = channel.data
    if attr.scale is not None:
        update_scale(df, attr.name, attr.scale)


def preprocess_channels(view: View):
    for channel in view.channels:
        match channel:
            case Position():
                if view._position_channel is None:
                    view._position_channel = channel
                    assert view.data_frame is not None
                    _preprocess_position_channel(view.data_frame, channel)
            case Normal():
                pass
            case Size():
                pass
            case Diffuse():
                pass
            case _:
                raise NotImplementedError(
                    f"Channel type {type(channel)} is not supported"
                )

    if view._position_channel is None:
        assert view.data_frame is not None
        view._position_channel = _generate_default_position_channel(view.data_frame)


def _process_position_channel(df: DataFrame, channel: Position):
    attr: Attribute = channel.data
    if attr.scale is not None:
        apply_scale(df, attr.name, attr.scale)


def process_channels(view: View):
    assert view.data_frame is not None
    if view._position_channel is not None:
        assert isinstance(view._position_channel, Position)
        _process_position_channel(view.data_frame, view._position_channel)
    if view._normal_channel is not None:
        pass
    if view._size_channel is not None:
        pass
    if view._uv_channel is not None:
        pass
    if view._material_channel is not None:
        pass
