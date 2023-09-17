from .view import View
from .attribute import update_scale, compute_scaled_attribute
from ..grammar.dataframe import DataFrame
from ..grammar.scale import Attribute
from ..grammar.channel import Channel, Position, Normal, Size, Diffuse

### Public API


def preprocess_channels(view: View):
    """Preprocess channels in a view.

    This step does two things:

    1. Some channel may involve data-dependent parameters. The purpose of the preprocessing step is
       to compute such parameters.
    2. Determine the active position, normal, size, uv and material channels. Among these, position,
       normal and uv channels can be automatically generate from data frame if not specified. Size
       and material will be set to default if not specified.

    :param view: The view to be pre-processed. Update will be made in place.
    """
    _preprocess_channels(view)


def process_channels(view: View):
    """Process the channels in a view.

    This step applies scales on the corresponding data.

    :param view: The view to be processed. Update will be made in place.
    """
    _process_channels(view)


### Private API


def _generate_default_position_channel(df: DataFrame):
    mesh = df.mesh
    assert mesh is not None

    attr = Attribute(name=mesh.attr_name_vertex_to_position)
    position_channel = Position(data=attr)
    return position_channel


def _preprocess_channel(df: DataFrame, channel: Channel):
    match channel:
        case Position() | Normal() | Size():
            attr = channel.data
            if isinstance(attr, Attribute):
                if attr.scale is not None:
                    update_scale(df, attr.name, attr.scale)
        case Diffuse():
            tex = channel.reflectance
            # update_texture(df, tex)


def _preprocess_channels(view: View):
    for channel in view.channels:
        match channel:
            case Position():
                if view._position_channel is None:
                    view._position_channel = channel
            case Normal():
                if view._normal_channel is None:
                    view._normal_channel = channel
            case Size():
                if view._size_channel is None:
                    view._size_channel = channel
            case Diffuse():
                if view._material_channel is None:
                    view._material_channel = channel
            case _:
                raise NotImplementedError(
                    f"Channel type {type(channel)} is not supported"
                )
        assert view.data_frame is not None
        _preprocess_channel(view.data_frame, channel)

    if view._position_channel is None:
        assert view.data_frame is not None
        view._position_channel = _generate_default_position_channel(view.data_frame)


def _process_channel(df: DataFrame, channel: Channel):
    match channel:
        case Position() | Normal():
            attr = channel.data
            compute_scaled_attribute(df, attr)
        case Size():
            if isinstance(channel.data, Attribute):
                attr = channel.data
                compute_scaled_attribute(df, attr)
        case Diffuse():
            # TODO
            pass


def _process_channels(view: View):
    assert view.data_frame is not None
    if view._position_channel is not None:
        assert isinstance(view._position_channel, Position)
        _process_channel(view.data_frame, view._position_channel)
    if view._normal_channel is not None:
        assert isinstance(view._normal_channel, Normal)
        _process_channel(view.data_frame, view._normal_channel)
    if view._size_channel is not None:
        assert isinstance(view._size_channel, Size)
        _process_channel(view.data_frame, view._size_channel)
    if view._uv_channel is not None:
        pass
    if view._material_channel is not None:
        pass
