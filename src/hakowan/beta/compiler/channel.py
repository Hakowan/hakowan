from .view import View
from .attribute import compute_scaled_attribute
from .texture import apply_texture
from ..grammar.dataframe import DataFrame
from ..grammar.scale import Attribute
from ..grammar.channel import Channel, Position, Normal, Size, Diffuse

### Public API


def preprocess_channels(view: View):
    """Preprocess channels in a view.

    Determine the active position, normal, size, uv and material channels. Among these, position,
    normal and uv channels can be automatically generate from data frame if not specified. Size and
    material will be set to default if not specified.

    :param view: The view to be pre-processed. Update will be made in place.
    """
    _preprocess_channels(view)


def process_channels(view: View):
    """Process the channels in a view.

    This step applies scales and textures on the corresponding data.

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


def _preprocess_channels(view: View):
    assert view.data_frame is not None
    for channel in view.channels:
        match channel:
            case Position():
                if view.position_channel is None:
                    view.position_channel = channel
            case Normal():
                if view.normal_channel is None:
                    view.normal_channel = channel
            case Size():
                if view.size_channel is None:
                    view.size_channel = channel
            case Diffuse():
                if view.material_channel is None:
                    view.material_channel = channel
            case _:
                raise NotImplementedError(
                    f"Channel type {type(channel)} is not supported"
                )

    if view.position_channel is None:
        assert view.data_frame is not None
        view.position_channel = _generate_default_position_channel(view.data_frame)


def _process_channels(view: View):
    assert view.data_frame is not None
    df = view.data_frame
    if view.position_channel is not None:
        assert isinstance(view.position_channel, Position)
        attr = view.position_channel.data
        compute_scaled_attribute(df, attr)
        view._active_attributes.append(attr)
    if view.normal_channel is not None:
        assert isinstance(view.normal_channel, Normal)
        attr = view.normal_channel.data
        compute_scaled_attribute(df, attr)
        view._active_attributes.append(attr)
    if view.size_channel is not None:
        assert isinstance(view.size_channel, Size)
        if isinstance(view.size_channel.data, Attribute):
            attr = view.size_channel.data
            compute_scaled_attribute(df, attr)
            view._active_attributes.append(attr)
    if view.material_channel is not None:
        match view.material_channel:
            case Diffuse():
                tex = view.material_channel.reflectance
                view._active_attributes += apply_texture(df, tex)
                view._uv_attribute = tex._uv
            case _:
                raise NotImplementedError(
                    f"Channel type {type(view.material_channel)} is not supported"
                )
