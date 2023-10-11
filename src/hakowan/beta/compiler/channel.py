from .view import View
from .attribute import compute_scaled_attribute
from .color import apply_colormap
from .texture import apply_texture
from ..grammar.channel import Channel, Position, Normal, Size, Material, Diffuse, Conductor
from ..grammar.dataframe import DataFrame
from ..grammar.mark import Mark
from ..grammar import scale
from ..grammar.scale import Attribute, Normalize
from ..grammar.texture import Uniform, ScalarField

import lagrange
import numpy as np
from numpy.linalg import norm

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

    bbox_min = np.amin(mesh.vertices, axis=0)
    bbox_max = np.amax(mesh.vertices, axis=0)
    max_side = np.amax(bbox_max - bbox_min)
    diag = norm(bbox_max - bbox_min)

    attr = Attribute(
        name=mesh.attr_name_vertex_to_position,
        scale=Normalize(
            bbox_min=-np.ones(mesh.dimension),
            bbox_max=np.ones(mesh.dimension),
            _child=scale.Uniform(factor=max_side / diag),
        ),
    )
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
            case Material():
                if view.material_channel is None:
                    view.material_channel = channel
            case _:
                raise NotImplementedError(
                    f"Channel type {type(channel)} is not supported"
                )

    # Generate default position channel if not specified.
    if view.position_channel is None:
        assert view.data_frame is not None
        view.position_channel = _generate_default_position_channel(view.data_frame)

    # Generate default normal channel if not specified.
    if view.mark == Mark.Surface and view.normal_channel is None:
        mesh = view.data_frame.mesh
        normal_attr_id = lagrange.compute_normal(mesh)
        normal_attr_name = mesh.get_attribute_name(normal_attr_id)
        view.normal_channel = Normal(data=Attribute(name=normal_attr_name))

    # Generate default material channel if not specified.
    if view.material_channel is None:
        view.material_channel = Diffuse(reflectance=Uniform(color="ivory"))


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
                apply_colormap(df, tex)
            case Conductor():
                # Nothing to do.
                pass
            case _:
                raise NotImplementedError(
                    f"Channel type {type(view.material_channel)} is not supported"
                )
