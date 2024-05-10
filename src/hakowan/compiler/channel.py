from .view import View
from .attribute import compute_scaled_attribute
from .color import apply_colormap
from .texture import apply_texture
from .utils import unique_name

from ..common import logger
from ..grammar.channel import (
    BumpMap,
    Channel,
    Covariance,
    Normal,
    NormalMap,
    Position,
    Shape,
    Size,
    VectorField,
)
from ..grammar.channel.material import (
    Conductor,
    Dielectric,
    Diffuse,
    Hair,
    Material,
    Plastic,
    Principled,
    RoughConductor,
    RoughDielectric,
    RoughPlastic,
    ThinDielectric,
    ThinPrincipled,
)
from ..grammar.channel.curvestyle import Bend
from ..grammar.dataframe import DataFrame
from ..grammar.mark import Mark
from ..grammar import scale
from ..grammar.scale import Attribute, Normalize
from ..grammar.texture import Texture, Uniform, ScalarField, Image

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
            case VectorField():
                if view.vector_field_channel is None:
                    view.vector_field_channel = channel
            case Covariance():
                if view.covariance_channel is None:
                    view.covariance_channel = channel
            case Shape():
                if view.shape_channel is None:
                    view.shape_channel = channel
            case Material():
                if view.material_channel is None:
                    view.material_channel = channel
            case BumpMap():
                if view.bump_map is None:
                    view.bump_map = channel
            case NormalMap():
                if view.normal_map is None:
                    view.normal_map = channel
            case _:
                raise NotImplementedError(
                    f"Channel type {type(channel)} is not supported"
                )

    # Generate default material channel if not specified.
    if view.material_channel is None:
        view.material_channel = Plastic(diffuse_reflectance=Uniform(color="ivory"))


def _process_channels(view: View):
    assert view.data_frame is not None
    df = view.data_frame
    if view.position_channel is not None:
        assert isinstance(view.position_channel, Position)
        assert isinstance(view.position_channel.data, Attribute)
        attr = view.position_channel.data
        compute_scaled_attribute(df, attr)
        view._active_attributes.append(attr)
    if view.normal_channel is not None:
        assert isinstance(view.normal_channel, Normal)
        assert isinstance(view.normal_channel.data, Attribute)
        attr = view.normal_channel.data
        compute_scaled_attribute(df, attr)
        view._active_attributes.append(attr)
    if view.size_channel is not None:
        assert isinstance(view.size_channel, Size)
        assert isinstance(view.size_channel.data, (Attribute, float))
        if isinstance(view.size_channel.data, Attribute):
            attr = view.size_channel.data
            compute_scaled_attribute(df, attr)
            view._active_attributes.append(attr)
    if view.vector_field_channel is not None:
        assert isinstance(view.vector_field_channel, VectorField)
        assert isinstance(view.vector_field_channel.data, Attribute)
        attr = view.vector_field_channel.data
        compute_scaled_attribute(df, attr)
        view._active_attributes.append(attr)
        match view.vector_field_channel.style:
            case Bend():
                style = view.vector_field_channel.style
                if isinstance(style.direction, str):
                    style.direction = Attribute(style.direction)
                compute_scaled_attribute(df, style.direction)
                view._active_attributes.append(style.direction)
    if view.covariance_channel is not None:
        assert isinstance(view.covariance_channel, Covariance)
        assert isinstance(view.covariance_channel.data, Attribute)
        attr = view.covariance_channel.data
        compute_scaled_attribute(df, attr)
        view._active_attributes.append(attr)
    if view.shape_channel is not None:
        assert isinstance(view.shape_channel, Shape)
        assert view.shape_channel.base_shape in ["sphere", "cube", "disk"]
        if view.shape_channel.orientation is not None:
            if isinstance(view.shape_channel.orientation, str):
                view.shape_channel.orientation = Attribute(view.shape_channel.orientation)
            assert isinstance(view.shape_channel.orientation, Attribute)
            attr = view.shape_channel.orientation
            compute_scaled_attribute(df, attr)
            view._active_attributes.append(attr)
    if view.bump_map is not None:
        tex = view.bump_map.texture
        assert tex is not None
        if isinstance(tex, Texture):
            view._active_attributes += apply_texture(df, tex, view.uv_attribute)
            view.uv_attribute = tex._uv
    if view.normal_map is not None:
        tex = view.normal_map.texture
        assert tex is not None
        if isinstance(tex, Texture):
            if isinstance(tex, Image):
                if not tex.raw:
                    logger.warn("Normal map texture image not in raw format may lead to incorrect result")
            view._active_attributes += apply_texture(df, tex, view.uv_attribute)
            view.uv_attribute = tex._uv
    if view.material_channel is not None:
        match view.material_channel:
            case Diffuse():
                if isinstance(view.material_channel.reflectance, Texture):
                    tex = view.material_channel.reflectance
                    view._active_attributes += apply_texture(df, tex, view.uv_attribute)
                    view.uv_attribute = tex._uv
                    apply_colormap(df, tex)
            case RoughConductor() | RoughDielectric():
                if isinstance(view.material_channel.alpha, Texture):
                    tex = view.material_channel.alpha
                    view._active_attributes += apply_texture(df, tex, view.uv_attribute)
                    view.uv_attribute = tex._uv
                    apply_colormap(df, tex)  # TODO: is this needed?
            case Conductor() | Dielectric() | ThinDielectric() | Hair():
                # Nothing to do.
                pass
            case RoughPlastic() | Plastic():
                if isinstance(view.material_channel.diffuse_reflectance, Texture):
                    tex = view.material_channel.diffuse_reflectance
                    view._active_attributes += apply_texture(df, tex, view.uv_attribute)
                    view.uv_attribute = tex._uv
                    apply_colormap(df, tex)
                if isinstance(view.material_channel.specular_reflectance, Texture):
                    tex = view.material_channel.specular_reflectance
                    view._active_attributes += apply_texture(df, tex, view.uv_attribute)
                    view.uv_attribute = tex._uv
            case Principled() | ThinPrincipled():
                if isinstance(view.material_channel.color, Texture):
                    tex = view.material_channel.color
                    view._active_attributes += apply_texture(df, tex, view.uv_attribute)
                    view.uv_attribute = tex._uv
                    apply_colormap(df, tex)
                if isinstance(view.material_channel.metallic, Texture):
                    tex = view.material_channel.metallic
                    view._active_attributes += apply_texture(df, tex, view.uv_attribute)
                    view.uv_attribute = tex._uv
                if isinstance(view.material_channel.roughness, Texture):
                    tex = view.material_channel.roughness
                    view._active_attributes += apply_texture(df, tex, view.uv_attribute)
                    view.uv_attribute = tex._uv
            case _:
                raise NotImplementedError(
                    f"Channel type {type(view.material_channel)} is not supported"
                )
