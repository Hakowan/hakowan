""" Utility functions for rendering. """
import warnings

import numpy as np
from numpy.linalg import norm

from .layer_data import LayerData, Channel
from .exception import InvalidSetting


def process_position_channel(layer_data: LayerData):
    """Compute positions from raw data."""
    channel = layer_data.channels.position
    if channel is None:
        raise InvalidSetting("Position channel is not defined!")

    attr = layer_data.data.get(channel.source, None)
    if attr is None:
        raise InvalidSetting(f"Invalid position channel setting: {channel}")

    pts = attr.values
    if len(pts) == 0:
        warnings.warn("Position channel contains 0 records!")
        return pts

    if channel.normalize:
        warnings.warn("Normalization of the position channel is not not supported!")

    if channel.mapping != "identity":
        if callable(channel.mapping):
            pts = np.array([channel.mapping(p) for p in pts])
        else:
            raise InvalidSetting(f"Invalid position channel mapping: {channel.mapping}")
    return pts


def extract_channel_values(channel: Channel, attr: Attribute):
    """Convert raw values stored in `attr` to channel values"""
    values = attr.values

    if channel.normalize:
        min_value = np.amin(values)
        max_value = np.amax(values)
        scale = max_value - min_value
        if scale < 1e-12:
            scale = 1
        values = (values - min_value) / scale

    if channel.mapping != "identity":
        if callable(channel.mapping):
            values = [channel.mappping(c) for c in values]
        else:
            raise InvalidSetting(f"Invalid channel mapping: {channel.mapping}")

    return values


def default_color_channel():
    """Get the default color channel"""
    channel = Channel()
    channel.source = "#FFEDB6"  # TODO: move this to config file.
    return channel


def default_size_channel():
    """Get the default size channel"""
    channel = Channel()
    channel.source = 1.0  # TODO: move to config file
    return channel


def process_color_channel(layer_data: LayerData):
    """Compute colors from raw data."""
    channel = layer_data.channels.color
    if channel is None:
        channel = default_color_channel()

    attr = layer_data.data.get(channel.source, None)
    if attr is None:
        # Constant color.
        colors = [channel.source]
    else:
        colors = extract_channel_values(channel, attr)
    return colors


def process_size_channel(layer_data: LayerData):
    """Compute size/radius from layer setting"""
    channel = layer_data.channels.size
    if channel is None:
        channel = default_size_channel()

    attr = layer_data.data.get(channel.source, None)
    if attr is None:
        # Constant size.
        sizes = [channel.source]
    else:
        sizes = extract_channel_values(channel, attr)

    return sizes
