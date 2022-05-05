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


def default_color_channel():
    """Get the default color channel"""
    channel = Channel()
    channel.source = "#FFEDB6"  # TODO: move this to config file.
    return channel


def process_color_channel(layer_data: LayerData):
    """Compute colors from raw data."""
    channel = layer_data.channels.color
    if channel is None:
        channel = default_color_channel()

    attr = layer_data.data.get(channel.source, None)
    if attr is None:
        # Constant color.
        color = channel.source
    else:
        values = attr.values
        if values.ndim == 2:
            values = norm(values, axis=1)
        elif values.ndim != 1:
            raise InvalidSetting(
                f"Invalid dimension in color source channel: {values.shape}"
            )

        if channel.normalize:
            min_value = np.amin(values)
            max_value = np.amax(values)
            scale = max_value - min_value
            if scale < 1e-12:
                scale = 1
            values = (values - min_value) / scale

        if channel.mapping != "identity":
            if callable(channel.mapping):
                color = [channel.mappping(c) for c in values]
            else:
                raise InvalidSetting(
                    f"Invalid color channel mapping: {channel.mapping}"
                )
    return color
