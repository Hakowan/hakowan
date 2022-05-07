""" Layer module """

from __future__ import annotations  # To allow type hint of the enclosing class.
from typing import Any
from textwrap import indent
import numpy as np
from .layer_data import LayerData, DataFrame, Mark, ChannelSetting, Transform


class Layer:
    """A layer represent a node in the visualization layer graph."""

    def __init__(
        self,
        *,
        data: DataFrame = None,
        channel_setting: dict[str, Any] = None,
        mark: Mark = None,
        transform: np.ndarray = None,
    ):
        """Construct a `Layer` object.

        Args:
            data (DataFrame): The 3D data frame to use.
            channel_setting (dict[str, Any]): A dict of channel settings.
            mark (Mark): The type of visualization.
        """
        self.layer_data = LayerData()
        self.children = []

        if data is not None:
            self.layer_data.data = data

        if channel_setting is not None:
            self.layer_data.channel_setting = ChannelSetting(**channel_setting)

        if mark is not None:
            self.layer_data.mark = mark

        if transform is not None:
            self.layer_data.transform = Transform(transform)

    def data(self, data_frame: DataFrame):
        """Specify data sources."""

        parent = Layer(data=data_frame)
        parent.children.append(self)
        return parent

    def channel(self, **kwargs):
        """Specify visualization channels."""
        parent = Layer(channel_setting=kwargs)
        parent.children.append(self)
        return parent

    def mark(self, value: Mark):
        """Specify marks."""
        parent = Layer(mark=value)
        parent.children.append(self)
        return parent

    def transform(self, value: np.ndarray, overwrite: bool = False):
        """Specify layer transform."""
        parent = Layer(transform=Transform(value, overwrite))
        parent.children.append(self)
        return parent

    def __add__(self, other: Layer):
        """Combine two layers together"""
        parent = Layer()
        parent.children = [self, other]
        return parent

    def __repr__(self):
        if len(self.children) > 0:
            children = "".join([indent(repr(c), "| ") for c in self.children])
            children_str = f"{len(self.children)} children:\n{children}"
        else:
            children_str = ""

        return f"{repr(self.layer_data)}\n{children_str}"
