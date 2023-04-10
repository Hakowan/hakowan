""" Layer module """

from __future__ import annotations  # To allow type hint of the enclosing class.
from typing import Any, Optional, Union
from textwrap import indent
import numpy as np
from .layer_data import LayerData, DataFrame, Mark, ChannelSetting, Transform
from pathlib import Path
import lagrange


class Layer:
    """A layer represent a node in the visualization layer graph."""

    def __init__(self):
        """Construct a `Layer` object."""
        self.layer_data = LayerData()
        self.children: list[Layer] = []

    def data(self, data_frame: Union[Path, str, lagrange.SurfaceMesh]) -> Layer:
        """Specify data sources."""

        if isinstance(data_frame, str):
            data_frame = Path(data_frame)
        if isinstance(data_frame, Path):
            assert data_frame.exists(), f"File {data_frame} does not exist."
            data_frame = lagrange.io.load_mesh(data_frame)

        data_frame = DataFrame(mesh=data_frame)
        data_frame.finalize()
        parent = Layer()
        parent.layer_data.data = data_frame
        parent.children.append(self)
        return parent

    def channel(self, **kwargs) -> Layer:
        """Specify visualization channels."""
        parent = Layer()
        parent.layer_data.channel_setting = ChannelSetting(**kwargs)
        parent.children.append(self)
        return parent

    def mark(self, value: Mark) -> Layer:
        """Specify marks."""
        parent = Layer()
        parent.layer_data.mark = value
        parent.children.append(self)
        return parent

    def transform(self, value: np.ndarray, overwrite: bool = False) -> Layer:
        """Specify layer transform."""
        parent = Layer()
        parent.layer_data.transform = Transform(value, overwrite)
        parent.children.append(self)
        return parent

    def __add__(self, other: Layer) -> Layer:
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
