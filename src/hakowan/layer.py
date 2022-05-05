""" Visualization Layer """

from __future__ import annotations  # To allow type hint of the enclosing class.
from .layer_data import LayerData, Attribute, Mark


class Layer:
    """A layer represent a node in the visualization layer tree."""

    layer_data = LayerData()
    children = []
    parent = None

    def data(self, attributes: dict[str, Attribute]):
        """Specify data sources."""
        parent = Layer()
        parent.layer_data.data = attributes
        parent.children.append(self)
        self.parent = parent
        return parent

    def channel(self, **kwargs):
        """Specify visualization channels."""
        parent = Layer()
        for key, val in kwargs:
            parent.update_channel(key, val)
        parent.children.append(self)
        self.parent = parent
        return parent

    def mark(self, value: Mark):
        """Specify marks."""
        parent = Layer()
        parent.layer_data.mark = value
        parent.children.append(self)
        self.parent = parent
        return parent

    def __add__(self, other: Layer):
        """Combine two layers together"""
        parent = Layer()
        parent.children = [self, other]
        self.parent = parent
        other.parent = parent
        return parent

    def __repr__(self):
        return (
            f"layer_data: {repr(self.layer_data)}\nnum children: {len(self.children)}"
        )

    def update_channel(self, key, val):
        """Update channel specified by `key` to be `val`."""
        if hasattr(self.layer_data.channels, key):
            setattr(self.layer_data.channels, key, val)
        else:
            raise NotImplementedError(f"Unsupported channel: {key}")
