import numpy as np
from typing import TypeAlias

ColorLike: TypeAlias = float | int | str | tuple | list
"""Color-like type alias is types that can be converted to a color."""


class Color:
    """A minimal representation of color."""

    @classmethod
    def from_hex(cls, hex_color):
        """Construct color from hex."""
        c = hex_color.lstrip("#")
        assert len(c) == 6
        return Color(
            int(c[0:2], 16) / 255.0,
            int(c[2:4], 16) / 255.0,
            int(c[4:6], 16) / 255.0,
        )

    def __init__(self, red=0.0, green=0.0, blue=0.0):
        """Construct color from RGB."""
        self.color = np.array([red, green, blue])

    def __getitem__(self, i):
        """Get color channel."""
        return self.color[i]

    def __add__(self, other):
        """Add two colors."""
        c = self.color + other.color
        return Color(*c)

    def __mul__(self, scale):
        """Multiply color by a scalar."""
        c = self.color * scale
        return Color(*c)

    def __rmul__(self, scale):
        """Multiply color by a scalar."""
        return self.__mul__(scale)

    def __eq__(self, other):
        """Check if two colors are equal."""
        return np.all(self.color == other.color)

    def __ne__(self, other):
        """Check if two colors are not equal."""
        return not self == other

    def __iter__(self):
        """Iterate over color channels."""
        return self.color.__iter__()

    def __repr__(self):
        """String representation of color."""
        return f"Color ({self.red}, {self.green}, {self.blue})"

    @property
    def red(self):
        """Red channel."""
        return self.color[0]

    @property
    def green(self):
        """Green channel"""
        return self.color[1]

    @property
    def blue(self):
        """Blue channel"""
        return self.color[2]

    @property
    def data(self):
        """Raw data"""
        return self.color
