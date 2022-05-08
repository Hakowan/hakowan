""" Color module """

import numpy as np


class Color:
    """A minimal representation of color."""

    @classmethod
    def from_hex(cls, hex_color, alpha=1.0):
        """Construct color from hex."""
        c = hex_color.lstrip("#")
        assert len(c) == 6
        return Color(
            int(c[0:2], 16) / 255.0,
            int(c[2:4], 16) / 255.0,
            int(c[4:6], 16) / 255.0,
            alpha,
        )

    def __init__(self, red=0.0, green=0.0, blue=0.0, alpha=1.0):
        self.color = np.array([red, green, blue, alpha])

    def __getitem__(self, i):
        return self.color[i]

    def __add__(self, other):
        c = self.color + other.color
        return Color(*c)

    def __mul__(self, scale):
        c = self.color * scale
        return Color(*c)

    def __rmul__(self, scale):
        return self.__mul__(scale)

    def __eq__(self, other):
        return np.all(self.color == other.color)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return f"Color ({self.red}, {self.green}, {self.blue}, {self.alpha})"

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
    def alpha(self):
        """Alpha channel"""
        return self.color[3]

    @property
    def data(self):
        """Raw data"""
        return self.color
