""" Predefined color maps. """

import math
import numpy as np

from ..color import Color


class ColorMap:
    """A color map is a function that linearly interpolate a set of color
    samples."""

    def __init__(self, samples: np.ndarray):
        assert len(samples) >= 2, "Color map must have at least 2 samples."
        self.samples = samples

    def __call__(self, value: float):
        value = max(0.0, min(1.0, value))

        n = len(self.samples) - 1
        i0 = math.floor(n * value)
        i1 = math.ceil(n * value)

        t = n * value - i0
        c0 = self.samples[i0]
        c1 = self.samples[i1]

        c = c0 * (1 - t) + c1 * t
        return Color(*c)
