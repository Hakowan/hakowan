import math
import numpy as np
import numpy.typing as npt

from ..color import Color


class ColorMap:
    """A color map is a function that linearly interpolate a set of color
    samples."""

    def __init__(self, samples: npt.NDArray):
        """Construct color map from color samples.

        Args:
            samples: A numpy array of shape (n, 3). Each row is a color sample.
        """
        assert len(samples) >= 2, "Color map must have at least 2 samples."
        self.samples = samples

    def __call__(self, value: float):
        """Evaluate color map at a value between 0 and 1.

        Args:
            value: A value between 0 and 1.

        Returns:
            (Color): The interpolated color.
        """
        value = max(0.0, min(1.0, value))

        n = len(self.samples) - 1
        i0 = math.floor(n * value)
        i1 = math.ceil(n * value)

        t = n * value - i0
        c0 = self.samples[i0]
        c1 = self.samples[i1]

        c = c0 * (1 - t) + c1 * t
        return Color(*c)

    def num_colors(self):
        """Number of color samples stored in this color map.

        Returns:
            (int): Number of colors in the color map.
        """
        return len(self.samples)
