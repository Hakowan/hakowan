import pytest
from hakowan.common.colormap.colormap import ColorMap
from hakowan.common.to_color import to_color
import numpy as np


class TestColorMap:
    def test_2_colors(self):
        colors = np.vstack([np.zeros(3), np.ones(3)])
        cm = ColorMap(colors)
        assert np.allclose(cm(0).data, colors[0])
        assert np.allclose(cm(1).data, colors[1])

    def test_3_colors(self):
        colors = np.vstack([np.zeros(3), np.ones(3) * 0.2, np.ones(3)])
        cm = ColorMap(colors)
        assert np.allclose(cm(0).data, colors[0])
        assert np.allclose(cm(0.5).data, colors[1])
        assert np.allclose(cm(1).data, colors[2])

    def test_5_colors(self):
        colors = np.vstack(
            [
                np.zeros(3),
                np.ones(3) * 0.25,
                np.ones(3) * 0.5,
                np.ones(3) * 0.75,
                np.ones(3),
            ]
        )
        cm = ColorMap(colors)
        assert np.allclose(cm(0).data, colors[0])
        assert np.allclose(cm(0.5).data, colors[2])
        assert np.allclose(cm(0.5000001).data, colors[2])
        assert np.allclose(cm(1).data, colors[4])

    def test_5_colors_again(self):
        colors = ["#060103", "#B7313C", "#CA8331", "#CAC732", "#F4F2D5"]
        colors = [to_color(c).data for c in colors]
        print(colors)
        cm = ColorMap(colors)
        assert np.allclose(cm(0).data, colors[0])
        assert np.allclose(cm(0.2500001).data, colors[1])
        assert np.allclose(cm(0.5).data, colors[2])
        assert np.allclose(cm(0.5000001).data, colors[2])
        assert np.allclose(cm(0.7500001).data, colors[3])
        assert np.allclose(cm(1).data, colors[4])
