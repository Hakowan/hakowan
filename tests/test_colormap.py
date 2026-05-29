import pytest
from hakowan.common.colormap.colormap import ColorMap
from hakowan.common.colormap.named_colormaps import get_colormap, named_colormaps
from hakowan.common.color import Color
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


class TestGetColormap:
    def test_builtin_name_resolves(self):
        cm = get_colormap("viridis")
        assert cm is named_colormaps["viridis"]

    def test_colorcet_name_resolves(self):
        colorcet = pytest.importorskip("colorcet")
        cm = get_colormap("fire")
        assert cm is not None
        # Endpoints must match the colorcet palette (linearized hex).
        palette = colorcet.palette["fire"]
        assert np.allclose(cm(0.0).data, Color.from_hex(palette[0]).data, atol=1e-6)
        assert np.allclose(cm(1.0).data, Color.from_hex(palette[-1]).data, atol=1e-6)

    def test_builtin_takes_precedence_over_colorcet(self):
        # "coolwarm" exists in both; the curated built-in must win.
        assert get_colormap("coolwarm") is named_colormaps["coolwarm"]

    def test_unknown_name_returns_none(self):
        assert get_colormap("definitely_not_a_colormap_123") is None
