import pytest
import hakowan.beta as hkw

from .asset import triangle, two_triangles


class TestRender:
    def test_render(self, triangle):
        mesh = triangle
        base = hkw.layer.Layer().data(mesh).mark(hkw.mark.Surface)
        config = hkw.config.Config()
        hkw.render(base, config)
