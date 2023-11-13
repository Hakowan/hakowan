import pytest
from hakowan import texture, scale
import numpy as np
from pathlib import Path


class TestTexture:
    def test_uniform(self):
        t = texture.Uniform(color="#000000")
        assert t.color == "#000000"
        t.color = (1, 1, 1)
        assert np.all(t.color == (1, 1, 1))

    def test_scalar_field(self):
        attr = scale.Attribute(name="index")
        t = texture.ScalarField(data=attr)
        assert t.data is attr
        assert t.colormap == "viridis"
        t.colormap = "turbo"
        assert t.colormap == "turbo"

    def test_image(self):
        attr = scale.Attribute(name="uv")
        t = texture.Image(uv=attr, filename=Path("file.png"))
        assert t.uv is attr
        assert t.filename == Path("file.png")

    def test_checkerboard(self):
        attr = scale.Attribute(name="uv")
        t1 = texture.Uniform(color="#000000")
        t2 = texture.Uniform(color="#ffffff")
        t = texture.Checkerboard(uv=attr, texture1=t1, texture2=t2)
        assert t.uv is attr
        assert t.texture1 is t1
        assert t.texture2 is t2

    def test_isocontour(self):
        attr = scale.Attribute(name="uv")
        t1 = texture.Uniform(color="#000000")
        t2 = texture.Uniform(color="#ffffff")
        t = texture.Isocontour(data=attr, texture1=t1, texture2=t2)
        assert t.data is attr
        assert t.texture1 is t1
        assert t.texture2 is t2
        assert t._uv is None
