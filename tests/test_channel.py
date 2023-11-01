import pytest
from hakowan import channel, scale, texture


class TestChannel:
    def test_position(self):
        attr = scale.Attribute(name="position")
        c = channel.Position(data=attr)
        assert c.data is attr

    def test_normal(self):
        attr = scale.Attribute(name="normal")
        c = channel.Normal(data=attr)
        assert c.data is attr

    def test_size(self):
        attr = scale.Attribute(name="size")
        c = channel.Size(data=attr)
        assert c.data is attr


class TestMaterial:
    def test_uniform(self):
        t = texture.Uniform(color=(1, 1, 1))
        m = channel.Diffuse(reflectance=t)
        assert m.reflectance is t
