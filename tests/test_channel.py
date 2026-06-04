from hakowan import channel, material, scale, texture


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
        m = material.Diffuse(reflectance=t)
        assert m.reflectance is t

    def test_back_side_keyword(self):
        # Positional front field still works alongside the keyword-only back_side.
        back = material.Diffuse(reflectance="blue")
        m = material.Diffuse("red", back_side=back)
        assert m.reflectance == "red"
        assert m.back_side is back

    def test_back_side_defaults_none(self):
        assert material.Principled(color="gold").back_side is None
