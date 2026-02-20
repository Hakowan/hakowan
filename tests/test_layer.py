import pytest
import hakowan as hkw
from hakowan import channel, dataframe, layer, scale
import lagrange


class TestLayer:
    def test_empty_layer(self):
        l = hkw.layer()
        assert l._spec.data is None
        assert l._spec.mark is None
        assert l._spec.channels == []
        assert l._spec.transform is None
        assert l._children == []

    def test_chain_layers(self):
        l0 = hkw.layer()
        mesh = lagrange.SurfaceMesh()
        l1 = l0.data(mesh)
        assert l0 in l1._children

        position = hkw.attribute(name="position")
        l2 = l0.channel(position=hkw.channel.Position(data=position))
        assert l0 in l2._children

        l = l1 + l2
        assert l._children == [l1, l2]

    def test_normal(self, triangle):
        mesh = triangle
        attr_id = lagrange.compute_vertex_normal(mesh)
        l0 = hkw.layer().channel(normal=mesh.get_attribute_name(attr_id))
        assert len(l0._spec.channels) == 1
        ch = l0._spec.channels[0]
        assert isinstance(ch, hkw.channel.Normal)
        assert isinstance(ch.data, hkw.attribute)
        assert ch.data.name == mesh.get_attribute_name(attr_id)
