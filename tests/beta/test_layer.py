import pytest
from hakowan.beta import channel, dataframe, layer, scale
import lagrange


class TestLayer:
    def test_empty_layer(self):
        l = layer.Layer()
        assert l._spec.data is None
        assert l._spec.mark is None
        assert l._spec.channels == []
        assert l._spec.transform is None
        assert l._children == []

    def test_chain_layers(self):
        l0 = layer.Layer()
        mesh = lagrange.SurfaceMesh()
        l1 = l0.data(mesh)
        assert l0 in l1._children

        position = scale.Attribute(name="position")
        l2 = l0.channel(channel.Position(data=position))
        assert l0 in l2._children

        l = l1 + l2
        assert l._children == [l1, l2]
