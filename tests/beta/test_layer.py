import pytest
from hakowan.beta import channel, dataframe, layer, scale
import lagrange


class TestLayer:
    def test_empty_layer(self):
        l = layer.Layer()
        assert l.data is None
        assert l.mark is None
        assert l.channels == []
        assert l.transform is None
        assert l._children == []

    def test_chain_layers(self):
        l0 = layer.Layer()
        mesh = lagrange.SurfaceMesh()
        l0.data = dataframe.DataFrame(mesh=mesh)

        l1 = layer.Layer()
        position = scale.Attribute(name="position")
        l1.channel = channel.Position(data=position)

        l = l0 + l1
        assert l._children == [l0, l1]
