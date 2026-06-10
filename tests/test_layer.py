import hakowan as hkw
import lagrange


class TestLayer:
    def test_empty_layer(self):
        lyr = hkw.layer()
        assert lyr._spec.data is None
        assert lyr._spec.mark is None
        assert lyr._spec.channels == []
        assert lyr._spec.transform is None
        assert lyr._children == []

    def test_chain_layers(self):
        l0 = hkw.layer()
        mesh = lagrange.SurfaceMesh()
        l1 = l0.data(mesh)
        assert l0 in l1._children

        position = hkw.attribute(name="position")
        l2 = l0.channel(position=hkw.channel.Position(data=position))
        assert l0 in l2._children

        lyr = l1 + l2
        assert lyr._children == [l1, l2]

    def test_juxtapose_operator(self):
        l1 = hkw.layer()
        l2 = hkw.layer()

        # `+` is a plain overlay node.
        overlay = l1 + l2
        assert overlay._layout is None

        # `|` is a juxtaposition node carrying default layout options.
        row = l1 | l2
        assert row._layout is not None
        assert row._children == [l1, l2]
        assert row._layout.axis == 0
        assert row._layout.gap == 0.05
        assert row._layout.normalize is False

    def test_juxtapose_chain(self):
        l1 = hkw.layer()
        l2 = hkw.layer()
        l3 = hkw.layer()
        row = l1 | l2 | l3
        assert row._layout is not None
        # Left-associative: (l1 | l2) | l3.
        assert row._children[1] is l3
        assert row._children[0]._children == [l1, l2]

    def test_juxtapose_method(self):
        l1 = hkw.layer()
        l2 = hkw.layer()
        l3 = hkw.layer()
        row = l1.juxtapose(l2, l3, axis="y", gap=0.25, normalize=True)
        assert row._layout is not None
        assert row._children == [l1, l2, l3]
        assert row._layout.axis == 1
        assert row._layout.gap == 0.25
        assert row._layout.normalize is True

    def test_juxtapose_bad_axis(self):
        import pytest

        with pytest.raises(ValueError):
            hkw.layer().juxtapose(hkw.layer(), axis="w")

    def test_normal(self, triangle):
        mesh = triangle
        attr_id = lagrange.compute_vertex_normal(mesh)
        l0 = hkw.layer().channel(normal=mesh.get_attribute_name(attr_id))
        assert len(l0._spec.channels) == 1
        ch = l0._spec.channels[0]
        assert isinstance(ch, hkw.channel.Normal)
        assert isinstance(ch.data, hkw.attribute)
        assert ch.data.name == mesh.get_attribute_name(attr_id)
