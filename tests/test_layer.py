import hakowan


class TestLayer:
    def test_simple(self):
        l0 = hakowan.Layer()
        l1 = hakowan.Layer()
        l2 = l0 + l1
        assert l0.parent == l2
        assert l1.parent == l2
        assert l0 in l2.children
        assert l1 in l2.children
