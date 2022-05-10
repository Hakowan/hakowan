import pytest
import hakowan

from .test_utils import triangle_data_frame


@pytest.fixture
def base_surface():
    return hakowan.layer(mark=hakowan.SURFACE, data=triangle_data_frame)


@pytest.fixture
def base_point():
    return hakowan.layer(mark=hakowan.POINT, data=triangle_data_frame)


class TestLayer:
    def test_construction(self, base_surface):
        l0 = base_surface
        l1 = l0.mark(hakowan.CURVE)
        l2 = l1.channel(color="red")

        assert l0 in l1.children
        assert l1 in l2.children

        assert l0.layer_data.mark == hakowan.SURFACE
        assert l1.layer_data.mark == hakowan.CURVE
        assert l2.layer_data.mark is None

        assert l0.layer_data.channel_setting.color == None
        assert l1.layer_data.channel_setting.color == None
        assert l2.layer_data.channel_setting.color == "red"

    def test_simple(self, base_surface, base_point):
        l0 = base_surface
        l1 = base_point
        l2 = l0 + l1
        assert l0 in l2.children
        assert l1 in l2.children
