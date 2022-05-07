import numpy as np
import pytest

from hakowan.grammar.layer_data import Mark, ChannelSetting, Transform, LayerData

from .test_utils import triangle_data_frame


@pytest.fixture
def d0(triangle_data_frame):
    return LayerData(mark=Mark.POINT, data=triangle_data_frame)


@pytest.fixture
def d1():
    return LayerData(
        mark=Mark.SURFACE,
        channel_setting=ChannelSetting(position="geometry"),
    )


class TestLayerData:
    def test_construction(self, d0, d1):
        assert d0 != d1
        assert d0.mark == Mark.POINT
        assert d0.data is not None
        assert d0.channel_setting is None
        assert d0.transform is None
        assert d1.data is None
        assert d1.channel_setting.position is not None
        assert d1.channel_setting.color is None
        assert d1.transform is None

    def test_combine(self, d0, d1):
        matrix = np.identity(4)
        matrix[0, 3] = 1
        d0.transform = Transform(matrix)
        d2 = d0 | d1

        assert d2.mark == Mark.SURFACE
        assert d2.channel_setting is not None
        assert d2.channel_setting.position == "geometry"
        assert d2.data is not None
        assert d2.data.geometry.values.shape == (3, 3)
        assert d2.data.geometry.indices.shape == (1, 3)
        assert d2.transform is not None

    def test_transform(self, d0, d1):
        translate = np.array([1, 0, 0])
        rotate = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ]
        )

        ex = np.array([1, 0, 0, 1])
        ey = np.array([0, 1, 0, 1])
        ez = np.array([0, 0, 1, 1])

        d0.transform = Transform()
        d0.transform.translation = translate
        d1.transform = Transform()
        d1.transform.rotation = rotate
        assert np.all(d0.transform.translation == translate)
        assert np.any(d1.transform.translation != translate)
        assert np.any(d0.transform.rotation != rotate)
        assert np.all(d1.transform.rotation == rotate)

        d2 = d0 | d1  # Translate, then rotate.
        assert d2.transform is not None
        assert not d2.transform.overwrite
        assert np.all(np.dot(d2.transform.matrix, ex) == np.array([0, 2, 0, 1]))
        assert np.all(np.dot(d2.transform.matrix, ey) == np.array([-1, 1, 0, 1]))
        assert np.all(np.dot(d2.transform.matrix, ez) == np.array([0, 1, 1, 1]))

        d3 = d1 | d0  # Rotate, then translate.
        assert d3.transform is not None
        assert not d3.transform.overwrite
        assert np.all(np.dot(d3.transform.matrix, ex) == np.array([1, 1, 0, 1]))
        assert np.all(np.dot(d3.transform.matrix, ey) == np.array([0, 0, 0, 1]))
        assert np.all(np.dot(d3.transform.matrix, ez) == np.array([1, 0, 1, 1]))
