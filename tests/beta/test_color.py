import pytest
from hakowan.beta.common.color import Color, ColorLike
from hakowan.beta.common.to_color import to_color


class TestColor:
    def test_init(self):
        c = Color(0.1, 0.2, 0.3)
        assert c.red == 0.1
        assert c.green == 0.2
        assert c.blue == 0.3

    def test_to_color(self):
        # Float
        c = to_color(0.2)
        assert c.red == 0.2
        assert c.green == 0.2
        assert c.blue == 0.2

        # Hex
        c = to_color("#000102")
        assert c.red == 0
        assert c.green == 1 / 255
        assert c.blue == 2 / 255

        # Color name
        c = to_color("red")
        assert c.red == 1
        assert c.green == 0
        assert c.blue == 0

        # RGB list
        c = to_color([0, 1, 0])
        assert c.red == 0
        assert c.green == 1
        assert c.blue == 0

        # RGB tuple
        c = to_color((0, 1, 0))
        assert c.red == 0
        assert c.green == 1
        assert c.blue == 0
