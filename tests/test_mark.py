import pytest
from hakowan import mark


class TestMark:
    def test_simple(self):
        m = mark.Surface
        assert m == mark.Surface
        assert m != mark.Curve
