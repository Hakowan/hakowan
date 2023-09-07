import pytest
from hakowan.beta import Mark

class TestMark:
    def test_simple(self):
        m = Mark.Surface
        assert m == Mark.Surface
        assert m != Mark.Curve
