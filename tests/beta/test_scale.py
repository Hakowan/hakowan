import pytest
from hakowan.beta import scale
import numpy as np


class TestAttribute:
    def test_attribute(self):
        a = scale.Attribute(name="name")
        assert a.scale is None
        a.name = "name"

        a.scale = scale.Uniform(factor=1)
        assert a.scale.factor == 1


class TestScale:
    def test_normalize(self):
        s = scale.Normalize(range_min=np.zeros(3), range_max=np.ones(3))
        assert s._child is None
        assert np.all(s.range_min == [0, 0, 0])
        assert np.all(s.range_max == [1, 1, 1])

        s.domain_min= 0
        s.domain_max= 1
        assert s.domain_min== 0
        assert s.domain_max== 1

    def test_log(self):
        s = scale.Log(base=2)
        assert s._child is None
        assert s.base == 2

    def test_uniform(self):
        s = scale.Uniform(factor=2)
        assert s._child is None
        assert s.factor == 2

    def test_offset(self):
        a = scale.Attribute(name="index")
        assert a.name == "index"
        assert a.scale is None
        s = scale.Offset(offset=a)
        assert s._child is None
        assert s.offset is a

        a.name = "curvature"
        assert s.offset.name == "curvature"


    def test_chaning(self):
        s1 = scale.Normalize(range_min=np.zeros(3), range_max=np.ones(3))
        s2 = scale.Log(base=10)
        s1._child = s2
        assert s1._child.base == 10
        assert s2._child is None
