import pytest
from hakowan.beta import transform, scale
import copy


class TestTransform:
    def test_filter(self):
        attr = scale.Attribute(name="index")
        t = transform.Filter(data=attr, condition=lambda x: True)
        assert t.data is attr
        assert t.condition(0)
        assert t._child is None

    def test_chaining_and_copy(self):
        attr0 = scale.Attribute(name="index")
        t0 = transform.Filter(data=attr0, condition=lambda x: True)
        attr1 = scale.Attribute(name="curvature")
        t1 = transform.Filter(data=attr1, condition=lambda x: True)
        t1.child = t0

        assert t1.data is attr1
        assert t1.child.data is attr0

        t2 = copy.deepcopy(t1)
        assert t2 is not t1
        assert t2.data is not t1.data
        assert t2.child is not t1.child
        assert t2.child.data is not t1.child.data
