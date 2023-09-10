import pytest
from hakowan.beta import layer, mark, dataframe, compiler
import lagrange


class TestCompile:
    def test_compile(self):
        data = lagrange.SurfaceMesh()
        l1 = layer.Layer().data(data)
        root = l1.mark(mark.Surface)
        scene = compiler.compile(root)
        assert len(scene.views) == 1
