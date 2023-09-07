import pytest
from hakowan.beta import DataFrame

import lagrange

class TestDataFrame():
    def test_empty(self):
        df = DataFrame()
        assert df.mesh is None

    def test_simple(self):
        df = DataFrame()
        df.mesh = lagrange.SurfaceMesh()
        df.mesh.add_vertex([0, 0, 0])
        df.mesh.add_vertex([1, 0, 0])
        df.mesh.add_vertex([1, 1, 0])
        assert df.mesh.num_vertices == 3
        df.mesh.add_triangle(0, 1, 2)
        assert df.mesh.num_facets == 1
