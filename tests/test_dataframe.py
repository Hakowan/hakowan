import pytest
from hakowan import dataframe

import lagrange


class TestDataFrame:
    def test_simple(self):
        mesh = lagrange.SurfaceMesh()
        mesh.add_vertex([0, 0, 0])
        mesh.add_vertex([1, 0, 0])
        mesh.add_vertex([1, 1, 0])

        df = dataframe.DataFrame(mesh=mesh)
        assert df.mesh.num_vertices == 3
        df.mesh.add_triangle(0, 1, 2)
        assert df.mesh.num_facets == 1
