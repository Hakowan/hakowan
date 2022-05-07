import numpy as np
import pytest

import hakowan


@pytest.fixture
def triangle_data_frame():
    """Generate a data frame consists of a single triangle."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    facets = np.array([[0, 1, 2]], dtype=int)
    data = hakowan.DataFrame()
    data.geometry = hakowan.Attribute(values=vertices, indices=facets)
    return data


@pytest.fixture
def quad_data_frame():
    """Generate a data frame consists of two triangles forming a quad."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    facets = np.array([[0, 1, 2], [2, 1, 3]], dtype=int)
    data = hakowan.DataFrame()
    data.geometry = hakowan.Attribute(values=vertices, indices=facets)
    return data
