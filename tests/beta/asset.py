import pytest
import lagrange
import numpy as np


def add_attributes(mesh: lagrange.SurfaceMesh):
    mesh.create_attribute(
        "vertex_index",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.arange(mesh.num_vertices, dtype=np.uint32),
    )
    mesh.create_attribute(
        "facet_index",
        element=lagrange.AttributeElement.Facet,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.arange(mesh.num_facets, dtype=np.uint32),
    )
    mesh.create_attribute(
        "corner_index",
        element=lagrange.AttributeElement.Corner,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.arange(mesh.num_corners, dtype=np.uint32),
    )
    if mesh.has_edges:
        mesh.create_attribute(
            "edge_index",
            element=lagrange.AttributeElement.Edge,
            usage=lagrange.AttributeUsage.Scalar,
            initial_values=np.arange(mesh.num_edges, dtype=np.uint32),
        )

    mesh.create_attribute(
        "uv",
        element=lagrange.AttributeElement.Indexed,
        usage=lagrange.AttributeUsage.UV,
        initial_values=mesh.vertices[:, :2].copy(),
        initial_indices=mesh.facets,
    )

    mesh.create_attribute(
        "vertex_data",
        element=lagrange.AttributeElement.Vertex,
        usage=lagrange.AttributeUsage.Scalar,
        initial_values=np.array([1, 2, 3], dtype=np.float32),
    )


@pytest.fixture
def triangle():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.eye(3))
    mesh.add_triangle(0, 1, 2)
    add_attributes(mesh)
    return mesh


@pytest.fixture
def two_triangles():
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices([0, 0, 0])
    mesh.add_vertices([1, 0, 0])
    mesh.add_vertices([1, 1, 0])
    mesh.add_vertices([0, 1, 0])
    mesh.add_triangle(0, 1, 2)
    mesh.add_triangle(0, 2, 3)
    add_attributes(mesh)
    return mesh
