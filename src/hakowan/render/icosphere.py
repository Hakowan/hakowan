import numpy as np
import lagrange


def refine_triangles(vertices: list, triangles: list):
    """Refine triangles by adding midpoints to edges.

    Args:
        vertices (list): Array of vertices.
        triangles (list): List of triangles.
    """
    new_vertices = vertices
    new_triangles = []

    edge_midpoints: dict[
        tuple[int, int], int
    ] = {}  # Store midpoints of edges to avoid duplication

    def get_midpoint(v1, v2):
        if (v1, v2) in edge_midpoints:
            return edge_midpoints[(v1, v2)]
        elif (v2, v1) in edge_midpoints:
            return edge_midpoints[(v2, v1)]
        else:
            p1 = np.array(vertices[v1])
            p2 = np.array(vertices[v2])
            midpoint = (p1 + p2) / 2
            midpoint = midpoint / np.linalg.norm(midpoint)
            midpoint_index = len(new_vertices)
            new_vertices.append(midpoint)
            edge_midpoints[(v1, v2)] = midpoint_index
            return midpoint_index

    for tri in triangles:
        v1, v2, v3 = tri
        midpoints = [get_midpoint(v1, v2), get_midpoint(v2, v3), get_midpoint(v3, v1)]
        new_triangles.extend(
            [
                [v1, midpoints[0], midpoints[2]],
                [v2, midpoints[1], midpoints[0]],
                [v3, midpoints[2], midpoints[1]],
                [midpoints[0], midpoints[1], midpoints[2]],
            ]
        )

    return new_vertices, new_triangles


def create_icosphere(refinement_level):
    """Generate icosphere centered at the origin with radius 1.

    Args:
        refinement_level (int): Number of times to refine the icosphere.

    Returns:
        lagrange.SurfaceMesh: The generated icosphere mesh.
    """
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array(
        [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ]
    )
    vertices = vertices / np.linalg.norm(vertices, axis=1)[:, None]
    vertices = vertices.tolist()

    triangles = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (2, 11, 10),
        (4, 5, 11),
        (9, 1, 5),
        (8, 7, 1),
        (6, 10, 7),
        (4, 9, 5),
        (9, 8, 1),
        (8, 6, 7),
        (6, 2, 10),
        (2, 4, 11),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
    ]

    for i in range(refinement_level):
        vertices, triangles = refine_triangles(vertices, triangles)

    icosphere = lagrange.SurfaceMesh()
    icosphere.add_vertices(np.array(vertices, dtype=np.float64))
    icosphere.add_triangles(np.array(triangles, dtype=np.uint32))
    icosphere.create_attribute(
        "vertex_normal",
        usage=lagrange.AttributeUsage.Normal,
        initial_values=np.array(vertices, dtype=np.float64),
    )
    return icosphere
