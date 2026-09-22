#!/usr/bin/env python
"""Demo: visualize a per-facet cross field on a sphere with evenly-spaced
streamlines, rendered via the Blender backend.

Cross field is constructed by projecting the world +X axis onto each face's
tangent plane. On a sphere this produces a "longitude flow" pattern with two
singularities at the poles.
"""

import lagrange
import numpy as np

import hakowan as hkw


def build_sphere(num_lat: int = 20, num_lng: int = 40, radius: float = 1.0):
    """Latitude-longitude triangulated sphere."""
    verts = []
    for i in range(num_lat + 1):
        theta = np.pi * i / num_lat
        for j in range(num_lng):
            phi = 2 * np.pi * j / num_lng
            verts.append([
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ])

    def vid(i, j):
        return i * num_lng + (j % num_lng)

    faces = []
    for i in range(num_lat):
        for j in range(num_lng):
            a, b = vid(i, j), vid(i, j + 1)
            c, d = vid(i + 1, j + 1), vid(i + 1, j)
            faces.append([a, b, c])
            faces.append([a, c, d])

    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(np.asarray(verts, dtype=np.float64))
    mesh.add_triangles(np.asarray(faces, dtype=np.uint32))
    return mesh


def project_world_x_to_face_tangent(mesh: lagrange.SurfaceMesh) -> np.ndarray:
    """Per-facet cross field: project +X onto each face's tangent plane."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    facets = np.asarray(mesh.facets, dtype=np.int64).reshape(-1, 3)
    p0, p1, p2 = vertices[facets[:, 0]], vertices[facets[:, 1]], vertices[facets[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    x_axis = np.array([1.0, 0.0, 0.0])
    proj = x_axis - normals * (normals @ x_axis)[:, None]
    proj /= np.maximum(np.linalg.norm(proj, axis=1, keepdims=True), 1e-20)
    return proj


def main(output: str = "blender_streamlines_demo.png"):
    mesh = build_sphere()
    cross = project_world_x_to_face_tangent(mesh)
    mesh.create_attribute(
        "cross_field",
        element=lagrange.AttributeElement.Facet,
        usage=lagrange.AttributeUsage.Vector,
        initial_values=cross,
    )

    # Render the background sphere (semi-transparent grey)
    surface_layer = hkw.layer(mesh).material("Plastic", diffuse_reflectance=0.7)

    # Render streamlines as thin tubes on top (50 seeds × 2 arms = up to 100 lines)
    streamline_layer = (
        hkw.layer(mesh)
        .transform(
            hkw.transform.Streamline(
                vec_field="cross_field",
                n=50,
                cross_field=True,
                num_steps=200,
                step_factor=0.4,
                seed=42,
            )
        )
        .mark("curve")
        .channel(size=0.005)
        .material("Diffuse", reflectance=[0.05, 0.05, 0.05])
    )

    config = hkw.config()
    config.film.width = 800
    config.film.height = 800
    config.sampler.sample_count = 64
    config.sensor.location = [2.5, -2.5, 1.5]
    config.sensor.target = [0.0, 0.0, 0.0]
    config.sensor.fov = 35

    hkw.render(surface_layer + streamline_layer, config=config, filename=output, backend="blender")
    print(f"Rendered to {output}")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "blender_streamlines_demo.png"
    main(out)
