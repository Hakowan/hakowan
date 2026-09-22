#!/usr/bin/env python
"""WebGL backend Phase 2 demo.

Exercises every mark type and the new material features so we can visually
confirm Phase 2 functionality:

  * **Surface** with a ``ScalarField`` colormap (viridis baked to vertex
    colors).
  * **Point** cloud with per-point sizes and viridis-colored spheres.
  * **Curve** drawn as the edges of a small mesh.

Run with ``pixi run python examples/webgl_phase2_demo.py``; open
``webgl_phase2_demo.html`` in any browser.
"""

from __future__ import annotations

import math

import hakowan as hkw
import lagrange
import numpy as np


def build_refined_icosphere(refinement: int = 2) -> lagrange.SurfaceMesh:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ],
        dtype=np.float64,
    )
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    tris = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [2, 11, 10], [4, 5, 11], [9, 1, 5], [8, 7, 1], [6, 10, 7],
            [4, 9, 5], [9, 8, 1], [8, 6, 7], [6, 2, 10], [2, 4, 11],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        ],
        dtype=np.uint32,
    )

    for _ in range(refinement):
        midpoints: dict[tuple[int, int], int] = {}
        verts_l: list[np.ndarray] = list(verts)
        new_tris: list[list[int]] = []

        def mid(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key in midpoints:
                return midpoints[key]
            m = (verts_l[a] + verts_l[b]) / 2.0
            m = m / np.linalg.norm(m)
            verts_l.append(m)
            midpoints[key] = len(verts_l) - 1
            return midpoints[key]

        for v1, v2, v3 in tris.tolist():
            m12 = mid(v1, v2); m23 = mid(v2, v3); m31 = mid(v3, v1)
            new_tris.extend(
                [[v1, m12, m31], [v2, m23, m12], [v3, m31, m23], [m12, m23, m31]]
            )
        verts = np.array(verts_l, dtype=np.float64)
        tris = np.array(new_tris, dtype=np.uint32)

    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(verts)
    mesh.add_triangles(tris)
    # Per-vertex scalar = y coordinate, in [-1, 1].
    mesh.create_attribute(
        "height",
        usage=lagrange.AttributeUsage.Scalar,
        element=lagrange.AttributeElement.Vertex,
        initial_values=verts[:, 1].astype(np.float64),
    )
    return mesh


def build_point_grid() -> lagrange.SurfaceMesh:
    xs, ys, zs = np.mgrid[-1:1:6j, -1:1:6j, -1:1:6j]
    pts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(pts)
    # Per-vertex scalar = distance from origin.
    radii = np.linalg.norm(pts, axis=1)
    mesh.create_attribute(
        "radius",
        usage=lagrange.AttributeUsage.Scalar,
        element=lagrange.AttributeElement.Vertex,
        initial_values=radii.astype(np.float64),
    )
    return mesh


def build_curve_mesh() -> lagrange.SurfaceMesh:
    # An icosahedron just for its edges.
    sphere = build_refined_icosphere(refinement=0)
    return sphere


def main() -> None:
    surface_mesh = build_refined_icosphere(refinement=3)
    point_mesh = build_point_grid()
    curve_mesh = build_curve_mesh()

    surface = (
        hkw.layer(surface_mesh)
        .mark(hkw.mark.Surface)
        .channel(
            material=hkw.material.Diffuse(
                reflectance=hkw.texture.ScalarField(
                    data=hkw.attribute(name="height"), colormap="viridis"
                )
            )
        )
        .translate(offset=(-2.5, 0, 0))
    )

    points = (
        hkw.layer(point_mesh)
        .mark(hkw.mark.Point)
        .channel(
            material=hkw.material.Diffuse(
                reflectance=hkw.texture.ScalarField(
                    data=hkw.attribute(name="radius"), colormap="plasma"
                )
            ),
            size=0.05,
        )
    )

    curves = (
        hkw.layer(curve_mesh)
        .mark(hkw.mark.Curve)
        .channel(material=hkw.material.Diffuse(reflectance="white"))
        .translate(offset=(2.5, 0, 0))
    )

    root = surface + points + curves

    config = hkw.config()
    config.film.width = 1280
    config.film.height = 720

    result = hkw.render(root, config, filename="webgl_phase2_demo.html", backend="webgl")
    print(f"Wrote {result.path}")


if __name__ == "__main__":
    main()
