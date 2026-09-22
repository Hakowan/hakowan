#!/usr/bin/env python
"""WebGL backend Phase 3 demo.

Showcases the new material + geometry features:

  * **Conductor** material (gold preset) on a refined icosphere.
  * **Checkerboard** texture on the cylinder mesh (UVs come from the OBJ).
  * **Disk** points with per-point orientation along the input normals.
  * **Cube** points with size variation.
  * **Tube** curves built around mesh edges using ``size_channel``.

Run with ``pixi run python examples/webgl_phase3_demo.py``; open
``webgl_phase3_demo.html`` in any browser.
"""

from __future__ import annotations

import math
from pathlib import Path

import hakowan as hkw
import lagrange
import numpy as np


HERE = Path(__file__).resolve().parent


def build_refined_icosphere(refinement: int = 3) -> lagrange.SurfaceMesh:
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
    return mesh


def build_disk_point_mesh() -> lagrange.SurfaceMesh:
    n = 96
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    # Per-point normals pointing radially outward in the xy plane.
    normals = np.stack(
        [np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1
    )
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(pts)
    mesh.create_attribute(
        "facing",
        usage=lagrange.AttributeUsage.Vector,
        element=lagrange.AttributeElement.Vertex,
        initial_values=normals.astype(np.float64),
    )
    return mesh


def build_cube_point_mesh() -> lagrange.SurfaceMesh:
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.0, 1.0, size=(80, 3))
    sizes = rng.uniform(0.04, 0.12, size=80)
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(pts)
    mesh.create_attribute(
        "marker_size",
        usage=lagrange.AttributeUsage.Scalar,
        element=lagrange.AttributeElement.Vertex,
        initial_values=sizes.astype(np.float64),
    )
    return mesh


def build_tube_curve_mesh() -> lagrange.SurfaceMesh:
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
    mesh = lagrange.SurfaceMesh()
    mesh.add_vertices(verts)
    mesh.add_triangles(tris)
    # Per-vertex tube radius varying with height.
    radii = 0.02 + 0.04 * (verts[:, 1] - verts[:, 1].min()) / (
        verts[:, 1].max() - verts[:, 1].min()
    )
    mesh.create_attribute(
        "tube_radius",
        usage=lagrange.AttributeUsage.Scalar,
        element=lagrange.AttributeElement.Vertex,
        initial_values=radii.astype(np.float64),
    )
    return mesh


def main() -> None:
    sphere = build_refined_icosphere(refinement=3)

    cylinder_path = HERE.parent / "test_cylinder.obj"

    gold_sphere = (
        hkw.layer(sphere)
        .mark(hkw.mark.Surface)
        .channel(
            material=hkw.material.RoughConductor(material="Au", alpha=0.12)
        )
        .translate(offset=(-2.5, 1.2, 0))
    )

    checker_cylinder = (
        hkw.layer()
        .data(str(cylinder_path))
        .mark(hkw.mark.Surface)
        .channel(
            material=hkw.material.Diffuse(
                reflectance=hkw.texture.Checkerboard(
                    texture1=hkw.texture.Uniform(color="white"),
                    texture2=hkw.texture.Uniform(color="steelblue"),
                    size=6,
                )
            )
        )
        .translate(offset=(2.5, 1.0, 0))
    )

    disk_points = (
        hkw.layer(build_disk_point_mesh())
        .mark(hkw.mark.Point)
        .channel(
            material=hkw.material.Diffuse(reflectance="crimson"),
            size=0.08,
            shape=hkw.channel.Shape(base_shape="disk", orientation="facing"),
        )
        .translate(offset=(-2.5, -1.2, 0))
    )

    cube_points = (
        hkw.layer(build_cube_point_mesh())
        .mark(hkw.mark.Point)
        .channel(
            material=hkw.material.Plastic(diffuse_reflectance="goldenrod"),
            size="marker_size",
            shape=hkw.channel.Shape(base_shape="cube"),
        )
    )

    tube_curves = (
        hkw.layer(build_tube_curve_mesh())
        .mark(hkw.mark.Curve)
        .channel(
            material=hkw.material.Diffuse(reflectance="mediumseagreen"),
            size="tube_radius",
        )
        .translate(offset=(2.5, -1.2, 0))
    )

    root = gold_sphere + checker_cylinder + disk_points + cube_points + tube_curves

    config = hkw.config()
    config.film.width = 1280
    config.film.height = 720

    result = hkw.render(root, config, filename="webgl_phase3_demo.html", backend="webgl")
    print(f"Wrote {result.path}")


if __name__ == "__main__":
    main()
