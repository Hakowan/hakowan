#!/usr/bin/env python
"""WebGL backend Phase 4 demo (Tier-1 feature parity).

Shows the new shader-injection + emitter features:

  * **Isocontour** texture on a sphere whose scalar is the y-coordinate.
  * **Principled** material with a ``ScalarField`` roughness — shiny equator,
    matte poles.
  * **Vector-field arrows** (``end_type='arrow'``, refinement_level=1) on
    triangle centroids.
  * **Point light** emitters illuminating the surfaces (in addition to the
    default envmap).
"""

from __future__ import annotations

import math

import hakowan as hkw
import lagrange
import numpy as np
from hakowan.setup.emitter import Point as PointEmitter, Envmap


def icosphere(refinement: int = 3) -> lagrange.SurfaceMesh:
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


def with_vertex_scalar(
    mesh: lagrange.SurfaceMesh, name: str, values: np.ndarray
) -> lagrange.SurfaceMesh:
    mesh.create_attribute(
        name,
        usage=lagrange.AttributeUsage.Scalar,
        element=lagrange.AttributeElement.Vertex,
        initial_values=values.astype(np.float64),
    )
    return mesh


def with_vertex_vector(
    mesh: lagrange.SurfaceMesh, name: str, values: np.ndarray
) -> lagrange.SurfaceMesh:
    mesh.create_attribute(
        name,
        usage=lagrange.AttributeUsage.Vector,
        element=lagrange.AttributeElement.Vertex,
        initial_values=values.astype(np.float64),
    )
    return mesh


def main() -> None:
    iso_mesh = icosphere(refinement=3)
    iso_mesh = with_vertex_scalar(iso_mesh, "h", iso_mesh.vertices[:, 1])
    iso_layer = (
        hkw.layer(iso_mesh)
        .mark(hkw.mark.Surface)
        .channel(
            material=hkw.material.Diffuse(
                reflectance=hkw.texture.Isocontour(
                    data=hkw.attribute(name="h"),
                    num_contours=8,
                    ratio=0.4,
                    texture1=hkw.texture.Uniform(color="black"),
                    texture2=hkw.texture.Uniform(color="lightsteelblue"),
                )
            )
        )
        .translate(offset=(-2.6, 0, 0))
    )

    rough_mesh = icosphere(refinement=3)
    # Equator (|y| small) = smooth (0); poles = rough (1).
    rough_attr = np.clip(np.abs(rough_mesh.vertices[:, 1]) * 1.3, 0.05, 0.95)
    rough_mesh = with_vertex_scalar(rough_mesh, "rough", rough_attr)
    rough_layer = (
        hkw.layer(rough_mesh)
        .mark(hkw.mark.Surface)
        .channel(
            material=hkw.material.Principled(
                color=hkw.texture.Uniform(color="goldenrod"),
                roughness=hkw.texture.ScalarField(
                    data=hkw.attribute(name="rough")
                ),
                metallic=0.9,
            )
        )
        .translate(offset=(0.0, 0, 0))
    )

    arrow_mesh = icosphere(refinement=1)
    # Vectors point outward from origin; magnitude = 0.18.
    centers = np.asarray(arrow_mesh.vertices)
    vectors = centers * 0.25
    arrow_mesh = with_vertex_vector(arrow_mesh, "vec", vectors)
    arrow_layer = (
        hkw.layer(arrow_mesh)
        .mark(hkw.mark.Curve)
        .channel(
            material=hkw.material.Diffuse(reflectance="crimson"),
            size=0.02,
            vector_field=hkw.channel.VectorField(
                data=hkw.attribute(name="vec"),
                end_type="arrow",
                refinement_level=0,
            ),
        )
        .translate(offset=(2.6, 0, 0))
    )

    root = iso_layer + rough_layer + arrow_layer

    config = hkw.config()
    config.film.width = 1280
    config.film.height = 720
    config.emitters = [
        Envmap(),
        PointEmitter(position=[3, 4, 5], intensity=15.0),
    ]

    result = hkw.render(root, config, filename="webgl_phase4_demo.html", backend="webgl")
    print(f"Wrote {result.path}")


if __name__ == "__main__":
    main()
