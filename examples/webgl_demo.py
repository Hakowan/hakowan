#!/usr/bin/env python
"""WebGL/three.js backend demo for hakowan.

Renders a simple icosphere as an interactive HTML page. Open the resulting
``webgl_demo.html`` in any modern browser — no server required (requires
internet on first load for three.js CDN).
"""

import hakowan as hkw
import lagrange
import numpy as np


def make_icosphere() -> lagrange.SurfaceMesh:
    mesh = lagrange.SurfaceMesh()
    vertices = np.array(
        [
            [0, 0, 1],
            [0, 0.894, -0.447],
            [0.851, -0.276, -0.447],
            [-0.851, -0.276, -0.447],
            [0, -0.894, -0.447],
        ],
        dtype=np.float64,
    )
    facets = np.array(
        [
            [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
            [1, 4, 2], [2, 4, 3], [3, 4, 1], [1, 3, 2],
        ],
        dtype=np.uint32,
    )
    mesh.add_vertices(vertices)
    mesh.add_triangles(facets)
    return mesh


def main() -> None:
    print(f"Available backends: {hkw.list_backends()}")

    mesh = make_icosphere()
    layer = (
        hkw.layer(mesh)
        .mark(hkw.mark.Surface)
        .channel(material=hkw.material.Diffuse(reflectance="orange"))
    )

    config = hkw.config()
    config.film.width = 1024
    config.film.height = 768

    result = hkw.render(layer, config, filename="webgl_demo.html", backend="webgl")
    print(f"Wrote interactive viewer to {result.path}")
    print("Open it in your browser (double-click or 'open webgl_demo.html').")


if __name__ == "__main__":
    main()
