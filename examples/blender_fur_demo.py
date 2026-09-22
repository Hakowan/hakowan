#!/usr/bin/env python
"""Demo: visualize a per-facet vector field as realistic fur flowing along the
field, rendered with the Blender backend (Principled Hair BSDF).

The field is a "longitude flow": the world +Z axis projected onto each face's
tangent plane, so the fur combs from the south pole toward the north pole.
"""

import lagrange
import numpy as np

import hakowan as hkw


def build_sphere(radius: float = 1.0, subdiv: int = 5) -> lagrange.SurfaceMesh:
    base = lagrange.primitive.generate_octahedron()
    return lagrange.primitive.generate_subdivided_sphere(
        base, radius=radius, subdiv_level=subdiv
    )


def project_axis_to_tangent(mesh: lagrange.SurfaceMesh, axis) -> np.ndarray:
    """Per-facet field: project ``axis`` onto each face's tangent plane."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    facets = np.asarray(mesh.facets, dtype=np.int64).reshape(-1, 3)
    p0, p1, p2 = vertices[facets[:, 0]], vertices[facets[:, 1]], vertices[facets[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    axis = np.asarray(axis, dtype=np.float64)
    proj = axis - normals * (normals @ axis)[:, None]
    proj /= np.maximum(np.linalg.norm(proj, axis=1, keepdims=True), 1e-20)
    return proj


def main(output: str = "blender_fur_demo.png"):
    mesh = build_sphere()
    flow = project_axis_to_tangent(mesh, [0.0, 0.0, 1.0])
    mesh.create_attribute(
        "flow",
        element=lagrange.AttributeElement.Facet,
        usage=lagrange.AttributeUsage.Vector,
        initial_values=flow,
    )

    # Skin: a diffuse sphere so the fur has a body underneath.
    skin = hkw.layer(mesh).material("Diffuse", reflectance=[0.35, 0.22, 0.12])

    # Fur: a modest set of guide strands that follow the surface along the field
    # (short streamlines with a gentle lift/curl, so the fur hugs the body),
    # expanded into dense, clumped fur by the Blender backend's Geometry Nodes
    # child-hair modifier. The strands render as native Cycles hair primitives
    # shaded with the Principled Hair BSDF.
    #
    # For a solid-colored coat, pass a constant color to the Hair material, e.g.
    # ``.material("Hair", color=[0.15, 0.35, 0.95])`` for blue fur; otherwise the
    # eumelanin/pheomelanin pigments give a natural (black→brown→red) palette.
    fur = (
        hkw.layer(mesh)
        .transform(
            hkw.transform.Fur(
                vec_field="flow",
                n=5000,
                length=0.16,
                lift=18.0,
                curl=0.15,
                root_radius=0.004,
                randomness=0.45,
                follow_surface=True,
                children=15,
                clump=0.7,
                seed=7,
            )
        )
        .mark("curve")
        .material("Hair", eumelanin=1.6, pheomelanin=0.5)
    )

    config = hkw.config()
    config.film.width = 800
    config.film.height = 800
    config.sampler.sample_count = 64
    config.sensor.location = [2.6, -2.6, 1.6]
    config.sensor.target = [0.0, 0.0, 0.0]
    config.sensor.fov = 35

    hkw.render(skin + fur, config=config, filename=output, backend="blender")
    print(f"Rendered to {output}")


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "blender_fur_demo.png"
    main(out)
